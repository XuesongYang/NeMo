import argparse
import json
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
from nemo.core.classes import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            oldest = next(iter(self.cache))
            del self.cache[oldest]


class AudioDataset(Dataset):
    def __init__(
        self,
        data,
        min_duration=1.0,
        max_duration=22.0,
        sample_rate=24000,
        pad_multiple=320,
        max_open_files=100,
    ):
        self.data = [record for record in data if min_duration <= record['end'] - record['start'] <= max_duration]
        self.sample_rate = sample_rate
        self.pad_multiple = pad_multiple
        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])
        self.file_handles = LRUCache(max_open_files)

    def __len__(self):
        return len(self.data)

    def _get_wav_segment(self, audio_filepath, start, end):
        try:
            sound_file = self.file_handles.get(audio_filepath)
            if sound_file is None:
                sound_file = sf.SoundFile(audio_filepath)
                self.file_handles.put(audio_filepath, sound_file)

            sample_rate = sound_file.samplerate
            start_frame = int(start * sample_rate)
            num_frames = int((end - start) * sample_rate)

            sound_file.seek(start_frame)
            audio_segment = sound_file.read(frames=num_frames, dtype='float32')

            if audio_segment.ndim > 1:
                audio_segment = audio_segment.mean(axis=1)

            audio_segment = torch.from_numpy(audio_segment).float()

            if sample_rate != self.sample_rate:
                audio_segment = torchaudio.functional.resample(audio_segment, sample_rate, self.sample_rate)

            return audio_segment
        except Exception as e:
            error_message = f"Error processing file {audio_filepath}: {str(e)}"
            logging.error(error_message)
            return None

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

        audio = self._get_wav_segment(sample["audio_filepath"], sample["start"], sample["end"])

        if audio is None:
            return None

        return {
            "audio": audio,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "audio_filepath": sample["audio_filepath"],
            "start": sample["start"],
            "end": sample["end"],
            "sentence": sample["sentence"],
            "speaker": sample["speaker"],
            "duration": sample["end"] - sample["start"],
        }

    def __del__(self):
        for handle in self.file_handles.cache.values():
            handle.close()


def load_existing_manifest(manifest_path):
    existing_records = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                key = (record['audio_filepath'], record['start_time'], record['end_time'])
                existing_records[key] = record
    return existing_records


def get_files_to_process(manifest_paths, existing_records):
    files_to_process = []
    for manifest_path in manifest_paths:
        with open(manifest_path, "r") as f:
            for line in f:
                record = json.loads(line)
                key = (record['audio_filepath'], record['start'], record['end'])
                if key not in existing_records:
                    files_to_process.append(record)
    return files_to_process


def save_problematic_files(problematic_files, problematic_files_path, mode='w'):
    with open(problematic_files_path, mode) as f:
        for file_info in problematic_files:
            f.write(f"{file_info}\n")
    logging.info(f"Saved/Updated problematic files list at {problematic_files_path}")


def pad_sequence(batch, pad_multiple=320):
    max_len = max(item.size(0) for item in batch)
    padded_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
    padded_batch = []
    for item in batch:
        padded_item = F.pad(item, (0, padded_len - item.size(0)))
        padded_batch.append(padded_item)
    return torch.stack(padded_batch)


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    audio = [item['audio'] for item in batch]
    audio_lengths = torch.tensor([len(x) for x in audio])
    audio_padded = pad_sequence(audio)

    return {'audio': audio_padded, 'audio_len': audio_lengths, 'items': batch}


def update_manifest(manifest_path, new_records):
    existing_records = load_existing_manifest(manifest_path)

    with open(manifest_path, 'a') as f:
        for record in new_records:
            key = (record['audio_filepath'], record['start_time'], record['end_time'])
            if key not in existing_records:
                json.dump(record, f)
                f.write('\n')

    logging.info(f"Updated manifest at {manifest_path} with {len(new_records)} new records.")


def main():
    parser = argparse.ArgumentParser(description='Create speech codes from audio segments')
    parser.add_argument('--manifest_paths', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--codec_model_path', type=str, required=True)
    parser.add_argument('--codec_bw', type=float, default=6.0)
    parser.add_argument('--max_open_files', type=int, default=100)
    parser.add_argument(
        '--save_interval',
        type=int,
        default=50000,
        help='Number of samples to process before saving codes and updating manifest',
    )
    args = parser.parse_args()

    problematic_files_path = os.path.join(args.out_dir, 'problematic_files.txt')

    codec_model = AudioCodecModel.restore_from(args.codec_model_path)
    # codec_model.to('cuda')
    codec_model.eval()
    codec_model_sample_rate = 24000  # TODO @xueyang: why not 22050?
    codec_model_downsampling_factor = 1024
    print("done loading")

    # Load existing records
    manifest_dir = os.path.join(args.out_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    phoneme_tts_manifest_path = os.path.join(manifest_dir, f"{args.dataset_name}_phoneme_tts.json")
    sentencepiece_tts_manifest_path = os.path.join(manifest_dir, f"{args.dataset_name}_sentencepiece_tts.json")
    existing_records = load_existing_manifest(phoneme_tts_manifest_path)
    existing_records.update(load_existing_manifest(sentencepiece_tts_manifest_path))

    # Get files to process
    files_to_process = get_files_to_process([args.manifest_paths], existing_records)
    logging.info(f"Found {len(files_to_process)} new files to process.")

    # Create a new dataset with only the files to process
    dataset = AudioDataset(
        data=files_to_process,
        sample_rate=codec_model_sample_rate,
        pad_multiple=int(codec_model_downsampling_factor),
        max_open_files=args.max_open_files,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=8,
    )

    _exp_name = f"{args.dataset_name}_{args.codec_bw}bw"
    codec_base_dir = os.path.join(args.out_dir, "codecs")
    os.makedirs(codec_base_dir, exist_ok=True)

    problematic_files = []
    samples_processed = 0
    new_phoneme_records = []
    new_sentencepiece_records = []

    for batch in tqdm(dataloader):
        if batch is None:
            continue

        audio_batch = batch['audio'].cuda()
        audio_len_batch = batch['audio_len'].cuda()

        with torch.no_grad():
            codec_codes, _ = codec_model.encode(audio=audio_batch, audio_len=audio_len_batch)

        for i, item in enumerate(batch['items']):
            codec_len = math.ceil(audio_len_batch[i].item() / codec_model_downsampling_factor)
            sample_codec_codes = codec_codes[i][:, :codec_len]

            example_name = item['rel_audio_path_as_text_id']
            start_time = item['start']
            end_time = item['end']
            speaker_name = item['speaker']

            # Create speaker-specific folder
            speaker_dir = os.path.join(codec_base_dir, _exp_name, speaker_name)
            os.makedirs(speaker_dir, exist_ok=True)

            # Save codec with speaker name in the filename
            target_codec_filepath = os.path.join(
                speaker_dir, f"target_codes_{speaker_name}_{example_name}__{start_time:.2f}_{end_time:.2f}.pt"
            )
            try:
                torch.save(sample_codec_codes.cpu().type(torch.int16), target_codec_filepath)
            except Exception as e:
                error_message = f"Error saving codec for file {item['audio_filepath']}: {str(e)}"
                logging.error(error_message)
                problematic_files.append(error_message)
                continue

            # Prepare new records for manifests
            tts_record = {
                "audio_filepath": item['audio_filepath'],
                "text": item['sentence'],
                "question": f"Text to speech this {item['sentence']}",
                "answer": target_codec_filepath,
                "context": "",
                "question_type": "TEXT",
                "answer_type": "AUDIOCODEC",
                "context_type": "REFSPEAKERCODEC",
                "context_duration": 0,
                "answer_duration": item['duration'],
                "taskname": "squad",
                "speaker": speaker_name,
                "start_time": start_time,
                "end_time": end_time,
            }

            phoneme_tts_record = {key: value for key, value in tts_record.items()}
            phoneme_tts_record["question"] = phoneme_tts_record["question"].replace(
                "Text to speech this", "Phoneme TTS"
            )

            new_phoneme_records.append(phoneme_tts_record)
            new_sentencepiece_records.append(tts_record)

            samples_processed += 1

            # Update manifests and log problematic files at intervals
            if samples_processed % args.save_interval == 0:
                update_manifest(phoneme_tts_manifest_path, new_phoneme_records)
                update_manifest(sentencepiece_tts_manifest_path, new_sentencepiece_records)
                save_problematic_files(problematic_files, problematic_files_path, mode='a')
                new_phoneme_records = []
                new_sentencepiece_records = []
                problematic_files = []

    # Update manifests and save any remaining problematic files
    if new_phoneme_records:
        update_manifest(phoneme_tts_manifest_path, new_phoneme_records)
    if new_sentencepiece_records:
        update_manifest(sentencepiece_tts_manifest_path, new_sentencepiece_records)
    if problematic_files:
        save_problematic_files(problematic_files, problematic_files_path, mode='a')

    logging.info(f"Processed {samples_processed} new samples.")


if __name__ == '__main__':
    main()
