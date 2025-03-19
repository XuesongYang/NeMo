import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest


class AudioDataset(Dataset):
    def __init__(self, record_list, base_audio_dir, sample_rate=16000):
        self.sample_rate = sample_rate
        self.combined_file_list = []
        for record in record_list:
            audio_filepath = os.path.join(base_audio_dir, record["audio_filepath"])
            offset = record.get("offset", None)
            if offset is None:
                duration = None
            else:
                duration = record.get("duration", None)
            self.combined_file_list.append(
                {
                    "rel_file_path": record["audio_filepath"],
                    "audio_file_path": audio_filepath,
                    "offset": offset,
                    "duration": duration,
                    "speaker": record["speaker"],
                    "start": record["start"],
                    "end": record["end"],
                }
            )

    def __len__(self):
        return len(self.combined_file_list)

    def get_wav_from_filepath(self, audio_filepath, offset_in_sec=0, duration_in_sec=None):
        """
        :param audio_filepath: path of file to load.
        :param offset_in_sec: offset in seconds when loading audio
        :param duration_in_sec: duration in seconds when loading audio
        """
        features = AudioSegment.from_file(
            audio_file=audio_filepath,
            target_sr=self.sample_rate,
            int_values=False,
            offset=offset_in_sec,
            duration=duration_in_sec,
            trim=False,
            channel_selector="average",
        )
        audio_samples = features.samples
        audio = torch.tensor(audio_samples)
        audio_length = torch.tensor(audio.size(0)).long()
        return audio, audio_length

    def __getitem__(self, idx):
        item = self.combined_file_list[idx]
        audio_file_path = item["audio_file_path"]
        rel_audio_path = item["rel_file_path"]
        offset = item["offset"]
        duration = item["duration"]
        speaker = item["speaker"]
        start = item["start"]
        end = item["end"]
        try:
            audio, audio_length = self.get_wav_from_filepath(audio_file_path, offset_in_sec=offset, duration_in_sec=duration)
        except Exception as e:
            print(f"******** {audio_file_path}: {str(e)} *******")
            audio, audio_length = None, None

        return {
            "audio": audio,
            "audio_length": audio_length,
            "audio_file_path": audio_file_path,
            "rel_audio_path": rel_audio_path,
            "offset": offset,
            "duration": duration,
            "speaker": speaker,
            "start": start,
            "end": end,
        }

    def collate_fn(self, batch):
        audios_padded = []
        audio_lengths = []
        audio_file_paths = []
        rel_audio_paths = []
        max_audio_length = max(item["audio_length"].item() for item in batch if item["audio"] is not None)
        for item in batch:
            if item["audio"] is None:
                continue
            audio = torch.nn.functional.pad(item["audio"], (0, max_audio_length - item["audio"].size(0)), value=0)
            audios_padded.append(audio)
            audio_lengths.append(item["audio_length"])
            rel_audio_paths.append(item["rel_audio_path"])
            audio_file_paths.append(item["audio_file_path"])

        return {
            "audios": torch.stack(audios_padded),
            "audio_lengths": torch.stack(audio_lengths),
            "audio_file_paths": audio_file_paths,
            "rel_audio_paths": rel_audio_paths,
            "offset": [item["offset"] for item in batch],
            "duration": [item["duration"] for item in batch],
            "speaker": [item["speaker"] for item in batch],
            "start": [item["start"] for item in batch],
            "end": [item["end"] for item in batch],
        }


class EmbeddingExtractor(pl.LightningModule):
    def __init__(self, out_dir):
        super().__init__()
        self.sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        self.sv_model.eval()
        self.out_dir = Path(out_dir)

    def forward(self, batch):
        with torch.no_grad():
            _, speaker_embeddings = self.sv_model.forward(
                input_signal=batch['audios'], input_signal_length=batch['audio_lengths']
            )
            return speaker_embeddings

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        speaker_embeddings = self(batch)
        for i, rel_path in enumerate(batch["rel_audio_paths"]):
            speaker_embedding = speaker_embeddings[i]
            start = round(batch["start"][i], 2)
            end = round(batch["end"][i], 2)
            speaker = batch["speaker"][i]
            # output file basename format: yt_tts_en_us_15_02_2024-18_03_38_audios__{speaker}__{start}_{end}.pt
            output_rel_path = "_".join(Path(rel_path).parent.parts) + f"__{speaker}__" + f"{start}_{end}.pt"
            out_file_path = self.out_dir / output_rel_path
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(speaker_embedding.cpu().type(torch.float32), out_file_path)
        return None


if __name__ == "__main__":
    """
    Usage:
    python scripts/t5tts/speaker_embedding_extraction.py \
        --manifest /home/shehzeenh/Code/NewT5TTS/manifests/libri360_val.json \
        --audio_base_dir /Data/LibriTTS \
        --save_dir /Data/tempspeakerembeddings \
        --batch_size 16 \
        --devices 2 \
        --num_nodes 1 
    
    example nvyt_v1.1 entry
    {
      "audio_filepath": "yt_tts_en_us_15_02_2024-18_03_38/audios/Pbz0INTbQAs.wav",
      "duration": 7.44,
      "text": "So we'll focus a little bit more on the mentorship program and go over what the task force is doing.",
      "start": 348.64,
      "end": 356.08,
      "offset": 348.64,
      "speaker": "Pbz0INTbQAs_SPEAKER_01"
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--audio_base_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    records = read_manifest(args.manifest)
    records.sort(key=lambda x: x["duration"], reverse=True)

    trainer = Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=args.num_nodes,
        log_every_n_steps=1,
        max_epochs=1,
        logger=False,
    )

    embedding_extractor = EmbeddingExtractor(args.save_dir)

    dataset = AudioDataset(
        record_list=records,
        base_audio_dir=args.audio_base_dir,
        sample_rate=16000,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    trainer.predict(embedding_extractor, dataloaders=dataloader)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print("Done")
