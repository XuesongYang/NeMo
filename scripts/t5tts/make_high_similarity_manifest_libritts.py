import argparse
import copy
import os
import random
from pathlib import Path
from collections import defaultdict
from operator import itemgetter
import torch
import tqdm
import multiprocessing

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir


def find_and_add_best_candidate(entry):
    """
    Find the best candidate for a given record from a list of candidate records, in terms of cosine similarity of embeddings.
    Returns the best similarity and the best candidate record.
    """
    _record, _candidate_records, embeddings_save_dir = entry
    record_audio_file_path = _record['audio_filepath']
    audio_file_basename = os.path.basename(record_audio_file_path).split(".")[0]
    _parts = audio_file_basename.split("_")
    record_embedding_fp = os.path.join(embeddings_save_dir, f"{_parts[0]}_{_parts[1]}_{audio_file_basename}.pt")
    record_embedding = torch.load(record_embedding_fp).cuda()

    cand_embeddings = []
    for candidate_record in _candidate_records:
        candidate_audio_file_path = candidate_record['audio_filepath']
        cand_audio_basename = os.path.basename(candidate_audio_file_path).split(".")[0]
        cand_parts = cand_audio_basename.split("_")
        cand_embedding_fp = os.path.join(embeddings_save_dir, f"{cand_parts[0]}_{cand_parts[1]}_{cand_audio_basename}.pt")
        cand_embedding = torch.load(cand_embedding_fp)
        cand_embeddings.append(cand_embedding)

    cand_embeddings = torch.stack(cand_embeddings)
    cand_embeddings = cand_embeddings.cuda()

    with torch.no_grad():
        similarities = torch.nn.functional.cosine_similarity(record_embedding.unsqueeze(0), cand_embeddings, dim=1)

    similarity_and_records = []
    for cidx, candidate_record in enumerate(_candidate_records):
        similarity_and_records.append((similarities[cidx].item(), candidate_record))

    similarity_and_records.sort(key=lambda x: x[0], reverse=True)
    _best_similarity, _best_candidate_record = similarity_and_records[0]

    _record.update(
        {
            "context_speaker_similarity": _best_similarity,
            "context_audio_filepath": _best_candidate_record["audio_filepath"],
            "context_audio_duration": _best_candidate_record["duration"],
        }
    )

    return _record


if __name__ == "__main__":
    """
    python scripts/t5tts/make_high_similarity_manifest.py \
        --manifest /home/shehzeenh/Code/NewT5TTS/manifests/libri360_val.json \
        --audio_base_dir /Data/LibriTTS \
        --embeddings_save_dir /Data/tempspeakerembeddings \
        --n_candidates_per_record 100 \
        --context_min_duration 5.0 \
        --similarity_threshold 0.6 ;
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str)
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("--audio_base_dir", type=str)
    parser.add_argument("--embeddings_save_dir", type=str)
    parser.add_argument("--n_candidates_per_record", type=int, default=100)
    parser.add_argument("--context_min_duration", type=float, default=5.0)
    # parser.add_argument("--similarity_threshold", type=float, default=0.6)
    parser.add_argument("--cpu_cores", type=int, default=8)
    args = parser.parse_args()

    print("Reading manifest...")
    records = read_manifest(args.manifest)

    print("Preparing context audio candidates...")
    speaker_records = defaultdict(list)
    # LibriTTS: {speaker}/{chapter}/{speaker}_{chapter}_{utterance}_{sub_utterance}.wav
    # re-organize records into a dict like {"some_speaker_id": list()}
    for record in tqdm.tqdm(records):
        parts = os.path.basename(record["audio_filepath"]).rsplit(".", 1)[0].split("_")
        speaker = parts[0]
        chapter, utterance, sub_utterance = map(int, parts[1:])
        speaker_records[speaker].append((chapter, utterance, sub_utterance, record))

    # Sort context audio candidates in the order of (chapter, utterance, sub_utterance)
    for speaker in speaker_records:
        speaker_records[speaker].sort(key=itemgetter(0, 1, 2))
        # replace tuples with just the record
        speaker_records[speaker] = [item[-1] for item in speaker_records[speaker]]

    # choose limited amount of candidates
    data_list = list()
    for record in tqdm.tqdm(records):
        speaker = os.path.basename(record["audio_filepath"]).split("_", 1)[0]

        candidates = list()
        for item in speaker_records[speaker]:
            if item != record and item["duration"] < args.context_min_duration:
                continue
            candidates.append(item)

        if len(candidates) == 0:
            continue

        if len(candidates) == 1 and record == candidates[0]:
            continue

        record_pos = candidates.index(record)
        start = max(0, record_pos - int(args.n_candidates_per_record / 2))
        end = min(len(candidates), args.n_candidates_per_record - start + 1)
        final_candidates = candidates[start:record_pos] + candidates[record_pos + 1:end]
        data_list.append((record, final_candidates, args.embeddings_save_dir))

    # find high similar context audio using multiple cpu cores.
    pool = multiprocessing.Pool(processes=args.cpu_cores)
    records_new = pool.map(find_and_add_best_candidate, data_list)
    pool.close()
    pool.join()

    #     audio_filepaths = []
    # for record in records:
    #     audio_filepaths.append(os.path.join(args.audio_base_dir, record['audio_filepath']))
    #     if record['speaker'] not in speakerwise_records:
    #         speakerwise_records[record['speaker']] = []
    #     speakerwise_records[record['speaker']].append(record)
    # base_dir_for_file_id = get_base_dir(audio_filepaths)
    #
    # filtered_records = []
    # for ridx, record in enumerate(tqdm.tqdm(records)):
    #     speaker_records = speakerwise_records[record['speaker']]
    #     random.shuffle(speaker_records)
    #     candidate_records = []
    #     for speaker_record in speaker_records:
    #         if (
    #             speaker_record['audio_filepath'] != record['audio_filepath']
    #             and speaker_record['duration'] >= args.context_min_duration
    #         ):
    #             candidate_records.append(speaker_record)
    #
    #     if len(candidate_records) == 0:
    #         # Only one record for this speaker
    #         continue
    #
    #     best_candidate_similarity, best_candidate_record = find_best_candidate(
    #         record, candidate_records, args.audio_base_dir, base_dir_for_file_id, args.embeddings_save_dir
    #     )
    #     if best_candidate_similarity > args.similarity_threshold:
    #         record_copy = copy.deepcopy(record)
    #         record_copy['context_speaker_similarity'] = best_candidate_similarity
    #         record_copy['context_audio_filepath'] = best_candidate_record['audio_filepath']
    #         record_copy['context_audio_duration'] = best_candidate_record['duration']
    #         filtered_records.append(record_copy)
    # do not filter by min SSIM. Better to process later to get knowledge how much records were discarded.
    out_manifest_path = args.manifest.replace(".json", "_withHighSimilarContextAudio.json")
    write_manifest(out_manifest_path, records_new)
    print("Length of original manifest: ", len(records))
    print("Length of filtered manifest: ", len(records_new))
    print("Written to ", out_manifest_path)
