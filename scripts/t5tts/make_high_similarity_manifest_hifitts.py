import argparse
import multiprocessing
import os
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import torch
import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


def find_and_add_best_candidate(entry):
    """
    Find the best candidate for a given record from a list of candidate records, in terms of cosine similarity of embeddings.
    Returns best candidate record.
    """
    _record, _candidate_records, _embeddings_save_dir = entry
    record_rel_audio_filepath = _record['audio_filepath']
    record_embedding_fp = (_embeddings_save_dir / record_rel_audio_filepath).with_suffix(".pt")
    record_embedding = torch.load(record_embedding_fp)
    # Code is faster on CPU for small number of candidate records
    # record_embedding = record_embedding.cuda()

    cand_embeddings = []
    for candidate_record in _candidate_records:
        candidate_rel_audio_filepath = candidate_record['audio_filepath']
        cand_embedding_fp = (_embeddings_save_dir / candidate_rel_audio_filepath).with_suffix(".pt")
        cand_embedding = torch.load(cand_embedding_fp)
        cand_embeddings.append(cand_embedding)

    cand_embeddings = torch.stack(cand_embeddings)
    # cand_embeddings = cand_embeddings.cuda()

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
        --embeddings_save_dir /Data/tempspeakerembeddings \
        --cpu_cores 8 \
        --n_candidates_per_record 100 \
        --context_min_duration 5.0 \
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--embeddings_save_dir", type=str)
    parser.add_argument("--cpu_cores", type=int, default=8)
    parser.add_argument("--n_candidates_per_record", type=int, default=100)
    parser.add_argument("--context_min_duration", type=float, default=5.0)
    args = parser.parse_args()

    manifest = args.manifest
    context_min_duration = args.context_min_duration
    embeddings_save_dir = Path(args.embeddings_save_dir)
    n_candidates_per_record = args.n_candidates_per_record
    cpu_cores = args.cpu_cores

    print("Reading manifest...")
    records = read_manifest(args.manifest)

    print(f"Processing {len(records)} records...")

    speaker_records = defaultdict(list)
    # HiFiTTS: audio/{speaker}_other/{book}/{chapter}_{utterance}.wav
    # re-organize records into a dict like {"some_speaker_id": list()}
    for record in tqdm.tqdm(records):
        parts = record["audio_filepath"].split("/")
        speaker = parts[1].split("_")[0]
        book = int(parts[2])
        chapter = parts[3].rsplit("_", 1)[0]
        utterance = int(parts[3].rsplit("_", 1)[1].split(".")[0])
        speaker_records[speaker].append((book, chapter, utterance, record))

    # Sort context audio candidates in the order of (book, chapter, utterance)
    for speaker in speaker_records:
        speaker_records[speaker].sort(key=itemgetter(0, 1, 2))
        # replace tuples with just the record
        speaker_records[speaker] = [item[-1] for item in speaker_records[speaker]]

    # choose limited amount of candidates
    data_list = list()
    for record in tqdm.tqdm(records):
        speaker = record["audio_filepath"].split("/")[1].split("_")[0]

        candidates = list()
        for item in speaker_records[speaker]:
            if item != record and item["duration"] < context_min_duration:
                continue
            candidates.append(item)

        if len(candidates) == 0:
            continue

        if len(candidates) == 1 and record == candidates[0]:
            continue

        record_pos = candidates.index(record)
        start = max(0, record_pos - int(n_candidates_per_record / 2))
        end = min(len(candidates), n_candidates_per_record - start + 1)
        final_candidates = candidates[start:record_pos] + candidates[record_pos + 1 : end]
        data_list.append((record, final_candidates, embeddings_save_dir))

    # find high similar context audio using multiple cpu cores.
    pool = multiprocessing.Pool(processes=cpu_cores)
    records_new = pool.map(find_and_add_best_candidate, data_list)
    pool.close()
    pool.join()

    # do not filter by min SSIM. Better add such filter during model training.
    out_manifest_path = manifest.replace(".json", f"_withContextAudioMinDur{int(context_min_duration)}.json")
    write_manifest(out_manifest_path, records_new)
    print("Length of original manifest: ", len(records))
    print("Length of filtered manifest: ", len(records_new))
    print("Written to ", out_manifest_path)
