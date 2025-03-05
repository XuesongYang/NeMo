import argparse
import gc
import json
import multiprocessing
import os
from collections import defaultdict
from itertools import islice
from operator import itemgetter
from pathlib import Path

import torch
import tqdm
from torch.cuda import empty_cache

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

_worker_emb_dir = None


def init_worker(embeddings_save_dir):
    """Initialize worker with embeddings directory"""
    global _worker_emb_dir
    _worker_emb_dir = Path(embeddings_save_dir)


def chunked_data_generator(data, chunk_size=1000):
    """Generate data in chunks to reduce memory pressure"""
    _it = iter(data)
    while _chunk := list(islice(_it, chunk_size)):
        yield _chunk


def find_and_add_best_candidate(entry):
    """
    Find the best candidate for a given record from a list of candidate records that has max cosine similarity.
    Returns best candidate record.
    """
    global _worker_emb_dir
    _record, _candidate_records = entry

    # Load record embedding
    record_rel_audio_filepath = _record['audio_filepath']
    record_embedding_fp = (_worker_emb_dir / record_rel_audio_filepath).with_suffix(".pt")
    record_embedding = torch.load(record_embedding_fp, map_location="cpu")

    cand_embeddings = []
    for candidate_record in _candidate_records:
        candidate_rel_audio_filepath = candidate_record['audio_filepath']
        cand_embedding_fp = (_worker_emb_dir / candidate_rel_audio_filepath).with_suffix(".pt")
        cand_embedding = torch.load(cand_embedding_fp, map_location="cpu")
        cand_embeddings.append(cand_embedding)

    # Compute similarities
    cand_embeddings = torch.stack(cand_embeddings)
    with torch.no_grad():
        similarities = torch.nn.functional.cosine_similarity(record_embedding.unsqueeze(0), cand_embeddings, dim=1)

    max_sim, max_idx = torch.max(similarities, dim=0)
    _best_similarity = max_sim.item()
    _best_candidate_record = _candidate_records[max_idx]

    # Explicit memory cleanup
    del record_embedding, cand_embeddings, similarities
    empty_cache() if torch.cuda.is_available() else gc.collect()

    # Update record
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

    print("Reading manifest...")
    records = read_manifest(args.manifest)

    # Grouping speakers
    print(f"Organizing speaker records...")
    speaker_records = defaultdict(list)
    # HiFiTTS-2: {speaker_id}/{book_id}/{speaker_id}_{book_id}_{chapter_str}_{utterance_id}.wav
    # re-organize records into a dict like {"some_speaker_id": list()}
    for record in tqdm.tqdm(records):
        parts = os.path.basename(record["audio_filepath"]).rsplit(".", 1)[0].split("_")
        speaker = parts[0]
        book = int(parts[1])
        chapter = int(record["chapter_id"])
        utterance = int(parts[-1])
        speaker_records[speaker].append((book, chapter, utterance, record))

    # Sort context audio candidates per speaker by (book, chapter, utterance)
    for speaker in speaker_records:
        speaker_records[speaker].sort(key=itemgetter(0, 1, 2))
        # replace tuples with just the record
        speaker_records[speaker] = [item[-1] for item in speaker_records[speaker]]

    # choose limited amount of candidates. Generator-based.
    def generate_data():
        for _record in records:
            _speaker = os.path.basename(_record["audio_filepath"]).split("_", 1)[0]

            _candidates = list()
            for _item in speaker_records[_speaker]:
                if _item != _record and _item["duration"] < args.context_min_duration:
                    continue
                _candidates.append(_item)

            if len(_candidates) == 0:
                raise ValueError(
                    "This should not happen because there should be at least 1 item in candidates, which is the record itself."
                )

            # skip the record that does not have any candidates.
            if len(_candidates) == 1 and _record == _candidates[0]:
                continue

            _record_pos = _candidates.index(_record)
            _start = max(0, _record_pos - args.n_candidates_per_record // 2)
            _end = _start + args.n_candidates_per_record
            # remove record itself from the candidates.
            _final_candidates = _candidates[_start:_record_pos] + _candidates[_record_pos + 1 : _end]
            yield _record, _final_candidates

    out_manifest_path = args.manifest.replace(".json", f"_withContextAudioMinDur{int(args.context_min_duration)}.json")
    with open(out_manifest_path, "w", encoding="utf-8") as fout:
        # find high similar context audio using multiple cpu cores.
        with multiprocessing.Pool(
            processes=args.cpu_cores, initializer=init_worker, initargs=(args.embeddings_save_dir,)
        ) as pool:
            chunk_generator = chunked_data_generator(generate_data(), chunk_size=1_000)
            processed_count = 0
            for chunk in tqdm.tqdm(chunk_generator, desc="Processing chunks"):
                results = pool.imap(find_and_add_best_candidate, chunk)
                for res in results:
                    fout.write(f"{json.dumps(res)}\n")
                    processed_count += 1

                # clean memory after processing each chunk
                fout.flush()
                os.fsync(fout.fileno())
                gc.collect()

    # Save results: this script only filter records in min dur of context audio. We should apply SSIM filter separately if needed.
    print(f"Processed {processed_count}/{len(records)} records.")
    print(f"Output manifest: {out_manifest_path}")
