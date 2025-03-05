import argparse
import gc
import json
import multiprocessing
import os
from itertools import islice
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
    candidates = list()
    for record in records:
        if record["duration"] < args.context_min_duration:
            continue
        candidates.append(record)
    if len(candidates) == 0:
        raise ValueError(f"All records are less than {args.context_min_duration} seconds.")

    def generate_data():
        for _record in records:
            parent_dir = os.path.dirname(_record["audio_filepath"])
            _candidates_with_same_parent_dir = [
                x for x in candidates if os.path.dirname(x["audio_filepath"]) == parent_dir
            ]
            if len(_candidates_with_same_parent_dir) == 0:
                print(f"Skip record that has no context audio: {_record}")
                continue
            if _record not in _candidates_with_same_parent_dir:
                _final_candidates = _candidates_with_same_parent_dir
            else:
                _record_pos = _candidates_with_same_parent_dir.index(_record)
                # remove record itself from the candidates.
                _final_candidates = (
                    _candidates_with_same_parent_dir[:_record_pos]
                    + _candidates_with_same_parent_dir[_record_pos + 1 :]
                )
            yield _record, _final_candidates

    out_manifest_path = args.manifest.replace(".json", f"_withContextAudioMinDur{int(args.context_min_duration)}.json")
    with open(out_manifest_path, "w", encoding="utf-8") as fout:
        # find high similar context audio using multiple cpu cores.
        with multiprocessing.Pool(
            processes=args.cpu_cores, initializer=init_worker, initargs=(args.embeddings_save_dir,)
        ) as pool:
            chunk_generator = chunked_data_generator(generate_data(), chunk_size=20)
            processed_count = 0
            for chunk in tqdm.tqdm(chunk_generator, desc="Processing chunks"):
                results = pool.imap(find_and_add_best_candidate, chunk)
                for res in results:
                    fout.write(f"{json.dumps(res, ensure_ascii=False)}\n")
                    processed_count += 1

                # clean memory after processing each chunk
                fout.flush()
                os.fsync(fout.fileno())
                gc.collect()

    # Save results: this script only filter records in min dur of context audio. We should apply SSIM filter separately if needed.
    print(f"Processed {processed_count}/{len(records)} records.")
    print(f"Output manifest: {out_manifest_path}")
