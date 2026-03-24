#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Transcribe audio from a NeMo-style JSONL manifest using Qwen3-ASR.

Supports rank-based sharding for multi-GPU / multi-node Slurm jobs.
Each rank processes its shard independently, writes a partial output,
and optionally rank 0 merges partials into a final manifest.

Dependencies (no NeMo required):
    pip install qwen-asr soundfile numpy

For vLLM backend (recommended for throughput):
    pip install "qwen-asr[vllm]"

For FlashAttention2 (reduces VRAM, speeds up long inputs):
    pip install flash-attn --no-build-isolation

Input JSONL (one JSON object per line):
    {"audio_filepath": "path.wav", "text": "ref", "duration": 5.0, "offset": 0.0, ...}

Output JSONL adds four fields per record:
    "lang"                          — detected language tag (e.g. "Chinese", "English")
    "lang_confidence_qwen3asr"      — placeholder (-1.0; Qwen3-ASR does not expose LID logprobs)
    "hyp_qwen3asr"                  — transcribed text
    "hyp_confidence_qwen3asr"       — placeholder (-1.0; Qwen3-ASR does not expose ASR logprobs)

Usage — single GPU:
    python scripts/magpietts/transcribe_manifest_qwen3asr.py \
        --manifest /data/manifest.json \
        --output-dir /data/output \
        --backend vllm \
        --batch-size 128

Usage — multi-GPU via Slurm (one task per GPU):
    srun --ntasks=8 --gpus-per-task=1 \
        python scripts/magpietts/transcribe_manifest_qwen3asr.py \
            --manifest /data/manifest.json \
            --output-dir /data/output \
            --backend vllm \
            --batch-size 128

    # Merge partial outputs (run once after srun completes):
    python scripts/magpietts/transcribe_manifest_qwen3asr.py \
        --manifest /data/manifest.json \
        --output-dir /data/output \
        --merge-only --world-size 8

Usage — multi-GPU via torchrun on a single node:
    torchrun --nproc_per_node=4 \
        scripts/magpietts/transcribe_manifest_qwen3asr.py \
            --manifest /data/manifest.json \
            --output-dir /data/output \
            --backend transformers \
            --batch-size 32
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

MODEL_TAG = "qwen3asr"
FIELD_LANG = "lang"
FIELD_LANG_CONF = f"lang_confidence_{MODEL_TAG}"
FIELD_HYP = f"hyp_{MODEL_TAG}"
FIELD_HYP_CONF = f"hyp_confidence_{MODEL_TAG}"

QWEN3_ASR_MAX_DURATION_SEC = 1200


def _detect_rank_and_world_size(args):
    """Resolve rank / world_size from explicit args, then SLURM, then torchrun env."""
    rank = args.rank
    world_size = args.world_size

    if rank is None:
        for env_var in ("SLURM_PROCID", "RANK", "LOCAL_RANK"):
            if env_var in os.environ:
                rank = int(os.environ[env_var])
                break
        else:
            rank = 0

    if world_size is None:
        for env_var in ("SLURM_NTASKS", "WORLD_SIZE"):
            if env_var in os.environ:
                world_size = int(os.environ[env_var])
                break
        else:
            world_size = 1

    return rank, world_size


def load_manifest(manifest_path: str) -> list:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {line_num}")
    return records


def load_audio_segment(filepath, offset=0.0, duration=None):
    """Load an audio segment as float32 numpy array.

    Returns (samples, sample_rate).  Multi-channel files are down-mixed to mono.
    """
    info = sf.info(filepath)
    sr = info.samplerate

    start_frame = int(offset * sr)
    stop_frame = start_frame + int(duration * sr) if duration else None

    audio, file_sr = sf.read(filepath, start=start_frame, stop=stop_frame, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, file_sr


def _build_model(args, rank):
    """Initialise the Qwen3-ASR model with the chosen backend."""
    import torch
    from qwen_asr import Qwen3ASRModel

    t0 = time.time()

    if args.backend == "vllm":
        model = Qwen3ASRModel.LLM(
            model=args.model_name_or_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        device = "cpu"
        if torch.cuda.is_available():
            local_id = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", rank)))
            device = f"cuda:{local_id % torch.cuda.device_count()}"

        model = Qwen3ASRModel.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    logger.info(f"Model loaded in {time.time() - t0:.1f}s  [backend={args.backend}]")
    return model


def _error_record(rec):
    out = rec.copy()
    out.update({FIELD_LANG: "ERROR", FIELD_LANG_CONF: -1.0, FIELD_HYP: "", FIELD_HYP_CONF: -1.0})
    return out


def _transcribe_shard(args, rank, world_size):
    """Core processing loop: load shard, transcribe, write partial output."""
    all_records = load_manifest(args.manifest)
    logger.info(f"Manifest has {len(all_records)} records total")

    my_records = all_records[rank::world_size]
    logger.info(f"This rank processes {len(my_records)} records  (interleaved shard {rank}/{world_size})")

    if not my_records:
        logger.warning("No records for this rank — nothing to do.")
        return

    model = _build_model(args, rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_stem = Path(args.manifest).stem
    partial_path = output_dir / f"{manifest_stem}_rank{rank}.json"

    total_ok = 0
    total_err = 0
    total_audio_sec = 0.0
    t_start = time.time()

    num_batches = math.ceil(len(my_records) / args.batch_size)

    with open(partial_path, "w", encoding="utf-8") as out_f:
        for batch_idx in range(num_batches):
            b_start = batch_idx * args.batch_size
            b_end = min(b_start + args.batch_size, len(my_records))
            batch_records = my_records[b_start:b_end]

            audio_inputs = []
            valid_indices = []

            for i, rec in enumerate(batch_records):
                filepath = rec["audio_filepath"]
                if args.audio_base_dir and not os.path.isabs(filepath):
                    filepath = os.path.join(args.audio_base_dir, filepath)

                offset = rec.get("offset", 0.0) or 0.0
                duration = rec.get("duration")

                if duration and duration > QWEN3_ASR_MAX_DURATION_SEC:
                    logger.warning(
                        f"Segment exceeds Qwen3-ASR limit ({duration:.1f}s > {QWEN3_ASR_MAX_DURATION_SEC}s): "
                        f"{rec['audio_filepath']} — truncating to {QWEN3_ASR_MAX_DURATION_SEC}s"
                    )
                    duration = QWEN3_ASR_MAX_DURATION_SEC

                try:
                    audio, sr = load_audio_segment(filepath, offset=offset, duration=duration)
                    audio_inputs.append((audio, sr))
                    valid_indices.append(i)
                    total_audio_sec += len(audio) / sr
                except Exception as e:
                    logger.warning(f"Audio load failed for {rec['audio_filepath']}: {e}")
                    out_f.write(json.dumps(_error_record(rec), ensure_ascii=False) + "\n")
                    total_err += 1

            if not audio_inputs:
                continue

            try:
                results = model.transcribe(audio=audio_inputs, language=None)
            except Exception as e:
                logger.error(f"Batch {batch_idx}/{num_batches} transcription failed: {e}")
                for idx in valid_indices:
                    out_f.write(json.dumps(_error_record(batch_records[idx]), ensure_ascii=False) + "\n")
                    total_err += 1
                continue

            for result, orig_idx in zip(results, valid_indices):
                rec = batch_records[orig_idx].copy()
                rec[FIELD_LANG] = result.language or "unknown"
                rec[FIELD_LANG_CONF] = -1.0
                rec[FIELD_HYP] = result.text or ""
                rec[FIELD_HYP_CONF] = -1.0
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_ok += 1

            # Progress every ~5 % of batches, and always for the last batch
            if (batch_idx + 1) % max(1, num_batches // 20) == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - t_start
                items_done = b_end
                rate = items_done / elapsed if elapsed > 0 else 0
                rtf = elapsed / total_audio_sec if total_audio_sec > 0 else float("inf")
                logger.info(
                    f"[{items_done}/{len(my_records)}] "
                    f"{100 * items_done / len(my_records):.1f}%  "
                    f"{rate:.1f} items/s  RTF={rtf:.4f}  errors={total_err}"
                )

    elapsed_total = time.time() - t_start
    overall_rtf = elapsed_total / total_audio_sec if total_audio_sec > 0 else float("inf")
    logger.info(
        f"Rank {rank} finished: {total_ok} ok, {total_err} errors, "
        f"{elapsed_total:.1f}s wall, {total_audio_sec / 3600:.2f}h audio, RTF={overall_rtf:.4f}"
    )
    logger.info(f"Partial output → {partial_path}")


def _merge_partials(args, world_size):
    """Concatenate per-rank partial manifests into a single output file."""
    output_dir = Path(args.output_dir)
    manifest_stem = Path(args.manifest).stem
    final_path = output_dir / f"{manifest_stem}_hyp_{MODEL_TAG}.json"

    total_lines = 0
    found_ranks = 0

    with open(final_path, "w", encoding="utf-8") as final_f:
        for r in range(world_size):
            partial_path = output_dir / f"{manifest_stem}_rank{r}.json"
            if not partial_path.exists():
                logger.warning(f"Missing partial: {partial_path}")
                continue
            found_ranks += 1
            with open(partial_path, "r", encoding="utf-8") as pf:
                for line in pf:
                    final_f.write(line)
                    total_lines += 1

    logger.info(
        f"Merged {found_ranks}/{world_size} partials → {final_path}  ({total_lines} records)"
    )

    if args.cleanup_partials:
        for r in range(world_size):
            partial_path = output_dir / f"{manifest_stem}_rank{r}.json"
            if partial_path.exists():
                partial_path.unlink()
        logger.info("Cleaned up partial files.")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe NeMo manifests with Qwen3-ASR (rank-independent, Slurm-friendly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("--manifest", type=str, required=True, help="Input JSONL manifest path.")
    io_group.add_argument(
        "--audio-base-dir", type=str, default="",
        help="Prepended to relative audio_filepath entries. Leave empty for absolute paths.",
    )
    io_group.add_argument("--output-dir", type=str, required=True, help="Directory for output manifests.")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-name-or-path", type=str, default="Qwen/Qwen3-ASR-1.7B",
        help="HuggingFace model ID or local path.",
    )
    model_group.add_argument(
        "--backend", type=str, default="vllm", choices=["vllm", "transformers"],
        help="Inference backend. vLLM gives highest throughput.",
    )
    model_group.add_argument("--batch-size", type=int, default=64, help="Max inference batch size.")
    model_group.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per utterance.")
    model_group.add_argument(
        "--gpu-memory-utilization", type=float, default=0.8,
        help="Fraction of GPU memory for vLLM KV cache (vLLM backend only).",
    )

    dist_group = parser.add_argument_group("Distribution / Slurm")
    dist_group.add_argument(
        "--rank", type=int, default=None,
        help="Process rank.  Auto-detected from SLURM_PROCID / RANK / LOCAL_RANK.",
    )
    dist_group.add_argument(
        "--world-size", type=int, default=None,
        help="Total processes.  Auto-detected from SLURM_NTASKS / WORLD_SIZE.",
    )

    merge_group = parser.add_argument_group("Merge")
    merge_group.add_argument(
        "--merge-only", action="store_true",
        help="Skip transcription; only merge existing per-rank partials.",
    )
    merge_group.add_argument(
        "--cleanup-partials", action="store_true",
        help="Delete per-rank partial files after a successful merge.",
    )

    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    rank, world_size = _detect_rank_and_world_size(args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s [RANK {rank}/{world_size}] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.merge_only:
        logger.info("Running in --merge-only mode.")
        _merge_partials(args, world_size)
        return

    logger.info(
        f"Transcription config: model={args.model_name_or_path}, "
        f"backend={args.backend}, batch_size={args.batch_size}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    _transcribe_shard(args, rank, world_size)

    if rank == 0 and world_size == 1:
        _merge_partials(args, world_size)


if __name__ == "__main__":
    main()
