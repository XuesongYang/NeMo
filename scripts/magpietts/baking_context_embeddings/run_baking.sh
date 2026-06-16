#!/usr/bin/env bash
# Bake per-speaker context embeddings into the Magpie-TTS v2605 decoder_ce checkpoint.
#
# Runs the in-repo (fork) bake script with the `nemo-main` conda env, exporting PYTHONPATH so
# the fork's `nemo/` shadows the conda-installed nemo_toolkit (which is a different build).
#
# Usage:
#   bash run_baking.sh                          # bake the roster into OUTPUT_NEMO (+ sidecar)
#   CUDA_VISIBLE_DEVICES=1 bash run_baking.sh    # pick a GPU
# Any extra args are forwarded to bake_context_embeddings.py (e.g. --allow-short-references).
# Produces a deployable .nemo (codec ref + inference_parameters + speaker_map embedded).
# To ear-check the result afterward (no re-baking), use run_verify.sh.
set -euo pipefail

# --- environment (override by exporting these before calling) ---
REPO_ROOT="${REPO_ROOT:-/mnt/sdb/sda_bk/xueyang/workspace/gitlab/NeMo}"
PYBIN="${PYBIN:-$HOME/miniconda3/envs/nemo-main/bin/python}"

# --- inputs / outputs (override via env vars if needed) ---
EXP_DIR="${EXP_DIR:-/mnt/sdb/sda_bk/xueyang/workspace/experiments/magpietts_v2605_bakingCE}"
CKPT_DIR="${CKPT_DIR:-$EXP_DIR/Magpie_TTS_May_2026/FS_zeroshot_inference_fix_checkpoint}"
HPARAMS="${HPARAMS:-$CKPT_DIR/config_2605.yaml}"
CHECKPOINT="${CHECKPOINT:-$CKPT_DIR/Magpie-TTS--val_cer_gt_0.0488-step_960.ckpt}"
CODEC="${CODEC:-$CKPT_DIR/21fps_causal_codecmodel.nemo}"
MANIFEST="${MANIFEST:-$EXP_DIR/reference_manifest.yaml}"
OUTPUT_NEMO="${OUTPUT_NEMO:-$EXP_DIR/magpie_tts_multilingual_357m.nemo}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the pickled OmegaConf objects
# inside a Lightning .ckpt. These are trusted local checkpoints, so force the pre-2.6 behavior via
# PyTorch's official escape hatch (avoids patching the NeMo loader).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# The codec's snake() activation is @torch.jit.script; torch 2.9+cu130's JIT fuser compiles it via
# nvrtc, which fails to open libnvrtc-builtins.so.13.0 in this env. Disable the JIT so it runs eager
# (identical math, no nvrtc) -- affects only speed, not correctness.
export PYTORCH_JIT=0

# Make the fork's nemo win over the installed nemo_toolkit; run from repo root so the model
# loader's relative paths (e.g. scripts/tts_dataset_files/...) resolve.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

# --- sanity checks ---
[[ -x "$PYBIN" ]] || { echo "ERROR: python not found/executable: $PYBIN" >&2; exit 1; }
[[ -f "scripts/magpietts/bake_context_embeddings.py" ]] || {
  echo "ERROR: bake script not found under REPO_ROOT=$REPO_ROOT" >&2
  exit 1
}

echo "REPO_ROOT            = $REPO_ROOT"
echo "PYBIN                = $PYBIN"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "nemo resolves to     : $("$PYBIN" -c 'import nemo; print(nemo.__file__)')"
echo

exec "$PYBIN" scripts/magpietts/bake_context_embeddings.py \
  --hparams-file    "$HPARAMS" \
  --checkpoint-file "$CHECKPOINT" \
  --codecmodel-path "$CODEC" \
  --manifest        "$MANIFEST" \
  --output-nemo     "$OUTPUT_NEMO" \
  "$@"
