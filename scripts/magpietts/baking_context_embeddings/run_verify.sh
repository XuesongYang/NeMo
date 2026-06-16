#!/usr/bin/env bash
# Ear-check a baked Magpie-TTS checkpoint: synthesize per-speaker, per-language samples.
# Runs scripts/magpietts/baking_context_embeddings/verify_baked_embeddings.py against the baked checkpoint
# produced by run_baking.sh -- no re-baking. Same env setup (nemo-main + fork on PYTHONPATH).
#
# Usage:
#   bash run_verify.sh                          # all supported languages, 3 samples each
#   bash run_verify.sh --languages en,es        # restrict languages
#   bash run_verify.sh --samples-per-language 2 --out-dir /tmp/samples
#   CUDA_VISIBLE_DEVICES=1 bash run_verify.sh
# Any extra args are forwarded to verify_baked_embeddings.py.
set -euo pipefail

# --- environment (override by exporting these before calling) ---
REPO_ROOT="${REPO_ROOT:-/mnt/sdb/sda_bk/xueyang/workspace/gitlab/NeMo}"
PYBIN="${PYBIN:-$HOME/miniconda3/envs/nemo-main/bin/python}"

# --- inputs (override via env vars if needed) ---
EXP_DIR="${EXP_DIR:-/mnt/sdb/sda_bk/xueyang/workspace/experiments/magpietts_v2605_bakingCE}"
CKPT_DIR="${CKPT_DIR:-$EXP_DIR/Magpie_TTS_May_2026/FS_zeroshot_inference_fix_checkpoint}"
CODEC="${CODEC:-$CKPT_DIR/21fps_causal_codecmodel.nemo}"
# The baked .nemo produced by run_baking.sh (its OUTPUT_NEMO); its speaker_map is embedded, and the
# sidecar is also written next to it. (hparams not needed -- the .nemo is self-contained.)
BAKED_NEMO="${BAKED_NEMO:-$EXP_DIR/magpie_tts_multilingual_357m.nemo}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the pickled OmegaConf objects
# inside a Lightning .ckpt. These are trusted local checkpoints, so force the pre-2.6 behavior.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# The codec's snake() activation is @torch.jit.script; torch 2.9+cu130's JIT fuser compiles it via
# nvrtc, which fails to open libnvrtc-builtins.so.13.0 in this env. Disable the JIT so snake runs
# eager (identical math, no nvrtc) -- affects only speed, not correctness.
export PYTORCH_JIT=0

# Make the fork's nemo win over the installed nemo_toolkit; run from repo root for relative paths.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

# --- sanity checks ---
[[ -x "$PYBIN" ]] || { echo "ERROR: python not found/executable: $PYBIN" >&2; exit 1; }
[[ -f "scripts/magpietts/baking_context_embeddings/verify_baked_embeddings.py" ]] || {
  echo "ERROR: verify script not found under REPO_ROOT=$REPO_ROOT" >&2
  exit 1
}

echo "REPO_ROOT            = $REPO_ROOT"
echo "PYBIN                = $PYBIN"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "BAKED_NEMO           = $BAKED_NEMO"
echo "nemo resolves to     : $("$PYBIN" -c 'import nemo; print(nemo.__file__)')"
echo

exec "$PYBIN" scripts/magpietts/baking_context_embeddings/verify_baked_embeddings.py \
  --baked-ckpt      "$BAKED_NEMO" \
  --codecmodel-path "$CODEC" \
  --languages "en" \
  "$@"
