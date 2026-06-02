# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Bake context-encoder outputs into per-speaker embeddings for a zero-shot Magpie-TTS
checkpoint (``model_type: decoder_ce``).

This converts a zero-shot checkpoint into a multi-speaker checkpoint with a fixed roster of
speakers selectable by integer index. For each speaker we run their reference audio through the
model's OWN runtime context path -- the same ``MagpieTTSDataset`` + ``prepare_context_tensors``
pipeline used at inference (codec -> context BOS/EOS -> audio embedding -> context encoder, with
the dataset's crop/tile to the context window and the text-conditioning padding/masking) -- and
take ``additional_decoder_input``, the context-encoder output the decoder cross-attends to. Going
through the model's real path (rather than re-implementing it) guarantees the baked tensor matches
what an un-baked zero-shot run feeds the decoder. The per-speaker ``(T_s, D)`` tensors are packed
into a single ``nn.Embedding`` lookup table indexed by speaker id, and the model is exported with
``save_to`` as a deployable ``.nemo`` -- ``state_dict()`` drops the now-unused ``context_encoder``
(and the codec / SV models) while keeping the four baked tensors.

The exported ``.nemo`` is deployment-ready and self-contained: it carries the tokenizer artifacts,
embeds an ``inference_parameters`` block (so ``do_tts`` works without CLI overrides, like the
published 357m ``.nemo``), references the public codec by name (``--nemo-codecmodel-path``), and
bundles the ``{speaker_name: index}`` map as a ``speaker_map`` artifact. The same map is also
written as a standalone sidecar JSON (browsable, and usable as a dataset ``speaker_path``).

To ear-check the baked voices afterward (without re-baking), use the companion script
``scripts/magpietts/verify_baked_embeddings.py``.

The baked contract (consumed by ``MagpieTTSModel.load_state_dict``) requires these four tensors in
the model weights, with all ``context_encoder.*`` keys absent:
    baked_context_embedding.weight   (N, T_max * D) float   one flattened row per speaker
    _baked_embedding_T               scalar int             T_max (longest speaker length)
    _baked_embedding_D               scalar int             D (context-encoder hidden size)
    baked_context_embedding_len      (N,) int               true unpadded T_s per speaker

Example usage (un-baked .ckpt + hparams + codec .nemo in, deployable baked .nemo out):
    python scripts/magpietts/bake_context_embeddings.py \
        --hparams-file    /path/config_2605.yaml \
        --checkpoint-file /path/Magpie-TTS--val_loss_9.9725-step_497031.ckpt \
        --codecmodel-path /path/21fps_causal_codecmodel.nemo \
        --manifest        /path/speakers.yaml \
        --output-nemo     /path/magpie_tts_multilingual_357m.nemo

Reference manifest (YAML or JSON, required). The order of speakers IS the integer index
contract:
    speakers:
      - name: Sofia
        clips: [/data/sofia_1.wav, /data/sofia_2.wav]
      - name: Aria
        clips: [/data/aria_1.wav]
      ...

A speaker entry's ``name`` may be omitted, in which case it defaults to the built-in roster
name for that position. The default roster is alphabetical (Aria, Jason, John Van Stan, Leo,
Sofia); ``clips`` are always required.

Each speaker's clips are concatenated into a single reference, which the model's context pipeline
then crops or tiles to a fixed context window (``--max-duration``, default the model's
``context_duration_max``) -- so every baked speaker gets the same context length, matching what
the model saw in training. By default the available reference must be at least ``--min-duration``
(default the model's ``context_duration_min``); a shorter one is refused (it would bake a weak
embedding) unless ``--allow-short-references`` is given, in which case the pipeline tiles it
(repeat-and-concatenate) up to the context window, exactly as MagpieTTSDataset does at
train/inference time.

Picking reference clips: the longest clips in a directory are often the best candidates. To
list the top-10 longest (longest first) with SoX (requires ``soxi`` on PATH):

    find /path/to/refs -type f -iname '*.wav' -print0 \
    | xargs -0 -P"$(nproc)" -I{} bash -c 'printf "%s\t%s\n" "$(soxi -D "$1")" "$1"' _ {} \
    | sort -rn | head -10

Deploying / using the baked .nemo
---------------------------------
The .nemo is self-contained: ``restore_from`` (or ``do_tts``) loads it with the embedded
inference_parameters and pulls the codec from the embedded ``codecmodel_path`` name. The
``{speaker_name: index}`` map is bundled as a ``speaker_map`` artifact (read back via
``model.cfg.speaker_map``) and also written alongside as a standalone sidecar JSON.

For batched eval via ``examples/tts/magpietts_inference.py``: the eval set carries NO context
audio; each manifest record has a ``speaker`` field, and the dataset's ``speaker_path`` points
at the sidecar (a flat ``{speaker_name: index}`` map) so ``speaker`` resolves to the baked
integer index. Example ``--datasets_json_path`` entry:

    {
      "my_evalset": {
        "manifest_path": "/path/eval_manifest.json",   # records have "text" + "speaker"
        "audio_dir": "/path/audio",
        "speaker_path": "/path/magpie_tts_multilingual_357m.nemo.speakers.json",
        "tokenizer_names": ["english_phoneme"]
      }
    }
"""

import argparse
import json
import logging
import os
import tempfile
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import open_dict

from nemo.collections.tts.data.text_to_speech_dataset import DatasetSample, MagpieTTSDataset
from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model

# Default speaker roster (used only to fix names/order when not given in the manifest).
# Names are kept in alphabetical order, which defines the default integer-index assignment.
DEFAULT_SPEAKER_ROSTER = ["Aria", "Jason", "John Van Stan", "Leo", "Sofia"]

# Public codec reference written into the deployed .nemo config (restore_from loads the codec from
# it). Confirmed identical to the local 21fps_causal_codecmodel.nemo used for baking.
DEFAULT_NEMO_CODEC = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"

# Slack on the minimum-duration check to absorb sample-rounding (a "10s" clip may be a few
# samples under 10.0s); references this close to the floor still carry enough material.
DURATION_TOLERANCE_SEC = 0.05

# The four tensors the loader expects in a baked checkpoint's inner state dict.
BAKED_TENSOR_KEYS = (
    "baked_context_embedding.weight",
    "_baked_embedding_T",
    "_baked_embedding_D",
    "baked_context_embedding_len",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake context-encoder outputs into per-speaker embeddings for a decoder_ce Magpie-TTS checkpoint."
    )

    # --- Source model (checkpoint mode: hparams + un-baked .ckpt + codec .nemo) ---
    parser.add_argument("--hparams-file", type=str, required=True, help="Path to the model hparams/config YAML.")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Path to the un-baked source .ckpt (must still contain context_encoder weights).",
    )
    parser.add_argument("--codecmodel-path", type=str, required=True, help="Path to the audio codec model (.nemo).")

    # --- Speaker reference audio ---
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="YAML/JSON manifest mapping speaker name -> list of clip paths. Order defines the integer index.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Trim each speaker's concatenated reference to this many seconds. "
        "Default: the model's context_duration_max (the duration it was trained with).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Require at least this many seconds of reference audio per speaker; baking a "
        "shorter reference yields a weak, out-of-distribution speaker embedding. "
        "Default: the model's context_duration_min.",
    )
    parser.add_argument(
        "--allow-short-references",
        action="store_true",
        help="Instead of erroring on a reference below --min-duration, bake it anyway -- the context "
        "pipeline tiles it (repeat-and-concatenate) up to the context window, as MagpieTTSDataset does.",
    )

    # --- Outputs ---
    parser.add_argument(
        "--output-nemo",
        type=str,
        default=None,
        help="Output baked .nemo path (deployable archive). Default: <source>_baked.nemo next to the source.",
    )
    parser.add_argument(
        "--output-sidecar",
        type=str,
        default=None,
        help="Output {speaker_name: index} sidecar JSON (browsable copy + eval `speaker_path`); it is also "
        "embedded inside the .nemo as a `speaker_map` artifact. Default: <output-nemo>.speakers.json.",
    )
    parser.add_argument(
        "--nemo-codecmodel-path",
        type=str,
        default=DEFAULT_NEMO_CODEC,
        help="codecmodel_path written into the deployed .nemo config (what restore_from will load). "
        f"Default: {DEFAULT_NEMO_CODEC}. Set to a local .nemo only for offline use.",
    )

    # --- Inference parameters embedded into the .nemo config (so do_tts works without overrides,
    #     like the published 357m .nemo). Defaults match examples/tts/magpietts_inference.py. ---
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--topk", type=int, default=80, help="Top-k sampling.")
    parser.add_argument("--cfg-scale", type=float, default=2.5, help="Classifier-free guidance scale.")
    parser.add_argument("--max-decoder-steps", type=int, default=500, help="Max autoregressive decoder steps.")
    parser.add_argument(
        "--apply-prior-to-layers",
        type=str,
        default="2,3,4,5,6,7,8,9,10",
        help="Comma-separated decoder layers to apply the attention prior to (empty string to disable).",
    )
    parser.add_argument(
        "--estimate-alignment-from-layers",
        type=str,
        default="4,5,8,9",
        help="Comma-separated decoder layers used to estimate alignment for the attention prior.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device to run baking on (cuda or cpu).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def load_manifest(manifest_path: str) -> List[Tuple[str, List[str]]]:
    """Load the speaker manifest, returning an ordered list of (name, [clip_paths]).

    Accepts either YAML or JSON with the structure::

        speakers:
          - name: Sofia
            clips: [a.wav, b.wav]
          - ...
    """
    with open(manifest_path, "r") as f:
        if manifest_path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    if "speakers" not in data:
        raise ValueError(f"Manifest {manifest_path} must have a top-level 'speakers' list.")

    roster = []
    for i, entry in enumerate(data["speakers"]):
        # Fall back to the default roster name (by position) when the entry omits one.
        name = entry.get("name") or (DEFAULT_SPEAKER_ROSTER[i] if i < len(DEFAULT_SPEAKER_ROSTER) else None)
        clips = entry.get("clips")
        if not name:
            raise ValueError(f"Speaker entry {i} in {manifest_path} is missing a 'name'.")
        if not clips:
            raise ValueError(f"Speaker '{name}' in {manifest_path} has no 'clips'.")
        roster.append((name, list(clips)))
    return roster


def load_and_concat_references(
    clip_paths: List[str], target_sr: int, max_duration: float
) -> Tuple[torch.Tensor, float]:
    """Load reference clips, resample to target_sr, concat into one mono waveform, trim to max_duration.

    Returns ``(waveform, available_duration_sec)`` where ``waveform`` is a 1-D float32 tensor of
    audio samples (already trimmed to max_duration) and ``available_duration_sec`` is the total
    concatenated reference duration *before* trimming -- used to gate against too-short references.
    """
    waveforms = []
    for clip_path in clip_paths:
        audio, sr = sf.read(clip_path, dtype="float32")
        if audio.ndim > 1:  # stereo -> mono
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        waveforms.append(torch.from_numpy(audio).float())

    waveform = torch.cat(waveforms, dim=0)
    available_duration_sec = waveform.numel() / target_sr
    max_samples = int(max_duration * target_sr)
    if waveform.numel() > max_samples:
        waveform = waveform[:max_samples]
    return waveform, available_duration_sec


def build_inference_dataset(model, context_window: float) -> MagpieTTSDataset:
    """Build a MagpieTTSDataset wired like the inference notebook.

    Running each speaker's reference through this dataset + ``model.prepare_context_tensors`` drives
    the model's REAL runtime context path (codec -> context BOS/EOS -> embed -> context encoder,
    including the text-conditioning padding/masking and the dataset's own crop/tile to the context
    window). That is the exact path that produced good audio in the un-baked zero-shot check, so the
    tensor we bake matches what the decoder consumes at inference.

    The context window is pinned to ``context_window`` seconds (min == max), so every speaker is
    deterministically cropped/tiled to the same length -- no random duration sampling, and it honors
    ``--max-duration`` rather than silently reverting to the model config's value.
    """
    ds = MagpieTTSDataset(
        dataset_meta={},
        sample_rate=model.sample_rate,
        min_duration=0.5,
        max_duration=20,
        codec_model_samples_per_frame=model.codec_model_samples_per_frame,
        bos_id=model.bos_id,
        eos_id=model.eos_id,
        num_audio_codebooks=model.num_audio_codebooks,
        prior_scaling_factor=None,
        load_cached_codes_if_available=False,
        dataset_type="test",
        tokenizer_config=None,
        load_16khz_audio=model.model_type == "decoder_ce",
        use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
        text_conditioning_tokenizer_name=model.text_conditioning_tokenizer_name,
        pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
        context_duration_min=context_window,
        context_duration_max=context_window,
    )
    ds.text_tokenizer = model.tokenizer
    return ds


@torch.no_grad()
def embed_speaker_context(model, dataset, ref_wav_path, dummy_target_wav, tokenizer_name, device):
    """Run the model's runtime context path on a reference wav; return (context_emb (T_s, D), T_s).

    Builds a one-record batch via the dataset collate (audio context, no text context), runs
    ``model.prepare_context_tensors``, and returns ``additional_decoder_input`` -- the context-encoder
    output the decoder cross-attends to -- sliced to its valid length. This is exactly what an
    un-baked zero-shot run feeds the decoder, so the baked tensor reproduces inference.
    """
    with sf.SoundFile(ref_wav_path) as f:
        ctx_duration = len(f) / f.samplerate
    entry = {
        "audio_filepath": dummy_target_wav,  # target audio is irrelevant to the context tensors
        "duration": 3.0,
        "text": "hello",
        "speaker": "dummy",
        "context_audio_filepath": ref_wav_path,
        "context_audio_duration": ctx_duration,
    }
    sample = DatasetSample(
        dataset_name="bake",
        manifest_entry=entry,
        audio_dir="/",
        feature_dir="/",
        text="hello",
        speaker=None,
        speaker_index=0,
        tokenizer_names=[tokenizer_name],
    )
    dataset.data_samples = [sample]
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=0)
    batch = next(iter(loader))
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    ctx = model.prepare_context_tensors(batch)
    emb = ctx.additional_decoder_input[0]  # (T_ctx, D) -- context-encoder output
    mask = ctx.additional_decoder_mask[0].bool()  # (T_ctx,)
    t_s = int(mask.sum().item())
    return emb[:t_s].detach().cpu().float(), t_s


def pack_embeddings(embeddings: List[torch.Tensor], lengths: List[int]) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    """Pack per-speaker (T_s, D) tensors into the baked nn.Embedding format.

    Returns (weight (N, T_max*D), T_max, D, lengths (N,)).
    """
    num_speakers = len(embeddings)
    t_max = max(lengths)
    d = embeddings[0].shape[1]

    packed = torch.zeros(num_speakers, t_max, d, dtype=torch.float)
    for i, (emb, t_s) in enumerate(zip(embeddings, lengths)):
        if emb.shape[1] != d:
            raise ValueError(
                f"Speaker {i} has hidden size {emb.shape[1]}, expected {d}. Inconsistent context encoder."
            )
        packed[i, :t_s] = emb

    weight = packed.reshape(num_speakers, t_max * d)  # nn.Embedding requires 2-D weights.
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return weight, t_max, d, lengths_tensor


def parse_int_list(s: str):
    """Parse a comma-separated int list (e.g. '2,3,4') -> [2, 3, 4]; empty/None -> None."""
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def apply_baked_embedding_to_model(model, weight, t_max, d, lengths_tensor):
    """Set the baked tensors on a live (un-baked) model so ``model.save_to`` exports a baked .nemo.

    Mirrors ``MagpieTTSModel.load_state_dict``'s baked branch: creates the ``nn.Embedding`` and fills
    the ``_baked_embedding_T/_D`` / ``baked_context_embedding_len`` buffers. Once these are set,
    ``model.has_baked_context_embedding`` becomes True, which makes ``state_dict()`` drop the
    ``context_encoder`` weights and keep the four baked tensors -- exactly the baked contract.
    """
    model.baked_context_embedding = torch.nn.Embedding(weight.size(0), weight.size(1))
    model.baked_context_embedding.weight.data = weight
    model.baked_context_embedding.weight.requires_grad_(False)
    model._baked_embedding_T = torch.tensor(t_max, dtype=torch.long)
    model._baked_embedding_D = torch.tensor(d, dtype=torch.long)
    model.baked_context_embedding_len = lengths_tensor


def build_inference_parameters_cfg(args) -> dict:
    """Inference parameters to embed in the .nemo config so do_tts works without CLI overrides.

    The published 357m .nemo carries these; our source config does not, so without them the
    attention-prior layer lists stay unset and the decoder terminates almost immediately.
    """
    return {
        "max_decoder_steps": args.max_decoder_steps,
        "temperature": args.temperature,
        "topk": args.topk,
        "cfg_scale": args.cfg_scale,
        "apply_attention_prior": True,
        "attention_prior_epsilon": 0.1,
        "attention_prior_lookahead_window": 5,
        "estimate_alignment_from_layers": parse_int_list(args.estimate_alignment_from_layers),
        "apply_prior_to_layers": parse_int_list(args.apply_prior_to_layers),
        "start_prior_after_n_audio_steps": 0,
        "ignore_finished_sentence_tracking": True,
        "eos_detection_method": "argmax_or_multinomial_any",
    }


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Resolve outputs.
    output_nemo = args.output_nemo
    if output_nemo is None:
        base, _ext = os.path.splitext(args.checkpoint_file)
        output_nemo = f"{base}_baked.nemo"
    output_sidecar = args.output_sidecar or f"{output_nemo}.speakers.json"

    # Load the un-baked model (eval mode, codec loaded).
    load_config = ModelLoadConfig(
        hparams_file=args.hparams_file, checkpoint_file=args.checkpoint_file, codecmodel_path=args.codecmodel_path
    )
    logging.info("Loading un-baked source model ...")
    model, _ = load_magpie_model(load_config, device=args.device)
    model.eval()
    if model.model_type != "decoder_ce":
        raise ValueError(f"Baking only applies to decoder_ce models; got model_type={model.model_type}.")
    if model.has_baked_context_embedding:
        raise ValueError("Source model already has a baked context embedding; nothing to bake.")

    # Resolve the reference-duration window from the model's training context window unless overridden.
    # The model was trained with context in [context_duration_min, context_duration_max]; baking with at
    # least context_duration_min keeps the embedding in-distribution.
    cfg_max = float(model.cfg.get("context_duration_max", 10.0))
    cfg_min = float(model.cfg.get("context_duration_min", cfg_max))
    max_duration = args.max_duration if args.max_duration is not None else cfg_max
    min_duration = args.min_duration if args.min_duration is not None else cfg_min
    if min_duration > max_duration:
        raise ValueError(
            f"--min-duration ({min_duration:.2f}s) exceeds --max-duration ({max_duration:.2f}s); "
            "references trimmed to max could never satisfy min."
        )
    logging.info(f"Reference duration window: require >= {min_duration:.2f}s, trim to <= {max_duration:.2f}s.")

    roster = load_manifest(args.manifest)
    logging.info(f"Baking {len(roster)} speakers: {[name for name, _ in roster]}")

    # Build the runtime dataset once (context window pinned to max_duration so the bake is
    # deterministic and honors --max-duration); pick a tokenizer for the (irrelevant) dummy text.
    dataset = build_inference_dataset(model, max_duration)
    available_tokenizers = list(model.tokenizer.tokenizers.keys())
    tokenizer_name = "english_phoneme" if "english_phoneme" in available_tokenizers else available_tokenizers[0]

    # Phase 1: per-speaker context embedding, computed via the model's real runtime path.
    embeddings, lengths = [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Dummy target waveform: the dataset needs a target audio file, but it does not affect the
        # context tensors we extract.
        dummy_target_wav = os.path.join(tmp_dir, "dummy_target.wav")
        sf.write(dummy_target_wav, np.zeros(int(model.sample_rate * 3), dtype=np.float32), model.sample_rate)

        for i, (name, clips) in enumerate(roster):
            logging.info(f"[{i}] {name}: {len(clips)} clip(s)")
            waveform, ref_duration = load_and_concat_references(clips, model.sample_rate, max_duration)
            # Small tolerance so sample-rounding doesn't reject a reference essentially at the floor.
            if ref_duration < min_duration - DURATION_TOLERANCE_SEC:
                msg = (
                    f"Speaker '{name}' has only {ref_duration:.2f}s of reference audio, below the required "
                    f"minimum of {min_duration:.2f}s (the model's context_duration_min). A reference this "
                    f"short bakes a weak speaker embedding -- provide more or longer clips."
                )
                if args.allow_short_references:
                    # The dataset tiles a short clip up to the context window (same as runtime), so just warn.
                    logging.warning(msg + " Proceeding; the dataset will tile it up to the context window.")
                else:
                    raise ValueError(msg + " Re-run with --allow-short-references to bake anyway.")

            # Write the (resampled, trimmed) reference to a temp wav so the dataset processes it
            # exactly as it would a real context clip, then take the context-encoder output.
            ref_wav = os.path.join(tmp_dir, f"ref_{i}.wav")
            sf.write(ref_wav, waveform.numpy(), model.sample_rate)
            emb, t_s = embed_speaker_context(model, dataset, ref_wav, dummy_target_wav, tokenizer_name, args.device)
            logging.info(f"     -> {ref_duration:.2f}s reference, context embedding ({t_s}, {emb.shape[1]})")
            embeddings.append(emb)
            lengths.append(t_s)

    # Phase 2: pack.
    weight, t_max, d, lengths_tensor = pack_embeddings(embeddings, lengths)
    logging.info(f"Packed baked embedding: N={len(roster)}, T_max={t_max}, D={d}, lengths={lengths}")

    # Phase 3: bake the tensors into the live model and export a deployable .nemo via save_to().
    apply_baked_embedding_to_model(model, weight, t_max, d, lengths_tensor)
    assert model.has_baked_context_embedding, "Model did not register the baked context embedding."

    # Contract check on the export state dict: four baked tensors present, no context_encoder.* left.
    export_sd = model.state_dict()
    leftover = [k for k in export_sd if "context_encoder" in k]
    assert not leftover, f"context_encoder.* unexpectedly present in the export state dict: {leftover[:3]}..."
    for required in BAKED_TENSOR_KEYS:
        assert required in export_sd, f"export state dict is missing required baked tensor '{required}'."

    # Bake the deployment config into the .nemo: the public codec reference (what restore_from loads)
    # and the inference parameters so do_tts works without overrides (like the published 357m .nemo).
    with open_dict(model.cfg):
        model.cfg.codecmodel_path = args.nemo_codecmodel_path
        model.cfg.inference_parameters = build_inference_parameters_cfg(args)

    # Write the standalone {speaker_name: index} sidecar (browsable / eval `speaker_path`) and embed
    # the same file inside the .nemo as a `speaker_map` artifact so the model is self-describing.
    speaker_index_map = {name: i for i, (name, _clips) in enumerate(roster)}
    with open(output_sidecar, "w") as f:
        json.dump(speaker_index_map, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved speaker_index_map sidecar -> {output_sidecar}")
    spk_artifact = model.register_artifact("speaker_map", output_sidecar, verify_src_exists=True)
    with open_dict(model.cfg):
        model.cfg.speaker_map = spk_artifact

    os.makedirs(os.path.dirname(os.path.abspath(output_nemo)), exist_ok=True)
    model.save_to(output_nemo)
    logging.info(
        f"Saved baked model -> {output_nemo}  (codecmodel_path={args.nemo_codecmodel_path}, "
        f"speaker_map embedded, inference_parameters embedded)"
    )

    logging.info(
        "Done baking. Ear-check the voices with:\n"
        f'  python scripts/magpietts/verify_baked_embeddings.py --baked-ckpt "{output_nemo}" '
        f'--codecmodel-path "{args.nemo_codecmodel_path}"'
    )


if __name__ == "__main__":
    main()
