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
Verify a baked Magpie-TTS checkpoint by synthesizing per-speaker, per-language samples.

This loads an already-baked ``decoder_ce`` checkpoint (produced by
``scripts/magpietts/bake_context_embeddings.py``) and synthesizes a few utterances for every
baked speaker, so you can ear-check that each voice's timbre matches its reference. It does NOT
bake -- run it as many times as you like without re-baking.

Speaker names/order come from the ``{speaker_name: index}`` map -- the sidecar JSON next to the
model, or the ``speaker_map`` artifact embedded in the .nemo -- since the weights carry no speaker
metadata. The integer index passed to the model is that map's value, matching what was baked in.

Works with a baked ``.nemo`` (self-contained) or a baked ``.ckpt`` (then pass ``--hparams-file``).

Outputs one wav per (language, speaker, sample) under:
    <out-dir>/<language>/<speaker>/sample_<j>.wav

Example usage (.nemo):
    python scripts/magpietts/verify_baked_embeddings.py \
        --baked-ckpt      /path/magpie_tts_multilingual_357m.nemo \
        --codecmodel-path nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps
    # restrict languages and sample count:
    python scripts/magpietts/verify_baked_embeddings.py ... --languages en,es --samples-per-language 2

By default it synthesizes with the standard tuned inference parameters (temperature / top-k / CFG
plus the attention prior) matching examples/tts/magpietts_inference.py. The attention prior is
essential -- without it (the model's config defaults leave its layer lists unset) the decoder does
not align to the text and terminates almost immediately. Override via --temperature / --topk /
--cfg-scale / --apply-prior-to-layers / --estimate-alignment-from-layers.

The synthesis ear-check is the only thing that catches a bad bake (wrong reference clip, wrong
speaker order, or a context-pipeline mismatch) -- the baked tensor shapes look correct regardless.
"""

import argparse
import json
import logging
import os
from typing import List, Tuple

import soundfile as sf

from nemo.collections.tts.models.magpietts import ModelInferenceParameters
from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model
from nemo.collections.tts.parts.utils.tts_dataset_utils import LANGUAGE_TOKENIZER_MAP

# Sample sentences per language. Keys are the language codes of LANGUAGE_TOKENIZER_MAP. Kept
# digit-free so they synthesize cleanly without text normalization. A language the model supports
# but that is missing here falls back to "en".
TEXTS_BY_LANGUAGE = {
    "en": [
        "Hello, this is a baked voice sample.",
        "The quick brown fox jumps over the lazy dog.",
        "Please confirm your appointment for next Tuesday.",
    ],
    "de": [
        "Hallo, dies ist eine Sprachprobe.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Bitte bestätigen Sie Ihren Termin für nächsten Dienstag.",
    ],
    "es": [
        "Hola, esta es una muestra de voz.",
        "El veloz zorro marrón salta sobre el perro perezoso.",
        "Por favor, confirme su cita para el próximo martes.",
    ],
    "fr": [
        "Bonjour, ceci est un échantillon de voix.",
        "Le vif renard brun saute par-dessus le chien paresseux.",
        "Veuillez confirmer votre rendez-vous pour mardi prochain.",
    ],
    "it": [
        "Ciao, questo è un campione vocale.",
        "La rapida volpe marrone salta sopra il cane pigro.",
        "Si prega di confermare il tuo appuntamento per martedì prossimo.",
    ],
    "vi": [
        "Xin chào, đây là một mẫu giọng nói.",
        "Con cáo nâu nhanh nhẹn nhảy qua con chó lười.",
        "Vui lòng xác nhận cuộc hẹn của bạn vào thứ Ba tới.",
    ],
    "zh": [
        "你好，这是一段语音样本。",
        "今天的天气非常好，适合出去散步。",
        "请确认您下周二的预约。",
    ],
    "hi": [
        "नमस्ते, यह एक आवाज़ का नमूना है।",
        "आज मौसम बहुत अच्छा है।",
        "कृपया अगले मंगलवार के लिए अपनी नियुक्ति की पुष्टि करें।",
    ],
    "ja": [
        "こんにちは、これは音声サンプルです。",
        "今日はとても良い天気です。",
        "来週の火曜日のご予約を確認してください。",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize per-speaker, per-language samples from a baked Magpie-TTS checkpoint."
    )
    parser.add_argument(
        "--hparams-file",
        type=str,
        default=None,
        help="Model hparams/config YAML. Required only when --baked-ckpt is a .ckpt (not for a .nemo).",
    )
    parser.add_argument(
        "--baked-ckpt",
        type=str,
        required=True,
        help="Path to the baked model: a .nemo (self-contained) or a .ckpt (needs --hparams-file).",
    )
    parser.add_argument("--codecmodel-path", type=str, required=True, help="Path to the audio codec model (.nemo).")
    parser.add_argument(
        "--sidecar",
        type=str,
        default=None,
        help="{speaker_name: index} JSON written by baking. Default: <baked-ckpt>.speakers.json.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes to synthesize (e.g. 'en,es,ja'). "
        "Default: every language the model supports.",
    )
    parser.add_argument(
        "--samples-per-language", type=int, default=3, help="Number of sample wavs per language per speaker."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Root dir for sample wavs, structured {language}/{speaker}/sample_*.wav. "
        "Default: <baked-ckpt dir>/baked_samples.",
    )
    # --- Inference parameters (defaults match the standard examples/tts/magpietts_inference.py run) ---
    # The attention prior is essential: without estimate_alignment_from_layers / apply_prior_to_layers
    # set, the decoder never aligns to the text and terminates almost immediately (instant EOS).
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--topk", type=int, default=80, help="Top-k sampling.")
    parser.add_argument("--cfg-scale", type=float, default=2.5, help="Classifier-free guidance scale.")
    parser.add_argument("--max-decoder-steps", type=int, default=500, help="Max autoregressive decoder steps.")
    parser.add_argument(
        "--apply-prior-to-layers",
        type=str,
        default="2,3,4,5,6,7,8,9,10",
        help="Comma-separated decoder layers to apply the attention prior to (empty string to disable the prior).",
    )
    parser.add_argument(
        "--estimate-alignment-from-layers",
        type=str,
        default="4,5,8,9",
        help="Comma-separated decoder layers used to estimate alignment for the attention prior.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device to run synthesis on (cuda or cpu).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def parse_int_list(s: str):
    """Parse a comma-separated int list (e.g. '2,3,4') -> [2, 3, 4]; empty/None -> None."""
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def load_roster_from_sidecar(sidecar_path: str) -> List[Tuple[str, int]]:
    """Return [(name, index), ...] ordered by index from a {speaker_name: index} sidecar JSON."""
    with open(sidecar_path, "r") as f:
        name_to_index = json.load(f)
    return sorted(name_to_index.items(), key=lambda kv: kv[1])


def get_supported_languages(model) -> List[str]:
    """Return the language codes the model supports (a candidate tokenizer is loaded for them)."""
    available = set(model.tokenizer.tokenizers.keys())
    return [lang for lang, candidates in LANGUAGE_TOKENIZER_MAP.items() if any(c in available for c in candidates)]


def resolve_languages(model, requested: str) -> List[str]:
    """Resolve the languages to synthesize: the requested subset (validated) or all supported."""
    supported = get_supported_languages(model)
    if not requested:
        return supported
    asked = [lang.strip() for lang in requested.split(",") if lang.strip()]
    languages = [lang for lang in asked if lang in supported]
    unsupported = [lang for lang in asked if lang not in supported]
    if unsupported:
        logging.warning(f"Requested languages not supported by the model, skipping: {unsupported}")
    return languages


def synthesize_samples(
    model, roster: List[Tuple[str, int]], out_dir: str, languages: List[str], samples_per_lang: int
):
    """Synthesize `samples_per_lang` utterances per language per speaker under out_dir/{lang}/{speaker}/."""
    model.eval()
    for language in languages:
        texts = TEXTS_BY_LANGUAGE.get(language)
        if texts is None:
            logging.warning(f"No built-in text for language '{language}'; using English samples.")
            texts = TEXTS_BY_LANGUAGE["en"]
        # Pick the requested number of prompts, cycling through the available ones if needed.
        chosen = [texts[j % len(texts)] for j in range(samples_per_lang)]

        for name, index in roster:
            safe_name = name.replace(" ", "_")
            speaker_dir = os.path.join(out_dir, language, safe_name)
            os.makedirs(speaker_dir, exist_ok=True)
            written = 0
            for j, text in enumerate(chosen):
                audio, audio_lens = model.do_tts(transcript=text, language=language, speaker_index=index)
                audio_np = audio[0, : int(audio_lens[0])].detach().cpu().float().numpy()
                if audio_np.size == 0:
                    logging.warning(f"  [{language}] {name} (idx {index}) sample {j}: empty audio, skipped.")
                    continue
                sf.write(os.path.join(speaker_dir, f"sample_{j}.wav"), audio_np, model.output_sample_rate)
                written += 1
            logging.info(f"  [{language}] {name} (idx {index}) -> {speaker_dir} ({written} samples)")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not os.path.isfile(args.baked_ckpt):
        raise ValueError(f"Baked model not found: {args.baked_ckpt}")

    if args.baked_ckpt.endswith(".nemo"):
        config = ModelLoadConfig(nemo_file=args.baked_ckpt, codecmodel_path=args.codecmodel_path)
    else:
        if not args.hparams_file:
            raise ValueError("--hparams-file is required when --baked-ckpt is a .ckpt (not for a .nemo).")
        config = ModelLoadConfig(
            hparams_file=args.hparams_file, checkpoint_file=args.baked_ckpt, codecmodel_path=args.codecmodel_path
        )
    logging.info("Loading baked model ...")
    model, _ = load_magpie_model(config, device=args.device)
    if not model.has_baked_context_embedding:
        raise ValueError(f"{args.baked_ckpt} has no baked context embedding -- bake it first.")

    # Resolve the speaker roster: explicit --sidecar, else the sidecar next to the model, else the
    # speaker_map embedded inside the .nemo (model.cfg.speaker_map points to the extracted artifact).
    sidecar = args.sidecar or f"{args.baked_ckpt}.speakers.json"
    if os.path.isfile(sidecar):
        roster = load_roster_from_sidecar(sidecar)
        logging.info(f"Speakers from sidecar {sidecar}")
    elif model.cfg.get("speaker_map") and os.path.isfile(model.cfg.speaker_map):
        roster = load_roster_from_sidecar(model.cfg.speaker_map)
        logging.info(f"Speakers from embedded .nemo speaker_map {model.cfg.speaker_map}")
    else:
        raise ValueError(f"No speaker map found (looked for {sidecar} and the embedded speaker_map). Pass --sidecar.")
    logging.info(f"index -> name: {[(i, n) for n, i in roster]}")
    if model.num_baked_speakers != len(roster):
        logging.warning(
            f"num_baked_speakers={model.num_baked_speakers} != roster count {len(roster)}; "
            "the speaker map may be stale for this checkpoint."
        )

    # Use the standard tuned inference parameters (same as examples/tts/magpietts_inference.py). The
    # config's defaults leave the attention-prior layer lists unset, so do_tts would otherwise run
    # with the prior off and terminate almost immediately. do_tts reads these from model.inference_parameters.
    model.inference_parameters = ModelInferenceParameters(
        max_decoder_steps=args.max_decoder_steps,
        temperature=args.temperature,
        topk=args.topk,
        cfg_scale=args.cfg_scale,
        apply_attention_prior=True,
        attention_prior_epsilon=0.1,
        attention_prior_lookahead_window=5,
        estimate_alignment_from_layers=parse_int_list(args.estimate_alignment_from_layers),
        apply_prior_to_layers=parse_int_list(args.apply_prior_to_layers),
        start_prior_after_n_audio_steps=0,
        ignore_finished_sentence_tracking=True,
        eos_detection_method="argmax_or_multinomial_any",
    )
    logging.info(
        f"Inference params: temperature={args.temperature}, topk={args.topk}, cfg_scale={args.cfg_scale}, "
        f"estimate_alignment_from_layers={model.inference_parameters.estimate_alignment_from_layers}, "
        f"apply_prior_to_layers={model.inference_parameters.apply_prior_to_layers}"
    )

    languages = resolve_languages(model, args.languages)
    if not languages:
        logging.warning("No languages to synthesize; nothing to do.")
        return

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.baked_ckpt)), "baked_samples")
    logging.info(
        f"Synthesizing: languages={languages}, {args.samples_per_language} sample(s)/speaker/language -> {out_dir}"
    )
    synthesize_samples(model, roster, out_dir, languages, args.samples_per_language)
    logging.info("Done.")


if __name__ == "__main__":
    main()
