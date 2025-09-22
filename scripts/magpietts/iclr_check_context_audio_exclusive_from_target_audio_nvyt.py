from pathlib import Path
import logging
from lhotse.serialization import load_jsonl
from tqdm import tqdm
from typing import Dict, Any, Tuple


def get_recording_id(relative_path: str) -> str:
    """Generate a recording ID from the relative audio path."""
    return "rec-" + relative_path.rsplit(".", 1)[0].replace("/", "-")


def process_manifest_entry(entry: Dict[str, Any]) -> Tuple[str, str] | None:
    target_audio_path_relative = entry.get("audio_filepath")
    target_audio_offset_in_seconds = entry.get("offset", 0.0)
    target_audio_duration = entry.get("duration")
    context_audio_path_relative = entry.get("context_audio_filepath")
    context_audio_duration = entry.get("context_audio_duration")
    context_audio_offset_in_seconds = entry.get("context_audio_offset", 0.0)

    # Create IDs
    target_recording_id = get_recording_id(target_audio_path_relative)
    context_recording_id = get_recording_id(context_audio_path_relative)

    # Create cut id
    target_cut_id = f"cut-{target_recording_id}-{target_audio_offset_in_seconds:.2f}-{target_audio_duration:.2f}"
    context_cut_id = f"cut-{context_recording_id}-{context_audio_offset_in_seconds:.2f}-{context_audio_duration:.2f}"

    return target_cut_id, context_cut_id


if __name__ == "__main__":
    nemo_manifest_path = Path("nemo_manifest/extend_nemo_manifest_with_context_audio/nvyt2505_train_minDur1.0_withContextAudioMinDur3.0MinSSIM0.6_removedMismatchedTranscripts_shuffled.json")
    logging.info(f"Reading NeMo manifest lazily from: {nemo_manifest_path}")
    manifest_iterable = load_jsonl(nemo_manifest_path)

    target_audio_cut_ids_filepath = Path("target_audio_cut_ids.txt")
    context_audio_cut_ids_filepath = Path("context_audio_cut_ids.txt")

    with open(target_audio_cut_ids_filepath, "w") as tf, open(context_audio_cut_ids_filepath, "w") as cf:
        for entry in tqdm(manifest_iterable, desc="Processing Entries"):
            target_cut_id, context_cut_id = process_manifest_entry(entry)
            tf.write(f"{target_cut_id}\n")
            cf.write(f"{context_cut_id}\n")