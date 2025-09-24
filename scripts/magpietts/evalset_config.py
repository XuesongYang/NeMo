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
dataset_meta_info = {
    'riva_hard_digits': {
        'manifest_path': '/Data/evaluation_manifests/hard-digits-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_letters': {
        'manifest_path': '/Data/evaluation_manifests/hard-letters-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_money': {
        'manifest_path': '/Data/evaluation_manifests/hard-money-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_short': {
        'manifest_path': '/Data/evaluation_manifests/hard-short-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'vctk': {
        'manifest_path': '/Data/evaluation_manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths_silence_trimmed.json',
        'audio_dir': '/Data/VCTK-Corpus-0.92',
        'feature_dir': '/Data/VCTK-Corpus-0.92',
    },
    'libritts_seen': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_seen_evalset_from_testclean_v2.json',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    'libritts_test_clean': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_test_clean_withContextAudioPaths.jsonl',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    # We need an4_val_ci just for CI tests
    'an4_val_ci': {
        'manifest_path': '/home/TestData/an4_dataset/an4_val_context_v1.json',
        'audio_dir': '/',
        'feature_dir': None,
    },

    ############# ICLR26 #############
    # full set
    'CML-TTS_dev_IT_original': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/IT/evaluation_manifests/CML-TTS_dev_IT_dev_original_withContextAudioMinDur3.0MinSSIM0.6.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/IT/audio_extracted',
        'tokenizer_names': ['italian_chartokenizer'],
        'whisper_language': 'italian',
    },
    'CML-TTS_dev_NL_original': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/NL/evaluation_manifests/CML-TTS_dev_NL_dev_original_withContextAudioMinDur3.0MinSSIM0.6.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/NL/audio_extracted',
        'tokenizer_names': ['dutch_chartokenizer'],
        'whisper_language': 'dutch',
    },
    'CML-TTS_dev_IT_enhanced': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/IT/evaluation_manifests/CML-TTS_dev_IT_dev_enhanced_withContextAudioMinDur3.0MinSSIM0.6.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/IT/audio_extracted',
        'tokenizer_names': ['italian_chartokenizer'],
        'whisper_language': 'italian',
    },
    'CML-TTS_dev_NL_enhanced': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/NL/evaluation_manifests/CML-TTS_dev_NL_dev_enhanced_withContextAudioMinDur3.0MinSSIM0.6.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/NL/audio_extracted',
        'tokenizer_names': ['dutch_chartokenizer'],
        'whisper_language': 'dutch',
    },
     'LibriTTS_test_clean_original': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/LibriTTS_test_clean/LibriTTS_test_clean_withContextAudioPaths.jsonl',
        'audio_dir': '/mnt/sdb/xueyang_data/LibriTTS',
        'tokenizer_names': ['english_chartokenizer'],
        'whisper_language': 'en',
    },

    # subset with 100 records each.
    'CML-TTS_dev_IT_original_subset_100': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/IT/evaluation_manifests/CML-TTS_dev_IT_dev_original_withContextAudioMinDur3.0MinSSIM0.6_100.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/IT/audio_extracted',
        'tokenizer_names': ['italian_chartokenizer'],
        'whisper_language': 'italian',
    },
    'CML-TTS_dev_NL_original_subset_100': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/NL/evaluation_manifests/CML-TTS_dev_NL_dev_original_withContextAudioMinDur3.0MinSSIM0.6_100.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/original/CML-TTS_dev/NL/audio_extracted',
        'tokenizer_names': ['dutch_chartokenizer'],
        'whisper_language': 'dutch',
    },
    'CML-TTS_dev_IT_enhanced_subset_100': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/IT/evaluation_manifests/CML-TTS_dev_IT_dev_enhanced_withContextAudioMinDur3.0MinSSIM0.6_100.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/IT/audio_extracted',
        'tokenizer_names': ['italian_chartokenizer'],
        'whisper_language': 'italian',
    },
    'CML-TTS_dev_NL_enhanced_subset_100': {
        'manifest_path': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/NL/evaluation_manifests/CML-TTS_dev_NL_dev_enhanced_withContextAudioMinDur3.0MinSSIM0.6_100.json',
        'audio_dir': '/mnt/sdb/xueyang_data/iclr_2026/data_prep/enhanced/CML-TTS_dev/NL/audio_extracted',
        'tokenizer_names': ['dutch_chartokenizer'],
        'whisper_language': 'dutch',
    },

}
