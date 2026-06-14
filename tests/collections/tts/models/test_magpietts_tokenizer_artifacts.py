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
Unit tests for ``MagpieTTSModel._register_tokenizer_artifacts`` (.nemo packaging of tokenizer files).

A code-switching tokenizer (e.g. Hindi = ``hi_prondict`` + ``ipa_cmudict``) stores ``g2p.phoneme_dict``
as a LIST of files. The fix registers each list element under a DOT-indexed config key
(``phoneme_dict.{i}``) so the save/restore connector's ``OmegaConf.update`` rewrites that element in
place. The underscore form (``phoneme_dict_{i}``) would instead create sibling keys
``phoneme_dict_0``/``phoneme_dict_1`` and leave ``phoneme_dict`` unresolved -- which ``IpaG2p`` rejects
on restore ("unexpected keyword argument 'phoneme_dict_0'").

The dot-indexed key is the whole contract the connector relies on, so the tests drive the real method
against a mocked model: no codec/encoders/GPU or .nemo round-trip needed.
"""

import os
from unittest.mock import MagicMock

import pytest
from omegaconf import ListConfig, OmegaConf

from nemo.collections.tts.models.magpietts import MagpieTTSModel

_PREFIX = "text_tokenizers.hindi_phoneme.g2p.phoneme_dict"


def _run_registration(phoneme_dict):
    """Run the real registration over a single-tokenizer config; return (cfg, [config_path, ...])."""
    cfg = OmegaConf.create(
        {
            "text_tokenizers": {
                "hindi_phoneme": {
                    "_target_": "IPATokenizer",
                    "g2p": {"_target_": "IpaG2p", "phoneme_dict": phoneme_dict},
                }
            }
        }
    )
    model = MagicMock(spec=MagpieTTSModel)
    # Mirror register_artifact's contract: resolve a src to an existing absolute path.
    model.register_artifact.side_effect = lambda config_path, src, verify_src_exists=True: os.path.abspath(src)

    MagpieTTSModel._register_tokenizer_artifacts(model, cfg)

    registered_keys = [call.args[0] for call in model.register_artifact.call_args_list]
    return cfg, registered_keys


class TestRegisterTokenizerArtifacts:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_list_phoneme_dict_registers_dot_indexed_and_stays_a_list(self):
        cfg, registered_keys = _run_registration(["/data/hi_prondict.dict", "/data/ipa_cmudict.txt"])

        # Each element is registered under a DOT-indexed key -- never the underscore form (the bug).
        assert registered_keys == [f"{_PREFIX}.0", f"{_PREFIX}.1"]
        assert not any("phoneme_dict_0" in key or "phoneme_dict_1" in key for key in registered_keys)

        # phoneme_dict stays a 2-element list of resolved paths (not collapsed to sibling keys / null).
        g2p = cfg.text_tokenizers.hindi_phoneme.g2p
        expected = [os.path.abspath("/data/hi_prondict.dict"), os.path.abspath("/data/ipa_cmudict.txt")]
        assert isinstance(g2p.phoneme_dict, ListConfig)
        assert list(g2p.phoneme_dict) == expected
        assert "phoneme_dict_0" not in g2p and "phoneme_dict_1" not in g2p

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_string_phoneme_dict_registers_under_plain_key(self):
        cfg, registered_keys = _run_registration("/data/en.dict")

        # The common single-file case still registers under the plain key and resolves in place.
        assert registered_keys == [_PREFIX]
        assert cfg.text_tokenizers.hindi_phoneme.g2p.phoneme_dict == os.path.abspath("/data/en.dict")
