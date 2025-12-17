# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Unit and integration tests for temperature-based data reweighting in Lhotse dataloader.
Integration tests mock data loading and verify that weights passed to mux() are correct.
"""
from unittest.mock import MagicMock, patch

import numpy as np
from lhotse import CutSet
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config, temperature_reweighting

# =============================================================================
# Tests for temperature_reweighting function
# =============================================================================


class TestTemperatureReweighting:
    """Tests for the temperature_reweighting function with various input types."""

    # --- Tests with list of 3 items (floats) ---
    def test_three_floats_temperature_1(self):
        """Three float weights with temperature=1.0 preserves ratios."""
        weights = [0.5, 0.3, 0.2]
        result = temperature_reweighting(weights, temperature=1.0)
        np.testing.assert_allclose(result, [0.5, 0.3, 0.2], rtol=1e-7)
        np.testing.assert_allclose(sum(result), 1.0, rtol=1e-7)

    def test_three_floats_temperature_0(self):
        """Three float weights with temperature=0.0 equalizes."""
        weights = [0.5, 0.3, 0.2]
        result = temperature_reweighting(weights, temperature=0.0)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], rtol=1e-7)

    def test_three_floats_temperature_05(self):
        """Three float weights with temperature=0.5."""
        weights = [0.9, 0.09, 0.01]
        result = temperature_reweighting(weights, temperature=0.5)
        assert result[0] > result[1] > result[2]
        assert result[0] < 0.9  # Less dominant than original
        np.testing.assert_allclose(sum(result), 1.0, rtol=1e-7)

    # --- Tests with list of 3 items (integers) ---
    def test_three_integers_temperature_1(self):
        """Three integer weights with temperature=1.0 preserves ratios."""
        weights = [197, 544, 1615]
        result = temperature_reweighting(weights, temperature=1.0)
        total = sum(weights)
        expected = [w / total for w in weights]
        np.testing.assert_allclose(result, expected, rtol=1e-7)
        np.testing.assert_allclose(sum(result), 1.0, rtol=1e-7)

    def test_three_integers_temperature_0(self):
        """Three integer weights with temperature=0.0 equalizes."""
        weights = [197, 544, 1615]
        result = temperature_reweighting(weights, temperature=0.0)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], rtol=1e-7)

    def test_three_integers_temperature_05(self):
        """Three integer weights with temperature=0.5."""
        weights = [100, 400, 1600]
        result = temperature_reweighting(weights, temperature=0.5)
        assert result[0] < result[1] < result[2]
        assert result[2] < 1600 / sum(weights)
        np.testing.assert_allclose(sum(result), 1.0, rtol=1e-7)

    # --- Tests with list of 1 item (float) ---
    def test_single_float_any_temperature(self):
        """Single float weight returns [1.0] regardless of temperature."""
        for temp in [0.0, 0.5, 1.0, 2.0]:
            result = temperature_reweighting([0.7], temperature=temp)
            np.testing.assert_allclose(result, [1.0], rtol=1e-7)

    # --- Tests with list of 1 item (integer) ---
    def test_single_integer_any_temperature(self):
        """Single integer weight returns [1.0] regardless of temperature."""
        for temp in [0.0, 0.5, 1.0, 2.0]:
            result = temperature_reweighting([1000], temperature=temp)
            np.testing.assert_allclose(result, [1.0], rtol=1e-7)

    # --- Edge cases ---
    def test_empty_list(self):
        """Empty list returns empty list."""
        result = temperature_reweighting([], temperature=1.0)
        assert result == []

    def test_zero_weights_raises_error(self):
        """Zero weights raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            temperature_reweighting([0, 0, 0], temperature=1.0)

    def test_negative_weights_raises_error(self):
        """Negative weights raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            temperature_reweighting([100, -50, 200], temperature=1.0)

    def test_mixed_zero_positive_raises_error(self):
        """Mixed zero and positive weights raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            temperature_reweighting([100, 0, 200], temperature=1.0)


# =============================================================================
# Tests for weights passed to mux() are correct
# =============================================================================


def make_mock_parser_fn(return_tarred=True):
    """Create a mock parser function that returns a mock CutSet."""

    def mock_parser(cfg):
        mock_cuts = MagicMock(spec=CutSet)
        mock_cuts.map = MagicMock(return_value=mock_cuts)
        return mock_cuts, return_tarred

    return mock_parser


class TestMuxWeights:
    """Tests that verify the weights passed to mux() are correctly temperature-reweighted."""

    def test_flat_structure_temperature_0_equalizes_weights(self):
        """Flat structure with temperature=0.0 should equalize weights."""
        config = OmegaConf.create(
            {
                "input_cfg": [
                    {"type": "lhotse_shar", "shar_path": "/fake/path1", "weight": 900},
                    {"type": "lhotse_shar", "shar_path": "/fake/path2", "weight": 100},
                ],
                "reweight_temperature": [0.0],
                "shuffle": True,
                "shard_seed": 42,
            }
        )

        captured_weights = []

        def mock_mux(*cuts, weights=None, **kwargs):
            captured_weights.append(weights)
            return MagicMock(spec=CutSet)

        with patch("nemo.collections.common.data.lhotse.cutset.mux", side_effect=mock_mux):
            with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=make_mock_parser_fn()):
                read_cutset_from_config(config)

        assert len(captured_weights) == 1
        np.testing.assert_allclose(captured_weights[0], [0.5, 0.5], rtol=1e-7)

    def test_flat_structure_temperature_1_preserves_weights(self):
        """Flat structure with temperature=1.0 should preserve original weights."""
        config = OmegaConf.create(
            {
                "input_cfg": [
                    {"type": "lhotse_shar", "shar_path": "/fake/path1", "weight": 900},
                    {"type": "lhotse_shar", "shar_path": "/fake/path2", "weight": 100},
                ],
                "reweight_temperature": [1.0],
                "shuffle": True,
                "shard_seed": 42,
            }
        )

        captured_weights = []

        def mock_mux(*cuts, weights=None, **kwargs):
            captured_weights.append(weights)
            return MagicMock(spec=CutSet)

        with patch("nemo.collections.common.data.lhotse.cutset.mux", side_effect=mock_mux):
            with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=make_mock_parser_fn()):
                read_cutset_from_config(config)

        assert len(captured_weights) == 1
        np.testing.assert_allclose(captured_weights[0], [0.9, 0.1], rtol=1e-7)

    def test_no_reweight_temperature_defaults_to_1(self):
        """Without reweight_temperature, default temp=1.0 preserves weights."""
        config = OmegaConf.create(
            {
                "input_cfg": [
                    {"type": "lhotse_shar", "shar_path": "/fake/path1", "weight": 800},
                    {"type": "lhotse_shar", "shar_path": "/fake/path2", "weight": 200},
                ],
                "shuffle": True,
                "shard_seed": 42,
            }
        )

        captured_weights = []

        def mock_mux(*cuts, weights=None, **kwargs):
            captured_weights.append(weights)
            return MagicMock(spec=CutSet)

        with patch("nemo.collections.common.data.lhotse.cutset.mux", side_effect=mock_mux):
            with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=make_mock_parser_fn()):
                read_cutset_from_config(config)

        assert len(captured_weights) == 1
        np.testing.assert_allclose(captured_weights[0], [0.8, 0.2], rtol=1e-7)

    def test_nested_groups_with_temperature_reweighting(self):
        """
        Nested group structure with reweight_temperature: [1.0, 0.0]

        input_cfg:
          - type: group,
            weight: 197,
            input_cfg:
              - type: lhotse_shar
                weight: 197
          - type: group,
            weight: 2159,
            input_cfg:
              - type: lhotse_shar
                weight: 544
              - type: lhotse_shar
                weight: 1615

        Expected:
        - Level 1: [197/2356, 2159/2356] (temp=1.0)
        - Level 2: [0.5, 0.5] (temp=0.0)
        """
        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "group",
                        "weight": 197,
                        "input_cfg": [
                            {"type": "lhotse_shar", "shar_path": "/fake/italian", "weight": 197},
                        ],
                    },
                    {
                        "type": "group",
                        "weight": 2159,
                        "input_cfg": [
                            {"type": "lhotse_shar", "shar_path": "/fake/ja_cv", "weight": 544},
                            {"type": "lhotse_shar", "shar_path": "/fake/ja_emilia", "weight": 1615},
                        ],
                    },
                ],
                "reweight_temperature": [1.0, 0.0],
                "shuffle": True,
                "shard_seed": 42,
            }
        )

        captured_mux_calls = []

        def mock_mux(*cuts, weights=None, **kwargs):
            captured_mux_calls.append({"weights": weights, "num_cuts": len(cuts)})
            mock_cuts = MagicMock(spec=CutSet)
            mock_cuts.map = MagicMock(return_value=mock_cuts)
            return mock_cuts

        with patch("nemo.collections.common.data.lhotse.cutset.mux", side_effect=mock_mux):
            with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=make_mock_parser_fn()):
                read_cutset_from_config(config)

        # 2 mux calls: level 2 group, then level 1 group
        assert len(captured_mux_calls) == 2

        # Level 2 group: temp=0.0 -> equalized
        np.testing.assert_allclose(captured_mux_calls[0]["weights"], [0.5, 0.5], rtol=1e-7)

        # Level 1 group: temp=1.0 -> preserved
        expected_level1 = [197 / 2356, 2159 / 2356]
        np.testing.assert_allclose(captured_mux_calls[1]["weights"], expected_level1, rtol=1e-7)
