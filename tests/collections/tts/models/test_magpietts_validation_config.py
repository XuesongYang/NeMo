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
Unit tests for MagpieTTSModel.setup_multiple_validation_data method.

Tests both the new config style (with 'datasets' key) and the legacy style
(without 'datasets' key) for backward compatibility.

Config styles tested:
1. New style (lhotse): validation_ds.datasets[].input_cfg with type: lhotse_shar
2. Legacy lhotse style: validation_ds.dataset.input_cfg with type: lhotse_shar (deprecated)
3. Legacy non-lhotse style: validation_ds.dataset._target_: MagpieTTSDataset (deprecated)
"""

import warnings
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo.collections.tts.models.magpietts import MagpieTTSModel


class TestSetupMultipleValidationData:
    """Test cases for MagpieTTSModel.setup_multiple_validation_data method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MagpieTTSModel instance with required methods mocked."""
        model = MagicMock(spec=MagpieTTSModel)
        model._update_dataset_config = MagicMock()
        model.setup_validation_data = MagicMock()
        model._setup_test_dataloader = MagicMock(return_value=MagicMock())
        return model

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_new_style_lhotse_single_dataset(self, mock_model):
        """New lhotse config with single dataset: no warning, merges shared config with dataset overrides."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'batch_duration': 100,  # Shared config
                'num_workers': 2,  # Shared config
                'datasets': [
                    {
                        'name': 'val_set_0',
                        'batch_duration': 50,  # Override shared
                        'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data'}],
                    }
                ],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, "New config style should not emit deprecation warning"

        # Should call setup_validation_data with merged config
        mock_model.setup_validation_data.assert_called_once()
        passed_config = mock_model.setup_validation_data.call_args[0][0]
        assert passed_config.batch_duration == 50  # Dataset override wins
        assert passed_config.num_workers == 2  # Shared config preserved

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_new_style_lhotse_multiple_datasets(self, mock_model):
        """New lhotse config with multiple datasets: creates multiple dataloaders, assigns default names if missing."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'batch_duration': 100,
                'datasets': [
                    {'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data0'}]},  # No name
                    {'name': 'custom_name', 'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data1'}]},
                ],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, "New config style should not emit deprecation warning"

        # Should call _setup_test_dataloader twice (once per dataset)
        assert mock_model._setup_test_dataloader.call_count == 2
        assert isinstance(mock_model._validation_dl, list)
        assert len(mock_model._validation_dl) == 2
        # First dataset gets default name, second uses explicit name
        assert mock_model._validation_names == ['val_set_0', 'custom_name']

    # ==================== Legacy Style Tests (deprecated, with 'dataset' key) ====================

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_legacy_lhotse_style_deprecated_but_functional(self, mock_model):
        """Legacy lhotse config (dataset.input_cfg with lhotse_shar) emits warning but still works."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'volume_norm': True,
                'dataset': {
                    'batch_duration': 100,
                    'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data'}],
                },
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1, "Legacy lhotse config should emit deprecation warning"
            assert "deprecated" in str(deprecation_warnings[0].message).lower()
            assert "datasets" in str(deprecation_warnings[0].message).lower()

        # Should still call setup_validation_data with original config for backward compatibility
        mock_model.setup_validation_data.assert_called_once_with(config)
        passed_config = mock_model.setup_validation_data.call_args[0][0]
        assert passed_config.use_lhotse is True
        assert 'dataset' in passed_config
        assert passed_config.dataset.input_cfg[0].type == 'lhotse_shar'

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_legacy_non_lhotse_style_deprecated_but_functional(self, mock_model):
        """Legacy non-lhotse config (dataset._target_: MagpieTTSDataset) emits warning but still works."""
        # Config structure from magpietts.yaml (non-lhotse style)
        config = OmegaConf.create(
            {
                'dataset': {
                    '_target_': 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset',
                    'dataset_meta': '/path/to/meta.yaml',
                },
                'dataloader_params': {'batch_size': 16},
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1, "Legacy non-lhotse config should emit deprecation warning"
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

        # Should still call setup_validation_data with original config
        mock_model.setup_validation_data.assert_called_once_with(config)
        passed_config = mock_model.setup_validation_data.call_args[0][0]
        assert passed_config.dataset._target_ == 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset'
        assert passed_config.dataloader_params.batch_size == 16

    # ==================== Error Case Tests ====================

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_datasets_raises_error(self, mock_model):
        """Empty 'datasets' list should raise ValueError."""
        config = OmegaConf.create({'use_lhotse': True, 'datasets': []})

        with pytest.raises(ValueError) as exc_info:
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        assert "empty or invalid" in str(exc_info.value).lower()
