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

Config styles tested:
1. Single dataset: validation_ds.datasets with one entry
2. Multiple datasets: validation_ds.datasets with multiple entries
3. Error cases: missing or empty 'datasets' key
"""

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
        model._setup_test_dataloader = MagicMock(return_value=MagicMock())
        return model

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_single_dataset_merges_shared_config(self, mock_model):
        """Single dataset entry merges shared config, strips 'name', and stores dataloader as a list."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'batch_duration': 100,  # Shared config
                'num_workers': 2,  # Shared config
                'datasets': [
                    {
                        'name': 'custom_single_val',
                        'batch_duration': 50,  # Override shared
                        'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data'}],
                    }
                ],
            }
        )

        MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        # Single dataset goes through the same unified loop as multi-dataset
        mock_model._setup_test_dataloader.assert_called_once()
        passed_config = mock_model._setup_test_dataloader.call_args[0][0]
        assert passed_config.batch_duration == 50  # Dataset override wins
        assert passed_config.num_workers == 2  # Shared config preserved
        assert 'name' not in passed_config  # 'name' stripped before dataloader setup
        assert isinstance(mock_model._validation_dl, list)
        assert len(mock_model._validation_dl) == 1
        assert mock_model._validation_names == ['custom_single_val']

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_multiple_datasets_with_default_and_custom_names(self, mock_model):
        """Multiple dataset entries create separate dataloaders and assign default names when unspecified."""
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

        MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        # Should call _setup_test_dataloader twice (once per dataset)
        assert mock_model._setup_test_dataloader.call_count == 2
        assert isinstance(mock_model._validation_dl, list)
        assert len(mock_model._validation_dl) == 2
        # First dataset gets default name, second uses explicit name
        assert mock_model._validation_names == ['val_set_0', 'custom_name']

    # ==================== Error Case Tests ====================

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_missing_datasets_key_raises_value_error(self, mock_model):
        """Config without 'datasets' key raises ValueError."""
        config = OmegaConf.create({'use_lhotse': True, 'batch_duration': 100})

        with pytest.raises(ValueError) as exc_info:
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        assert "datasets" in str(exc_info.value).lower()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_datasets_list_raises_value_error(self, mock_model):
        """Empty 'datasets' list raises ValueError."""
        config = OmegaConf.create({'use_lhotse': True, 'datasets': []})

        with pytest.raises(ValueError) as exc_info:
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        assert "empty or invalid" in str(exc_info.value).lower()
