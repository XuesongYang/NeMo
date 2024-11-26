# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, List

import pytorch_lightning as pl
import torch
from megatron.core import parallel_state
from omegaconf.omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.multimodal.speech_llm.data.audio_text_dataset import (
    get_audio_text_dataset_from_config,
    get_tarred_audio_text_dataset_from_config,
)
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import TextProcessing
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.modules.common.prompt_table import VirtualPromptSource
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging, model_utils
from nemo.collections.tts.data.speechllm.t5_speechllm_dataset import T5SpeechLMDataset


def build_virtual_prompt_dataset_from_config(
    config: DictConfig,
    tokenizer: Any,
    virtual_prompt_source: VirtualPromptSource,
    task_templates: Dict,
    pseudo_tokens,
    pad_token_id: int,
    lm_vocab_size: int,
    is_train: bool = True,
    seq_pattern: str = "parallel",
    english_only_model: bool = False,
    context_conditioning: str = "decoder",
    use_beta_binomial_interpolator: bool = False,
):
    dataset = T5SpeechLMDataset(
        datasets=config.get("manifest_filepath", None),
        tokenizer=tokenizer,
        sample_rate=config.get('sample_rate', 24000),
        virtual_prompt_source=virtual_prompt_source,
        task_templates=task_templates,
        pseudo_tokens=pseudo_tokens,
        pad_token_id=pad_token_id,
        max_seq_length=config.get('max_seq_length', 1),
        min_seq_length=config.get('min_seq_length', 1),
        add_bos=config.get('add_bos', False),
        add_eos=config.get('add_eos', True),
        decoder_starts_with_pad=config.get('decoder_starts_with_pad', False),
        add_eos_to_decoder_output=config.get('add_eos_to_decoder_output', True),
        add_sentinel_to_input=config.get('add_sentinel_to_input', True),
        ul2_prompt_token=config.get('ul2_prompt_token', None),
        for_train=is_train,
        segment_max_duration=config.get('segment_max_duration', None),
        trim=config.get('trim', None),
        trim_ref=config.get('trim_ref', None),
        trim_top_db=config.get('trim_top_db', None),
        trim_frame_length=config.get('trim_frame_length', None),
        trim_hop_length=config.get('trim_hop_length', None),
        pad_multiple=config.get('pad_multiple', 1),
        pitch_augment=config.get('pitch_augment', None),
        sup_data_path=config.get('sup_data_path', None),
        codec_folder=config.get('codec_folder', None),
        speech_offset=config.get('speech_offset', None),
        train_task=config.get('train_task', "tts"),
        seq_pattern=seq_pattern,
        use_attention_prior=config.get('use_attention_prior', False),
        attention_prior_scaling_factor=config.get('attention_prior_scaling_factor', 1.0),
        cross_attention_epsilon=config.get('cross_attention_epsilon', 0.0),
        lm_vocab_size=lm_vocab_size,
        num_speech_codebooks=config.get("num_speech_codebooks", 8),
        codebook_fps=config.get('codebook_fps', 86),
        add_special_tokens_to_only_first_codebook=config.get(
            'add_special_tokens_to_only_first_codebook', False
        ),
        context_pattern=config.get('context_pattern', 'parallel'),
        context_duration_min=config.get('context_duration_min', 3.0),
        context_duration_max=config.get('context_duration_max', 5.0),
        g2p=config.get('g2p', None),
        skip_datasets=config.get('skip_datasets', []),
        english_only_model=english_only_model,
        use_ipa=config.get('use_ipa', False),
        context_conditioning=context_conditioning,
        use_beta_binomial_interpolator=use_beta_binomial_interpolator,
        context_slice_method=config.get('context_slice_method', 'random'),
        phoneme_probability=config.get('phoneme_probability', 0.5),
        encoder_type=config.get('encoder_type', 'single_transformer'),
    )

    return dataset


class T5SpeechGenerationDataModule(pl.LightningDataModule, IOMixin):
    def __init__(
        self,
        config: Union[DictConfig, Dict],
        tokenizer: TokenizerSpec,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: Dict,
        pseudo_tokens: Any,
        pad_token_id: int,
        lm_vocab_size: int,
        seq_pattern: str = "parallel",
        english_only_model: bool = False,
        context_conditioning: str = "decoder",
        use_beta_binomial_interpolator: bool = False,
    ):
        super().__init__()
        self.cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        self.tokenizer = tokenizer
        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None
        self._validation_names = None
        self._test_names = None
        self.init_global_step = 0
        self.data_sampler = None
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pad_token_id = pad_token_id
        self.lm_vocab_size = lm_vocab_size
        self.seq_pattern = seq_pattern
        self.english_only_model = english_only_model
        self.context_conditioning = context_conditioning
        self.use_beta_binomial_interpolator = use_beta_binomial_interpolator

    def prepare_data(self) -> None:
        # download, tokenize, etc.
        # TODO @xueyang: fill this func.
        pass

    def setup(self, stage: str) -> None:
        # build vocab, perform train/val/test splits, create datasets, apply transforms, etc.
        if stage == 'fit' or stage is None:
            self._train_ds = self._create_dataset('train')
            self._validation_ds = self._create_dataset('validation')
        elif stage == 'validate' or stage is None:
            self._validation_ds = self._create_dataset('validation')
        if stage == 'test' or stage is None:
            self._test_ds = self._create_dataset('test')

        if stage != 'predict':
            self.data_sampler = MegatronDataSampler(
                seq_len=self.cfg.max_seq_length,
                micro_batch_size=self.cfg.micro_batch_size,
                global_batch_size=self.cfg.global_batch_size,
                rampup_batch_size=self.cfg.get("rampup_batch_size", None),
                dataloader_type="batch",  # "batch" should be used for SFT,
            )

            # Follows the calculation in nemo.collections.nlp.data.language_modeling.megatron.
            # base_dataset_utils.get_datasets_weights_and_num_samples
            self.max_train_samples = int(math.ceil(self.cfg.global_batch_size * self.trainer.max_steps * 1.005))

    @lru_cache
    def _create_dataset(self, mode: str):
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataset creation as it is not specified in the config: {self.cfg}")
            return None

        if data_cfg.get("use_lhotse", False):
            NotImplementedError(f"Lhotse support is not ready yet. Fill this code blocks once it is ready.")

        setattr(self, f"_{mode}_names", data_cfg.get('name', None))
        if data_cfg.get("is_tarred", False):
            NotImplementedError(f"Tarred dataset support is not ready yet.")
        else:
            return T5SpeechLMDataset(
                datasets=data_cfg.manifest_filepath,
                tokenizer=self.tokenizer,
                sample_rate=data_cfg.get('sample_rate', 24000),
                virtual_prompt_source=self.virtual_prompt_source,
                task_templates=self.task_templates,
                pseudo_tokens=self.pseudo_tokens,
                pad_token_id=self.pad_token_id,
                max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
                add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
                add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
                ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
                for_train=for_train,
                segment_max_duration=self.cfg.data.get('segment_max_duration', None),
                trim=self.cfg.data.get('trim', None),
                trim_ref=self.cfg.data.get('trim_ref', None),
                trim_top_db=self.cfg.data.get('trim_top_db', None),
                trim_frame_length=self.cfg.data.get('trim_frame_length', None),
                trim_hop_length=self.cfg.data.get('trim_hop_length', None),
                pad_multiple=self.cfg.data.get('pad_multiple', 1),
                pitch_augment=self.cfg.data.get('pitch_augment', None),
                sup_data_path=self.cfg.data.get('sup_data_path', None),
                codec_folder=self.cfg.data.get('codec_folder', None),
                speech_offset=self.cfg.data.get('speech_offset', None),
                train_task=self.cfg.data.get('train_task', "tts"),
                seq_pattern=self.cfg.get('seq_pattern', 'delay_parallel'),
                use_attention_prior=self.cfg.data.get('use_attention_prior', False),
                attention_prior_scaling_factor=self.cfg.data.get('attention_prior_scaling_factor', 1.0),
                cross_attention_epsilon=self.cfg.data.get('cross_attention_epsilon', 0.0),
                lm_vocab_size=self.lm_vocab_size,
                num_speech_codebooks=self.num_speech_codebooks,
                codebook_fps=self.cfg.data.get('codebook_fps', 86),
                add_special_tokens_to_only_first_codebook=self.cfg.data.get(
                    'add_special_tokens_to_only_first_codebook', False
                ),
                context_pattern=self.cfg.data.get('context_pattern', 'parallel'),
                context_duration_min=self.cfg.data.get('context_duration_min', 3.0),
                context_duration_max=self.cfg.data.get('context_duration_max', 5.0),
                g2p=self.cfg.data.get('g2p', None),
                skip_datasets=self.cfg.data.get('skip_datasets', []),
                english_only_model=self.cfg.get('english_only_model', False),
                use_ipa=data_cfg.get('use_ipa', False),
                context_conditioning=data_cfg.get('context_conditioning', "decoder"),
                use_beta_binomial_interpolator=data_cfg.get('use_beta_binomial_interpolator', False),
                context_slice_method=data_cfg.get('context_slice_method', 'random'),
                phoneme_probability=data_cfg.get('phoneme_probability', 0.5),
                encoder_type=data_cfg.get('encoder_type', 'single_transformer'),
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_cfg = self.cfg.get("train_ds", None)
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._train_ds, "train")
        else:
            return self._create_nemo_dataloader(self._train_ds, "train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_cfg = self.cfg.get("validation_ds", None)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._validation_ds, 'validation')
        else:
            if isinstance(self._validation_ds, list):
                if len(self._validation_ds) > 1:
                    return [self._create_nemo_dataloader(ds, 'validation') for ds in self._validation_ds]
                else:
                    return self._create_nemo_dataloader(self._validation_ds[0], 'validation')
            else:
                return self._create_nemo_dataloader(self._validation_ds, 'validation')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        data_cfg = self.cfg.get("test_ds", None)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._test_ds, 'test')
        else:
            if isinstance(self._test_ds, list):
                if len(self._test_ds) > 1:
                    return [self._create_nemo_dataloader(ds, 'test') for ds in self._test_ds]
                else:
                    return self._create_nemo_dataloader(self._test_ds[0], 'test')
            else:
                return self._create_nemo_dataloader(self._test_ds, 'test')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if "predict_ds" not in self.cfg and "test_ds" in self.cfg:
            data_cfg = self.cfg.get("test_ds", None)
            data_key = 'test'
        elif "predict_ds" not in self.cfg and "validation_ds" in self.cfg:
            data_cfg = self.cfg.get("validation_ds", None)
            data_key = 'validation'
        else:
            data_cfg = self.cfg.get("predict_ds", None)
            data_key = 'predict'

        self._test_ds = self._create_dataset(data_key)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._test_ds, 'predict')
        else:
            if isinstance(self._test_ds, list):
                if len(self._test_ds) > 1:
                    return [self._create_nemo_dataloader(ds, 'predict') for ds in self._test_ds]
                else:
                    return self._create_nemo_dataloader(self._test_ds[0], 'predict')
            else:
                return self._create_nemo_dataloader(self._test_ds, 'predict')

    def _create_lhotse_dataloader(self, dataset: Any, mode: str, **kwargs) -> DataLoader:
        NotImplementedError("T5 Lhotse dataloader implementation is under planning.")
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataloader creation as it is not specified in the config: {self.cfg}")
            return None

        if mode == "train":
            return get_lhotse_dataloader_from_config(
                data_cfg,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
                dataset=dataset,
            )
        # for eval, we need to create separate dataset so as to report splitted numbers
        else:
            dls = []
            if hasattr(data_cfg, 'manifest_filepath'):
                manifest_filepath = data_cfg.manifest_filepath
                for cur_manifest_filepath in manifest_filepath:
                    conf = copy.deepcopy(data_cfg)
                    conf['manifest_filepath'] = cur_manifest_filepath
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )
            else:
                input_cfg = data_cfg.input_cfg
                if isinstance(input_cfg, (str, Path)):
                    # Resolve /path/to/input_cfg.yaml into config contents if needed.
                    input_cfg = OmegaConf.load(input_cfg)
                    assert len(input_cfg) == 1, "Only one dataset with multiple manifest paths is supported for eval"
                    data_cfg.input_cfg = input_cfg
                    # for getting names
                    manifest_filepath = [ic.manifest_filepath for ic in input_cfg[0].input_cfg]
                for cur_input_cfg in input_cfg[0].input_cfg:
                    conf = copy.deepcopy(data_cfg)
                    conf.input_cfg[0].input_cfg = [cur_input_cfg]
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )

            if 'name' not in data_cfg:
                names = []
                for cur_manifest_filepath in manifest_filepath:
                    names.append(Path(cur_manifest_filepath).stem)
                OmegaConf.update(data_cfg, 'name', names, force_add=True)
                logging.info(f'Update dataset names as {names}')
            return dls

    def _create_nemo_dataloader(self, dataset: Any, mode: str, **kwargs) -> DataLoader[Any] | None:
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataloader creation as it is not specified in the config: {self.cfg}")
            return None

        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        elif hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries like ChainDataset
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        if isinstance(dataset, torch.utils.data.IterableDataset):
            data_parallel_size = parallel_state.get_data_parallel_world_size()
            num_micro_batches = data_cfg.global_batch_size // (data_cfg.micro_batch_size * data_parallel_size)
            global_batch_size_on_this_data_parallel_rank = num_micro_batches * data_cfg.micro_batch_size
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                shuffle=False,
                batch_size=global_batch_size_on_this_data_parallel_rank,
                drop_last=True,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
            )
            return dataloader

        return DataLoader(
            dataset,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.get("persistent_workers", False),
            collate_fn=collate_fn,
            **kwargs,
        )
