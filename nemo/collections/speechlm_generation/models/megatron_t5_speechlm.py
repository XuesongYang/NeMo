import re
import copy
import torch
from torch import Tensor
from hydra.utils import instantiate
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Dict, List, Any

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.utils import logging

from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import OptimizerModule, MegatronOptimizerModule

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishPhonemesTokenizer, IPATokenizer

from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model import get_pseudo_tokens
from nemo.collections.tts.data.speechllm.t5_speechllm_dataset import _get_default_text_tokenizer_conf, TextTokenizer, EnglishIpaTextTokenizer

from nemo.collections.speechlm.models.base import SpeechLanguageModel
from nemo.collections.speechlm_generation.modules.prompt_table import VirtualPromptPlaceholderToken, VirtualPromptSource
from nemo.collections.llm import fn, T5Model, T5Config, T5Config220M


def get_pseudo_tokens(num_virtual_tokens):
    """
    Takes in an integer and returns a list of strings where each string
    is a numbered virtual token placeholder. If
    num_virtual_tokens = 3, then this function returns:

    ["<prompt_0>", "<prompt_1>", "<prompt_2>"]

    Args:
        num_virtual_tokens: (int) Number of virtual token strings you want to make

    returns a list of string.

    """
    pseudo_tokens = [
        VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
        for i in range(num_virtual_tokens)
    ]

    return pseudo_tokens

@dataclass
class MCoreSpeechT5ModuleConfig(T5Config):
    # copied from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder.MegatronTokenLevelEncoderDecoderSpeechLLMModule.__init__
    seq_pattern: str = "parallel"
    speech_head_type: str = "token_level"
    attn_prior_scaledown_start_step: int = 10000
    attn_prior_end_step: int = 11000
    use_alignment_loss: bool = False
    return_all_crossattention_probs: bool = False
    logging_step: bool = False
    num_cross_attention_heads: int = 12  # 12 for 220m T5, 16 for 11b T5
    enc_output_to_layers: Optional[List[List[int]]] = None


@dataclass
class SpeechT5Config(TransformerConfig, io.IOMixin):
    train_from_scratch: bool = True
    language_model_path: Optional[str] = None
    # it is possible to override keys here:
    #   attention_dropout, ffn_hidden_size, hidden_size, hidden_dropout, kv_channels, num_attention_heads, num_layers

    model_type: ModelType = ModelType.encoder_and_decoder
    attn_prior_scaledown_start_step: int = 10000
    attn_prior_end_step: int = 11000
    num_cross_attention_heads: int = 12
    lm_vocab_size: int = 30000
    context_pattern: str = "parallel"
    context_conditioning: str = "decoder"
    context_duration_min: float = 2.9
    context_duration_max: float = 2.9
    codebook_fps: int = 86
    decoder_context_len: int = 0

    speech_offset: int = 30000
    speech_codebook_size: int = 1024
    num_speech_codebooks: int = 8
    codecmodel_type: str = "nemo_codec"
    enc_output_to_layers: Optional[List[List[int]]] = None

    english_only_model: bool = True
    phoneme_tokenizer: Optional[Union[EnglishPhonemesTokenizer, IPATokenizer]] = None

    # task template config
    task_templates: Optional[List[DictConfig]] = None
    task_templates_dict: Optional[Dict] = None
    task_id_num_to_name: Optional[Dict] = None
    max_virtual_tokens: Optional[int] = None
    new_tasks: List[str] = field(default_factory=lambda: ['squad'])
    total_new_task_virtual_tokens: Optional[int] = None

    # pseudo tokens
    pseudo_tokens: Optional = None


    enc_dec_model_config: Optional[MCoreSpeechT5ModuleConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.context_conditioning == "decoder":
            assert self.context_duration_min == self.context_duration_max, "Decoder context duration must be fixed"
            self.decoder_context_len = int(self.codebook_fps * self.context_duration_min)

        if self.enc_output_to_layers is not None:
            # Convert from listconfig to list
            self.enc_output_to_layers = [ [l for l in encoder_layer] for encoder_layer in self.enc_output_to_layers ]

        if self.task_templates is not None:
            self.load_task_templates()

    def load_task_templates(self):
        """
        copied from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model.MegatronBaseSpeechLM.load_task_templates

        Takes in the task template portion of the config and turns
        it into a table where each task's prompt template and
        the number of virtual tokens to insert in a given part of
        the prompt template are specified.
        """
        task_id_num = 0
        for task in self.task_templates:
            self.task_templates_dict[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", False),
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
                "total_virtual_tokens": task.total_virtual_tokens,
                "virtual_token_splits": task.virtual_token_splits,
                "task_id_num": task_id_num,
            }

            self.max_virtual_tokens = max(self.max_virtual_tokens, task.total_virtual_tokens)
            self.task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1

        # Check that all new tasks have the same total num virtual tokens
        # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
        if self.new_tasks:
            new_task_name = self.new_tasks[0]
            self.total_new_task_virtual_tokens = self.task_templates_dict[new_task_name]["total_virtual_tokens"]

            assert all(
                self.task_templates_dict[taskname]["total_virtual_tokens"] == self.total_new_task_virtual_tokens
                for taskname in self.new_tasks
            ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."



    def prepare_pseudo_tokens(self):
        """
        copied from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model.MegatronBaseSpeechLM.init_model

        Prepare pseudo token ids for virtual/virtual prompt tokens
        """
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            self.tokenizer.add_special_tokens(self.pseudo_tokens)
        else:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0] if self.pseudo_token_ids else None
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

    def configure_model(self, tokenizer):
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state
        from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model

        encoder_config = copy.deepcopy(self)
        encoder_config.num_layers = self.encoder_num_layers
        if self.pipeline_model_parallel_size > 1:
            assert self.encoder_pipeline_model_parallel_size > 0, "Need to know how to shard the encoder & decoder."
            encoder_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(encoder_config=encoder_config, decoder_config=self)

        # model = MCoreT5Model(
        #     config=self,
        #     encoder_config=encoder_config,
        #     transformer_encoder_layer_spec=transformer_layer_spec[0],
        #     transformer_decoder_layer_spec=transformer_layer_spec[1],
        #     vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
        #     max_sequence_length=self.max_position_embeddings,
        #     fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
        #     parallel_output=self.parallel_output,
        #     share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
        #     position_embedding_type=self.position_embedding_type,
        #     rotary_percent=self.rotary_percent,
        #     seq_len_interpolation_factor=self.seq_len_interpolation_factor,
        #     pre_process=parallel_state.is_pipeline_first_stage(),
        #     post_process=parallel_state.is_pipeline_last_stage(),
        # )

        model = MCoreT5SpeechGenerationModule(
            config=self,
            tokenizer=tokenizer
        )
        return model

class MCoreT5SpeechGenerationModule(MegatronModule, fn.FNMixin):
    """
    This module replaces nemo1.0's self.frozen_model.enc_dec_model, i.e. MegatronTokenLevelEncoderDecoderSpeechLLMModule
    """
    def __init__(self, config: MCoreSpeechT5ModuleConfig, tokenizer):
        super().__init__(config=config)

    def forward(
        self,
        enc_input_ids=None,
        enc_attn_mask=None,
        dec_input_ids=None,
        dec_attn_mask=None,
        token_type_ids=None,
        labels=None,
        batch_data=None,  # additional data to be passed to hiddens module
        enc_output=None,  # Result of running the entire encoder
        enc_output_attn_mask=None,
        enc_input=None,  # Result of running encoder embedding only
        output_enc_hidden_only=False,
        speech_mask=None,
        cross_attention_prior=None,
        text_limits=None,
        global_step=None,
        set_inference_key_value_memory=False,
        decoder_max_sequence_len=None,
        encoder_max_sequence_len=None,
    ):
        """
        copied from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder.MegatronTokenLevelEncoderDecoderSpeechLLMModule.forward
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        (
            encoder_self_attention_relative_position_bias,
            decoder_self_attention_relative_position_bias,
            decoder_cross_attention_relative_position_bias,
        ) = (None, None, None)

        if enc_input is not None and enc_output is not None:
            raise ValueError(
                """Both enc_input and enc_output are not None.
                You should only be passing one of them.
                enc_input is the result of the encoder embedding layer
                enc_output is the result of running the entire transformer encoder."""
            )

        # In order of precedence, we use enc_output, enc_input, and then enc_input_ids to determine the encoder sequence length.
        if enc_output is not None:
            # If enc_output is provided in `batch_for_pipeline`, we need to transpose it from [B x S x H] -> [S x B x H].
            if isinstance(enc_output, list):
                encoder_self_attention_relative_position_bias = [None for _ in enc_output]
                enc_output = [x.transpose(0, 1) for x in enc_output]
                enc_seq_length = [x.size(0) for x in enc_output]
            else:
                enc_output = enc_output.transpose(0, 1)
                enc_seq_length = enc_output.size(0)
        elif enc_input is not None:
            # If enc_input is provided, we need to transpose it from [B x S x H] -> [S x B x H].
            if isinstance(enc_input, list):
                encoder_self_attention_relative_position_bias = [None for _ in enc_input]
                enc_input = [x.transpose(0, 1) for x in enc_input]
                enc_seq_length = [x.size(0) for x in enc_input]
            else:
                enc_input = enc_input.transpose(0, 1)
                enc_seq_length = enc_input.size(0)
        # Only need to run encoder embedding and position ids if enc_input or enc_output is not provided.
        elif enc_input_ids is not None:
            assert False, "This should not be reached for speech models"
            enc_seq_length = enc_input_ids.size(1)
            if self.pre_process and self.add_encoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.encoder_relative_position_embedding is None:
                    enc_input_ids_p = enc_input_ids[:, 0, :] if enc_input_ids.dim() == 3 else enc_input_ids
                    enc_position_ids = build_position_ids(enc_input_ids_p)
                else:
                    enc_position_ids = None
                enc_input = self.encoder_embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
                if self.is_adapter_available():
                    _sq, _bs, _hs = enc_input.size()
                    ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
                    v = ptuning_adapter.virtual_tokens
                    if (
                            ptuning_adapter and _sq >= v
                    ):  # The sequence should be longer the v to insert virtual embeddings.
                        virtual_embeddings = ptuning_adapter(_bs)
                        enc_input = enc_input[
                                    v:, :, :
                                    ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                        enc_input = torch.concat([virtual_embeddings, enc_input], dim=0)
            else:
                enc_input = None
        else:
            assert False, "This should not be reached for speech models"
            # This should only happen with PP > 1 for enc-dec prompt learning models
            enc_seq_length = enc_attn_mask.size(1)

        if self.add_encoder and self.encoder_relative_position_embedding is not None:
            assert False, "Not implemented for speech models yet."
            encoder_self_attention_relative_position_bias = self.encoder_relative_position_embedding(
                query_seq_length=enc_seq_length, key_seq_length=enc_seq_length,
            )

        if output_enc_hidden_only:
            assert False, "Not implemented for speech models yet."
            # When pipeline parallel > 1 we need to make sure encoder exist (will be missing in decoder)
            # SpeechT5 should not go here for inference
            if enc_output is None and self.enc_dec_model.encoder is not None:
                enc_output = self.enc_dec_model.encode(
                    enc_input=enc_input,
                    enc_attn_mask=enc_attn_mask,
                    enc_layer_past=None,
                    enc_get_key_value=False,
                    enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                    batch_data=batch_data,
                )
            else:
                enc_output = self.enc_dec_model.encoder_hidden_state

            return enc_output
        else:
            if enc_output_attn_mask is None:
                enc_output_attn_mask = enc_attn_mask

            if self.pre_process and self.add_decoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.decoder_relative_position_embedding is None:
                    dec_input_ids_p = dec_input_ids[:, 0, :] if dec_input_ids.dim() == 3 else dec_input_ids
                    dec_position_ids = build_position_ids(dec_input_ids_p)
                else:
                    dec_position_ids = None
                dec_input = self.get_decoder_embeddings(dec_input_ids, dec_position_ids, token_type_ids)
                if not set_inference_key_value_memory and (decoder_max_sequence_len or encoder_max_sequence_len):
                    # In inference
                    # On step 0 when set_inference_key_value_memory is True, we need all inputs in case
                    # we are using decoder context
                    # Else on step >= 1, only need last input
                    logging.debug("Clipping dec_input and only keep the last input.")
                    dec_input = dec_input[-1, :, :].unsqueeze(0)  # shape (b, embed_dim)
            else:
                # Note: This is when the decoder itself is split across PP ranks.
                dec_input = None

            if self.add_decoder and self.decoder_relative_position_embedding is not None:
                assert False, "This should not be reached."
                decoder_self_attention_relative_position_bias = self.decoder_relative_position_embedding(
                    query_seq_length=dec_input_ids.size(1), key_seq_length=dec_input_ids.size(1)
                )
                if not self.decoder_cfg.relative_position_bias_self_attention_only:
                    decoder_cross_attention_relative_position_bias = self.decoder_cross_attention_relative_position_embedding(
                        query_seq_length=dec_input_ids.size(1), key_seq_length=enc_seq_length,
                    )
                else:
                    decoder_cross_attention_relative_position_bias = None

            return_all_crossattention_probs = self.return_all_crossattention_probs
            single_encoder = False
            if not isinstance(cross_attention_prior, list):
                single_encoder = True
                cross_attention_prior = [cross_attention_prior]

            decoder_cross_attention_relative_position_bias = []
            for _cross_attention_prior in cross_attention_prior:
                _decoder_cross_attention_relative_position_bias = None
                if _cross_attention_prior is not None:
                    # cross_attention_prior shape [B, dec_len, enc_len]
                    # Repeat it to make it [B, 12, dec_len, enc_len]
                    attn_prior_end_step = self.attn_prior_end_step
                    attn_prior_scaledown_start_step = self.attn_prior_scaledown_start_step
                    num_attention_heads = self.num_cross_attention_heads
                    assert attn_prior_scaledown_start_step <= attn_prior_end_step
                    logging.debug(
                        f"attn_prior_scaledown_start_step: {attn_prior_scaledown_start_step}, attn_prior_scaledown_start_step: {attn_prior_end_step}"
                    )
                    if global_step >= attn_prior_end_step:
                        _decoder_cross_attention_relative_position_bias = None
                    elif global_step > attn_prior_scaledown_start_step and global_step < attn_prior_end_step:
                        total_annealing_steps = attn_prior_end_step - attn_prior_scaledown_start_step
                        curr_annealing_step = global_step - attn_prior_scaledown_start_step
                        curr_cross_attention_prior = _cross_attention_prior + (
                                (1.0 - _cross_attention_prior) * curr_annealing_step / total_annealing_steps
                        )
                        _decoder_cross_attention_relative_position_bias = curr_cross_attention_prior.unsqueeze(1).repeat(
                            1, num_attention_heads, 1, 1
                        )
                        _decoder_cross_attention_relative_position_bias = torch.log(
                            _decoder_cross_attention_relative_position_bias + 1e-8)
                    else:
                        _decoder_cross_attention_relative_position_bias = _cross_attention_prior.unsqueeze(1).repeat(
                            1, num_attention_heads, 1, 1
                        )
                        _decoder_cross_attention_relative_position_bias = torch.log(
                            _decoder_cross_attention_relative_position_bias + 1e-8)
                decoder_cross_attention_relative_position_bias.append(_decoder_cross_attention_relative_position_bias)

            return_all_crossattention_probs = return_all_crossattention_probs or self.logging_step

            if single_encoder:
                decoder_cross_attention_relative_position_bias = decoder_cross_attention_relative_position_bias[0]

            output = self.enc_dec_model(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                dec_input=dec_input,
                dec_attn_mask=dec_attn_mask,
                enc_layer_past=None,
                enc_get_key_value=False,
                enc_output=enc_output,
                enc_output_attn_mask=enc_output_attn_mask,
                dec_layer_past=None,
                dec_get_key_value=False,
                enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                dec_self_attention_relative_position_bias=decoder_self_attention_relative_position_bias,
                dec_cross_attention_relative_position_bias=decoder_cross_attention_relative_position_bias,
                return_all_crossattention_probs=return_all_crossattention_probs,
                batch_data=batch_data,
                set_inference_key_value_memory=set_inference_key_value_memory,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
                enc_output_to_layers=self.enc_output_to_layers
            )

            alignment_loss = None
            if self.post_process and self.add_decoder:
                dec_output, enc_output = output  # [s, b, h]
                if return_all_crossattention_probs:
                    dec_output, attention_scores = dec_output
                    attention_probs = [torch.softmax(attention_score, dim=-1) for lidx, attention_score in
                                       enumerate(attention_scores) if lidx in self.alignment_decoder_layerids]

                    if text_limits is not None and self.use_alignment_loss and hasattr(self, "forward_sum_loss"):
                        attention_scores_filtered = [
                            attention_scores[lidx] for lidx in self.alignment_decoder_layerids
                        ]
                        attention_scores_combined = torch.cat(attention_scores_filtered, dim=1)
                        text_start_idx = text_limits[0, 0].item()
                        assert torch.all(
                            text_limits[:, 0] == text_start_idx
                        )  # all texts should start at the same index
                        end_offset = self.alignment_text_end_offset
                        # align_every_n_head: e.g. if set to 2, will skip every other head
                        # if set to 12, will select 1 head from every layer
                        align_every_n_head = self.align_every_n_head
                        dec_start_idx = self.decoder_context_len + 1  # +1 to remove bos
                        attention_scores_sliced = attention_scores_combined[
                                                  :, ::align_every_n_head, dec_start_idx:, text_start_idx:-(2 + end_offset)
                                                  ]  # -2 to remove eos and pad
                        attention_logprobs = (
                            attention_scores_sliced  # not taking log_softmax, since we will do that in loss function
                        )
                        attention_logprobs = torch.mean(attention_logprobs, dim=1, keepdim=True)
                        dec_len = torch.sum(dec_attn_mask, dim=1) - dec_start_idx
                        enc_len = text_limits[:, 1] - text_limits[:, 0] - end_offset
                        alignment_loss = self.forward_sum_loss(
                            attn_logprob=attention_logprobs, in_lens=enc_len, out_lens=dec_len
                        )
                else:
                    attention_probs = None
                # project decoder output to vocabulary-size dimensions
                if self.share_decoder_tokens_head_embeddings:
                    first_layer_vocabsize = (
                            self.speech_offset + self.speech_codebook_size
                    )  # variables set in __init__ of speechlm model
                    token_logits = self.tokens_head(dec_output, self.word_embeddings_weight())  # s, b, vocab
                    if self.seq_pattern in ["parallel", "delay_parallel"]:
                        # For flat seq_pattern we need all the logits
                        token_logits = token_logits[:, :, :first_layer_vocabsize]
                    speech_layers = self.num_speech_codebooks - 1
                    last_layer_output = dec_output
                    last_layer_logits = token_logits

                    # speech_logits_list will be used in loss calculation (parallel output)
                    speech_logits_list = []
                    if self.seq_pattern in ["parallel", "delay_parallel"] and torch.count_nonzero(speech_mask) > 0:
                        for i in range(speech_layers):
                            last_layer_logits = self.speech_tokens_heads[i](dec_output)[0]  # T, B, 1024
                            speech_logits_list.append(last_layer_logits)  # T, B, 1024
                else:
                    token_logits = self.tokens_head(dec_output)[0]  # T, B, WordEmbSize

                if labels is not None:
                    if labels.dim() == 2:
                        # [b, s] -> [s, b]
                        labels = labels.transpose(0, 1).contiguous()
                    elif labels.dim() == 3:
                        # [b, c, s] -> [c, s, b]
                        labels = labels.permute(1, 2, 0).contiguous()

                    # Set label smoothing to 0 if in eval mode.
                    label_smoothing = self.label_smoothing if self.training else 0.0

                    # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                    if self.fp16_cross_entropy:
                        assert token_logits.dtype == torch.half
                        if labels.dim() == 3:
                            raise NotImplementedError("fp16_cross_entropy is not support for labels of dimension 3")
                        tokens_loss = vocab_parallel_cross_entropy(token_logits, labels, label_smoothing)
                    else:
                        if labels.dim() == 2:
                            tokens_loss = vocab_parallel_cross_entropy(token_logits.float(), labels, label_smoothing)
                        elif labels.dim() == 3:
                            if token_logits.size()[0] != labels[0, :, :].size()[0]:
                                raise Exception("TODO: add a permute")
                            tokens_loss = vocab_parallel_cross_entropy(
                                token_logits.float(), labels[0, :, :], label_smoothing
                            )
                            logging.debug(f"token_loss: {tokens_loss}")
                            logging.debug(f"token_loss: {torch.all(torch.isfinite(tokens_loss))}")
                            if (
                                    self.seq_pattern in ["parallel", "delay_parallel"]
                                    and torch.count_nonzero(speech_mask) > 0
                            ):
                                for i in range(speech_layers):
                                    if speech_logits_list[i].size()[0] != labels[i + 1, :, :].size()[0]:
                                        raise Exception("TODO: add a permute")
                                    curr_codebook_loss = (
                                            vocab_parallel_cross_entropy(
                                                speech_logits_list[i].float(), labels[i + 1, :, :], label_smoothing
                                            )
                                            * speech_mask.T
                                    )
                                    tokens_loss += curr_codebook_loss
                                    logging.debug(f"token_loss_{i}: {tokens_loss}")
                                    logging.debug(f"token_loss_{i}: {torch.all(torch.isfinite(tokens_loss))}")

                    # [s, b] -> [b, s]
                    tokens_loss = tokens_loss.transpose(0, 1).contiguous()

                    # check if hiddens is used
                    if self.hiddens_cfg is not None:
                        raise NotImplementedError("Not currently implemented for speechllm")
                    else:
                        return tokens_loss, [token_logits, speech_logits_list, attention_probs, alignment_loss]
                else:
                    # else return token logits (and hiddens if needed)
                    # [s, b, h] -> [b, s, h]
                    # If labels is None then we are in inference mode, and we return the gathered logits
                    if self.parallel_output:
                        # Gather logits from tensor parallel if in parallel_output mode
                        token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                            token_logits
                        )  # T, B, 30208
                        for _i in range(len(speech_logits_list)):
                            speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                speech_logits_list[_i]
                            )  # T, B, 1024

                    token_logits = token_logits.transpose(0, 1).contiguous()  # (B, T, 30208)
                    speech_logits = torch.stack(speech_logits_list, dim=-1)  # T, B, 1024, 7
                    speech_logits = speech_logits.transpose(0, 1).contiguous()  # (B, T, 1024, 7)

                    _si = self.speech_offset
                    _ei = _si + self.speech_codebook_size
                    first_layer_speech_logits = token_logits[:, :, _si:_ei].unsqueeze(-1)  # (b, s, 1023, 1)

                    all_speech_logits = torch.cat(
                        [first_layer_speech_logits, speech_logits], dim=-1
                    )  # (b, s, 1024, 8)

                    if self.hiddens_cfg is not None:
                        raise NotImplementedError("Not currently implemented for speechllm")
                    else:
                        # all_speech_logits: tensor, (b, s, 1024, 8), all layers of speech.
                        # token_logits: tensor, (b, s, vocab_size), text token logits.
                        # speech_logits: tensor, (b, s, 1024, 7), 1-7 layers of speech.
                        # attention_probs: tensor or None, (b, s, )
                        # enc_output: tensor, (virtual_token_len+context_token_len+question_token_len+extra_id_0+[SEP], b, )
                        return all_speech_logits, [token_logits, speech_logits, attention_probs, enc_output]

            elif self.add_decoder and not self.add_encoder:
                decoder_output, _ = output
                return decoder_output
            else:
                encoder_output = output
                return encoder_output
# @dataclass
# class SpeechT5ConfigFromScratch(SpeechT5Config):
#     train_from_scratch: bool = True
#     # it is possible to override keys here:
#     #   attention_dropout, ffn_hidden_size, hidden_size, hidden_dropout, kv_channels, num_attention_heads, num_layers
#
# @dataclass
# class SpeechT5ConfigFromTextT5(SpeechT5Config):
#     train_from_scratch: bool = False
#     language_model_path: str = None
#
#     def __post_init__(self):
#         super().__post_init__()
#         if self.language_model_path is None:
#             raise ValueError(
#                 "T5-TTS SFT on pretrained model checkpoint requires `langauge_model_path` in its config."
#             )
#
# @dataclass
# class T5SpeechConfig220M_(SpeechT5Config):
#     english_only_model: bool = True
#     phoneme_tokenizer: Optional[Union[EnglishPhonemesTokenizer, IPATokenizer]] = None
#     task_templates: Optional[List[DictConfig]] = None
#
#     # speech codecs
#     codecmodel_type: str = "nemo_codec"
#     codebook_fps: int = 86
#     speech_codebook_size: int = 1024
#     num_speech_codebooks: int = 8
#
#     speech_offset: int = 30000
#
#     # cross-attention
#     attn_prior_scaledown_start_step: int = 10000
#     attn_prior_end_step: int = 11000
#     num_cross_attention_heads: int = 12
#
#     lm_vocab_size: int = 30000
#     context_pattern: str = "parallel"
#     context_conditioning: str = "decoder"
#     context_duration_min: float = 2.9
#     context_duration_max: float = 2.9
#     decoder_context_len: int = 0
#     # if context_conditioning == "decoder":
#     #     assert self.context_duration_min == self.context_duration_max, "Decoder context duration must be fixed"
# 	#     decoder_context_len = int(self.codebook_fps * self.context_duration_min)
# 	enc_output_to_layers = cfg.get('enc_output_to_layers', None)
#     if self.enc_output_to_layers is not None:
#         # Convert from listconfig to list
#         enc_output_to_layers = [[l for l in encoder_layer] for encoder_layer in self.enc_output_to_layers]
#
# 	frozen_model.enc_dec_model.speech_offset = speech_offset
# 	frozen_model.enc_dec_model.speech_codebook_size = speech_codebook_size
# 	frozen_model.enc_dec_model.num_speech_codebooks = num_speech_codebooks
# 	frozen_model.enc_dec_model.seq_pattern = cfg.get('seq_pattern', 'parallel')
# 	frozen_model.enc_dec_model.attn_prior_scaledown_start_step = attn_prior_scaledown_start_step
# 	frozen_model.enc_dec_model.attn_prior_end_step = attn_prior_end_step
# 	frozen_model.enc_dec_model.alignment_decoder_layerids = cfg.get('alignment_decoder_layerids',
#                                                                          list(range(0, 12)))
# 	frozen_model.enc_dec_model.return_all_crossattention_probs = cfg.get('return_all_crossattention_probs', False)
# 	frozen_model.enc_dec_model.num_cross_attention_heads = num_cross_attention_heads
# 	frozen_model.enc_dec_model.context_conditioning = self.context_conditioning
# 	frozen_model.enc_dec_model.decoder_context_len = self.decoder_context_len
# 	frozen_model.enc_dec_model.enc_output_to_layers = self.enc_output_to_layers
#
# 	alignment_loss_start_step: int = 0
# 	alignment_loss_end_step: int = float('inf')
# 	use_alignment_loss: bool = False
#     alignment_loss_scale: Optional[float] = None
#
# 	frozen_model.enc_dec_model.use_alignment_loss = True
# 	frozen_model.enc_dec_model.forward_sum_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)
# 	frozen_model.enc_dec_model.alignment_text_end_offset = cfg.get('alignment_text_end_offset', 0)
# 	frozen_model.enc_dec_model.align_every_n_head = cfg.get('align_every_n_head', 1)
# 	alignment_loss_start_step = cfg.get('alignment_loss_start_step', 0)
# 	alignment_loss_end_step = cfg.get('alignment_loss_end_step', float('inf'))
#
#     def __post_init__(self):
#         super().__post_init__()
#         if self.use_alignment_loss:
#             self.alignment_loss_scale = 1.0
#         pass
#
#     def configure_model(self, tokenizer):
#         # copy, paste, and modify from nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model.MegatronT5SpeechLMModel.load_frozen_model
#         language_model_path = self.language_model_path
#         frozen_model = self.frozen_model
#         if not (bool(language_model_path) ^ bool(frozen_model)):
#             raise ValueError(
#                 "T5-TTS requires either 'language_model_path' or 'frozen_model' in its config, but not both."
#             )
#         model =
#
#         return model

class MegatronT5SpeechLMModel(SpeechLanguageModel):
    def __init__(
        self,
        config: SpeechT5Config,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
    ):
        super().__init__()
        self.module = None
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        # self._inference_config = None
        # self._speech_model = self.config.speech_model_config.configure_model()

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        virtual_tokens,
        context_and_question_tokens,
        enc_mask,
        dec_input,
        dec_mask,
        position_ids,
        taskname_ids,
        labels=None,
        speech_mask=None,
        inference=False,
        inference_step=0,
        cross_attention_prior=None,
        text_limits=None,
        decoder_max_sequence_len=None,
        encoder_max_sequence_len=None,
    ) -> torch.Tensor:
        """
        copied from nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model.MegatronT5SpeechLMModel.forward
        """
        if isinstance(context_and_question_tokens, list):
            multi_encoder = True
            assert isinstance(enc_mask, list)
            assert isinstance(position_ids, list)
            if cross_attention_prior is None:
                cross_attention_prior = [None for _ in range(len(context_and_question_tokens))]
            assert isinstance(cross_attention_prior, list)
            assert len(context_and_question_tokens) == len(enc_mask) == len(position_ids) == len(cross_attention_prior)
        else:
            multi_encoder = False
            context_and_question_tokens = [context_and_question_tokens]
            enc_mask = [enc_mask]
            position_ids = [position_ids]
            cross_attention_prior = [cross_attention_prior]

        enc_output = None
        logging.debug(f"self.first_stage_of_pipeline()={self.first_stage_of_pipeline()}\tinference_step={inference_step}")
        if self.first_stage_of_pipeline() and inference_step == 0:
            # Get embeddings for text tokens and insert virtual token embeddings
            encoder_input_list = []
            for ei in range(len(context_and_question_tokens)):
                input_embeds = self.get_embeddings_and_combine(
                    [virtual_tokens, context_and_question_tokens[ei]], taskname_ids, inference
                )
                # TODO: This check needs to be revisited with PP support.
                if hasattr(self.module.encoder_embedding, 'position_embeddings'):
                    position_embeddings = self.module.encoder_embedding.position_embeddings(
                        position_ids[ei]
                    )
                    encoder_input = input_embeds + position_embeddings
                else:
                    encoder_input = input_embeds
                encoder_input_list.append(encoder_input)
        else:
            encoder_input_list = None
            encoder_input = None
            if inference_step != 0:
                enc_output = context_and_question_tokens if multi_encoder else context_and_question_tokens[0]

        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        dec_mask[:, 0] = 1

        if not self.cfg.data.get('use_attention_prior', False):
            cross_attention_prior = [None for _ in range(len(cross_attention_prior))]

        _encoder_input = encoder_input_list
        if not multi_encoder:
            context_and_question_tokens = context_and_question_tokens[0]
            enc_mask = enc_mask[0]
            position_ids = position_ids[0]
            cross_attention_prior = cross_attention_prior[0]
            _encoder_input = encoder_input_list[0] if encoder_input_list is not None else None

        # Call forward on T5 model with preprocessed embeddings
        if inference and inference_step == 0:
            set_inference_key_value_memory = True
        else:
            set_inference_key_value_memory = False

        if self.autocast_dtype == torch.float32:
            output, out_logits = self.module(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=_encoder_input,
                enc_output=enc_output,
                speech_mask=speech_mask,
                cross_attention_prior=cross_attention_prior,
                text_limits=text_limits,
                global_step=self.global_step,
                set_inference_key_value_memory=set_inference_key_value_memory,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output, out_logits = self.module(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=_encoder_input,
                    enc_output=enc_output,
                    speech_mask=speech_mask,
                    cross_attention_prior=cross_attention_prior,
                    text_limits=text_limits,
                    global_step=self.global_step,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    decoder_max_sequence_len=decoder_max_sequence_len,
                    encoder_max_sequence_len=encoder_max_sequence_len,
                )

        return output, encoder_input, out_logits
        # output_tensor = self.module(
        #     encoder_input_ids=encoder_input_ids,
        #     decoder_input_ids=decoder_input_ids,
        #     encoder_attn_mask=encoder_attn_mask,
        #     decoder_attn_mask=decoder_attn_mask,
        #     encoder_decoder_attn_mask=encoder_decoder_attn_mask,
        #     lm_labels=lm_labels,
        #     inference_params=inference_params,
        # )
        #
        # return output_tensor

    def setup(self, stage: Optional[str] = None):
        pass

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        return self.inference_step(batch, mode="validation")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass

    def inference_step(self, batch, mode: str):
        pass

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        pass

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        pass

    def first_stage_of_pipeline(self):
        if self.module.pre_process and parallel_state.get_pipeline_model_parallel_rank() == 0:
            return True
        return False

    def get_embeddings_and_combine(self, token_list, taskname_ids, inference: bool = False):
        """
        copied from nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model.MegatronT5SpeechLMModel.get_embeddings_and_combine
        """
        embedding_list = []
        for tokens in token_list:
            embedding_list.append(self.get_embeddings(tokens, taskname_ids, inference))
        return torch.cat(embedding_list, dim=1)

    def get_embeddings(self, tokens, taskname_ids, inference: bool = False):
        """
        copied from nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model.MegatronT5SpeechLMModel.get_embeddings
        """
        out = None
        if tokens.dim() > 2:
            for i in range(tokens.size()[1]):  # for 8 channels
                if i == 0:
                    # Embed first layer using word embeddings
                    out = self.embed_input(tokens[:, i, :], taskname_ids, inference)  # (B, T, D)
                else:
                    # Embed other layers using speech embeddings
                    cur = self.frozen_model.enc_dec_model.speech_tokens_embeddings[i - 1](tokens[:, i, :])
                    # do not add embeddings of zero tokens of other channels (except the first channel)
                    non_zero_flag = tokens[:, i, :] != 0  # (B, T)
                    cur = cur * non_zero_flag.unsqueeze(2)
                    out = out + cur
        else:
            out = self.embed_input(tokens, taskname_ids, inference)
        return out

    def embed_input(self, input_ids: Tensor, taskname_ids: Tensor, use_cached_reps: bool):
        """
        copied from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model.MegatronBaseSpeechLM.embed_input

        Replaces the virtual tokens in the input_ids with embeddings
        calculated from either the 'prompt_table' or 'prompt_encoder'.
        The virtual token placeholders have token_ids listed in
        `self.pseudo_token_ids`.

        params:
            input_ids: the input token ids
            taskname_ids: the NLP task tag token ids
        returns:
            the token embedding for the LM model.
        """
        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_ids[(input_ids >= self.pseudo_token_ids_start)] = self.pad_token_id
        discrete_token_embeds = self.module.encoder_embedding.word_embeddings(discrete_token_ids).clone()

        # Find the indices where virtual tokens should be inserted
        virtual_token_locations = input_ids >= self.pseudo_token_ids_start

        # If there are no virtual tokens, just return discrete token embeds
        if not virtual_token_locations.any():
            return discrete_token_embeds

        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            # taskname_embeddings = self.word_embeddings(taskname_ids)
            batch_size, _ = taskname_ids.size()
            virtual_token_embeds = self.prompt_encoder(batch_size=batch_size, use_cached_reps=use_cached_reps)
        else:
            raise ValueError("invalid VirtualPromptSource.")

        # Create index template specifying where virtual token embeddings should be placed
        batch_size, _, embedding_size = discrete_token_embeds.shape
        virtual_token_index = virtual_token_locations.nonzero().reshape((batch_size, -1, 2))[:, :, 1][:, :, None]
        virtual_token_index = virtual_token_index.expand(
            batch_size, self.config.total_new_task_virtual_tokens, embedding_size
        )

        # Make sure discrete_token_embeds and virtual_token_embeds share the same dtype
        discrete_token_embeds = discrete_token_embeds.type(virtual_token_embeds.dtype)

        # Insert virtual token embeddings where they belong among the discrete token embeddings
        discrete_token_embeds.scatter_(1, virtual_token_index, virtual_token_embeds)
        input_embeds = discrete_token_embeds

        return input_embeds

__all__ = [
    "MegatronT5SpeechLMModel"
]