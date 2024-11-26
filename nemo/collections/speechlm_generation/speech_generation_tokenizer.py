from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.utils import logging
from copy import deepcopy
import re
from typing import List
from nemo.core.classes.common import Serialization, typecheck
import omegaconf
from omegaconf.omegaconf import DictConfig, OmegaConf
from nemo.collections.speechlm_generation.utils import get_object_list_from_config
from nemo.collections.speechlm_generation.data import T5SpeechGenerationDataModule
from nemo.collections.speechlm_generation.models.megatron_t5_speechlm import SpeechT5Config, MegatronT5SpeechLMModel
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceSpeechLLMTTSTokenizer, SentencePieceTokenizer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.tts.data.speechllm.t5_speechllm_dataset import _get_default_text_tokenizer_conf
from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model import get_pseudo_tokens
from hydra.utils import instantiate


def _load_task_templates(config: DictConfig):
    """
    copied from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model.MegatronBaseSpeechLM.load_task_templates

    Takes in the task template portion of the config and turns
    it into a table where each task's prompt template and
    the number of virtual tokens to insert in a given part of
    the prompt template are specified.
    """
    task_templates_dict = {}
    task_id_num_to_name = {}
    max_virtual_tokens = 0

    task_id_num = 0
    for task in config.task_templates:
        task_templates_dict[task.taskname] = {
            "prompt_template": task.prompt_template,
            "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
            "answer_only_loss": task.get("answer_only_loss", False),
            "answer_field": task.get("answer_field", None),
            "truncate_field": task.truncate_field,
            "total_virtual_tokens": task.total_virtual_tokens,
            "virtual_token_splits": task.virtual_token_splits,
            "task_id_num": task_id_num,
        }

        max_virtual_tokens = max(max_virtual_tokens, task.total_virtual_tokens)
        task_id_num_to_name[task_id_num] = task.taskname
        task_id_num += 1

    # Check that all new tasks have the same total num virtual tokens
    # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
    # TODO @xueyang: remove this section and `new_tasks` if no other places applied them. Probably legacy codes.
    #   `existing_tasks` can be ignore since it is only used in gpt model. This function implementation can be simplified.
    if config.new_tasks:
        new_task_name = config.new_tasks[0]
        total_new_task_virtual_tokens = task_templates_dict[new_task_name]["total_virtual_tokens"]

        assert all(
            task_templates_dict[taskname]["total_virtual_tokens"] == total_new_task_virtual_tokens
            for taskname in config.new_tasks
        ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."

    return {
        "task_templates": task_templates_dict,
        "task_id_num_to_name": task_id_num_to_name,
        "max_virtual_tokens": max_virtual_tokens,
    }


def _build_tokenizer(config: DictConfig):
    """
    this func is copied and modified from  nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model.MegatronT5OverrideModel._build_tokenizer
    build a static basic tokenizer that would be expanded in future according to the model initialization.

    Default tokenizer is based on available nemo tokenizers.
    Override this method to use an external tokenizer.
    All tokenizers are expected to provide compatible interface.
    Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
    """
    if hasattr(config, "sentencepiece_legacy"):
        legacy = config.sentencepiece_legacy
    else:
        legacy = True if config.library == 'sentencepiece' else False

    if config.library == "sentencepiece":
        tokenizer = SentencePieceSpeechLLMTTSTokenizer(
            model_path=config.get("model_path", None),
            legacy=legacy,
        )
    else:
        tokenizer = get_nmt_tokenizer(
            library=config.library,
            model_name=config.get("type", None),
            tokenizer_model=config.get('model', None),
            vocab_file=config.get('vocab_file', None),
            merges_file=config.get('merges_file', None),
            use_fast=config.get('use_fast', False),
            delimiter=config.get('delimiter', None),
            special_tokens=config.get('special_tokens', None),
            trust_remote_code=config.get('trust_remote_code', False),
            legacy=legacy,
            chat_template=config.get("chat_template", None),
        )

    if config.get('additional_special_tokens', None) is not None:
        tokens_list = omegaconf.OmegaConf.to_object(config.additional_special_tokens)
        tokenizer.add_special_tokens(tokens_list)

    return tokenizer


def _build_vocab(config: DictConfig, tokenizer, dataset_type: str = "t5"):
    """
    copied from nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model._build_vocab
    Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
    TODO: add config to allow to disable it?
    Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size.
    """
    tokenizer = _add_special_tokens_to_tokenizer(
        tokenizer=tokenizer,
        tokenizer_cfg=config,
        dataset_type=dataset_type,
        add_sentinel_tokens_in_reverse_order=config.get("add_sentinel_tokens_in_reverse_order", False),
        add_sentinel_tokens_first=config.get("add_sentinel_tokens_first", False),
    )

    orig_vocab_size = tokenizer.vocab_size
    make_vocab_size_divisible_by = config.get("make_vocab_size_divisible_by", 128)
    tensor_model_parallel_size = config.get("tensor_model_parallel_size", 1)

    padded_vocab_size = orig_vocab_size  # padded_vocab_size
    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    padded_vocab_size = ((padded_vocab_size + multiple - 1) // multiple) * multiple
    logging.info(
        f'Padded vocab_size: {padded_vocab_size}, original vocab_size: {orig_vocab_size}, dummy tokens: {padded_vocab_size - orig_vocab_size}.'
    )

    return tokenizer, padded_vocab_size


def _add_special_tokens_to_tokenizer(
    tokenizer,
    tokenizer_cfg,
    dataset_type="t5",
    add_sentinel_tokens_in_reverse_order=False,
    add_sentinel_tokens_first=False,
):
    """
    copied from nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model.add_special_tokens_to_tokenizer
    """
    # T5-related construction
    if tokenizer_cfg.library == 'huggingface' or tokenizer_cfg.library == 'megatron':
        additional_tokens = {
            'additional_special_tokens': [
                f'<extra_id_{i}>' for i in range(tokenizer_cfg.get('num_sentinel_tokens', 0))
            ]
        }
        if dataset_type == "ul2":
            mask_types = ['r', 's', 'x']
            for mask_type in mask_types:
                additional_tokens['additional_special_tokens'].extend([f'<extra_id_{mask_type}>'])
        if additional_tokens['additional_special_tokens']:
            tokenizer.add_special_tokens(additional_tokens)

    if tokenizer_cfg.library == 'sentencepiece':
        # NOTE: This is an ugly way to support both NeMo-Megatron trained checkpoints and huggingface checkpoints.
        # Huggingface and Google checkpoints will add sentinel tokens first (right after the base vocabulary), but in NeMo-Megatron, we add <cls>, <sep>, <mask>, <pad>, <bos> etc. beofore sentinel tokens <extra_id_xx>.
        if add_sentinel_tokens_first:
            if tokenizer_cfg.get('num_sentinel_tokens', 0) > 0:
                tokenizer = _add_sentinel_tokens(
                    tokenizer, tokenizer_cfg.num_sentinel_tokens, add_sentinel_tokens_in_reverse_order
                )
            tokenizer = _add_base_special_tokens(tokenizer, is_huggingface_converted_model=True)
        else:
            tokenizer = _add_base_special_tokens(tokenizer, is_huggingface_converted_model=False)
            if tokenizer_cfg.get('num_sentinel_tokens', 0) > 0:
                tokenizer = _add_sentinel_tokens(
                    tokenizer, tokenizer_cfg.num_sentinel_tokens, add_sentinel_tokens_in_reverse_order
                )

        if dataset_type == "ul2":
            for mask_type in ['r', 's', 'x']:
                if len(tokenizer.text_to_ids(f'▁<extra_id_{mask_type}>')) == 1:
                    tokenizer.special_token_to_id[f'<extra_id_{mask_type}>'] = tokenizer.text_to_ids(
                        f'<extra_id_{mask_type}>'
                    )[0]
                else:
                    tokenizer.add_special_tokens([f'<extra_id_{mask_type}>'])

    return tokenizer


def _add_sentinel_tokens(tokenizer, num_sentinel_tokens, add_sentinel_tokens_in_reverse_order):
    """
    copied from nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model._add_sentinel_tokens
    """
    # Special check to see if <extra_id_{}> is already present in the tokenizer. If it is, only modify the additional_special_tokens function.
    for i in range(num_sentinel_tokens):
        if add_sentinel_tokens_in_reverse_order:
            i = num_sentinel_tokens - i - 1
        if len(tokenizer.text_to_ids(f'<extra_id_{i}>')) == 1:
            tokenizer.special_token_to_id[f'<extra_id_{i}>'] = tokenizer.text_to_ids(f'<extra_id_{i}>')[0]
        else:
            tokenizer.add_special_tokens([f'<extra_id_{i}>'])

    return tokenizer


def _add_base_special_tokens(tokenizer, is_huggingface_converted_model):
    """
    copied from nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model._add_base_special_tokens
    """
    # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
    # If cls, sep and mask are not attributes of the tokenizer, add it.
    if not hasattr(tokenizer, 'cls_token'):
        tokenizer.add_special_tokens({'cls_token': '<cls>'})
    if not hasattr(tokenizer.tokenizer, 'sep_id'):
        tokenizer.add_special_tokens({'sep_token': '<sep>'})
    if not hasattr(tokenizer.tokenizer, 'mask_id'):
        tokenizer.add_special_tokens({'mask_token': '<mask>'})

    # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
    if not hasattr(tokenizer, 'pad_token'):
        # TODO: Figure out how to do backward compat with pad_id > 0 and >= 0.
        if is_huggingface_converted_model:
            if hasattr(tokenizer.tokenizer, 'pad_id') and tokenizer.tokenizer.pad_id() >= 0:
                tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.pad_id())
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})
        else:
            if hasattr(tokenizer.tokenizer, 'pad_id') and tokenizer.tokenizer.pad_id() > 0:
                tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.pad_id())
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})
    else:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if not hasattr(tokenizer, 'bos_token'):
        if hasattr(tokenizer.tokenizer, 'bos_id') and tokenizer.tokenizer.bos_id() > 0:
            tokenizer.bos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.bos_id())
        else:
            tokenizer.add_special_tokens({'bos_token': '<bos>'})
    else:
        tokenizer.add_special_tokens({'bos_token': '<s>'})

    if not hasattr(tokenizer, 'eos_token'):
        if hasattr(tokenizer.tokenizer, 'eos_id') and tokenizer.tokenizer.eos_id() > 0:
            tokenizer.eos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.eos_id())
        else:
            tokenizer.add_special_tokens({'eos_token': '<eos>'})
    else:
        tokenizer.add_special_tokens({'eos_token': '</s>'})

    return tokenizer

############### 1. build static basic tokenizer #########################
# TODO @xueyang: should remove english_only_model option and instead let user config language. Providing ipa and arpabet phoneme symbol options.
# TODO @xueyang: better to wrap all these ops into a single function and move the codes to other places, such as nemo/collections/speechlm_generation/tokenizers/.
# TODO @xueyang: `t5_cfg` can be obtained from either MegatronT5Model.restore_from(cfg_language_model_path, trainer=trainer, return_config=True)
#   or cfg.get('frozen_model', None) depending on whether load pretrained text ckpt.
#   1. if load text pretrained model: tokenizer should be the one used by that model. We can ignore this condition for now.
#   2. if load frozen model: tokenizer should be the same as frozen_model.tokenizer when initializing MegatronT5OverrideModel.
#   this is a basic and static tokenizer. It will be expanded by adding new tokens dynamically according to model setup and dataset setup.
# tokenizer = build_tokenizer(cfg["tokenizer"])
# the tokenizer was simplified as below by tracing back nemo.collections.nlp.modules.common.tokenizer_utils.get_nmt_tokenizer
# and according to the yaml config.
# tokenizer = AutoTokenizer(
#     pretrained_model_name=cfg.tokenizer.get("pretrained_model_name", "bert-large-cased"),
#     vocab_file=cfg.tokenizer.get("vocab_file", None),
#     merges_file=cfg.tokenizer.get("merges_file", None),
# )

def build_tokenizer(cfg: DictConfig):
    # static tokenizer
    tokenizer = _build_tokenizer(config=cfg.tokenizer)
    tokenizer, padded_vocab_size = _build_vocab(config=cfg.tokenizer, tokenizer=tokenizer, dataset_type=cfg.data.datatype)

    # introduce new virtual tokens by the tasks.
    # TODO @xueyang: not yet add them to tokenizer. They are added to tokenizer in datasets, but should decouple the changes here from datasets.
    #   ref:  nemo.collections.tts.data.speechllm.t5_speechllm_dataset.T5SpeechLMDataset.load_data
    task_templates = _load_task_templates(config=cfg.tasks)

    # introduce pseudo tokens and add to tokenizer
    pseudo_tokens = get_pseudo_tokens(task_templates["max_virtual_tokens"])
    if isinstance(tokenizer, SentencePieceTokenizer):
        tokenizer.add_special_tokens(pseudo_tokens)
    else:
        tokenizer.add_special_tokens({'additional_special_tokens': pseudo_tokens})

    # introduce phone tokens and add to tokenizer if multiple languages are applied.
    # TODO @xueyang: should remove english_only_model option and instead let user config language. Providing ipa and arpabet phoneme symbol options.
    if not cfg.tokenizer.get('english_only_model', False):
        tokenizer.add_phone_tokens_to_special_tokens()

    # build phoneme tokenizer if english_only_model
    phoneme_tokenizer = None
    if cfg.tokenizer.get("english_only_model", False):
        phoneme_tokenizer = instantiate(
            _get_default_text_tokenizer_conf(phoneme_probability=cfg.tokenizer.phoneme_probability, use_ipa=cfg.tokenizer.use_ipa)
        ).text_tokenizer
    else:
        raise NotImplementedError("Multiple languages tokenization is not fully implemented yet. Will iterate the implementation after figuring out English language.")
        g2p = {"fr": lambda x: x}
        if cfg.tokenizer.get("g2p", None):
            if "english" in cfg.tokenizer["g2p"]:
                english_g2p = instantiate(cfg.tokenizer["g2p"]["english"])
                g2p["en"] = lambda x: english_g2p(x)
            if "spanish" in cfg.tokenizer["g2p"]:
                spanish_g2p = instantiate(cfg.tokenizer["g2p"]["spanish"])
                g2p["es"] = lambda x: spanish_g2p(x)
            if "mandarin" in cfg.tokenizer["g2p"]:
                mandarin_g2p = instantiate(cfg.tokenizer["g2p"]["mandarin"])
                g2p["zh"] = lambda x: mandarin_g2p(x)
            if "german" in cfg.tokenizer["g2p"]:
                german_g2p = instantiate(cfg.tokenizer["g2p"]["german"])
                g2p["de"] = lambda x: german_g2p(x)
    return tokenizer, phoneme_tokenizer