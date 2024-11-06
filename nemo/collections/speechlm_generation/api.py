from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging
from copy import deepcopy
import re
from typing import List

from nemo.core.classes.common import Serialization, typecheck
from omegaconf.omegaconf import DictConfig, OmegaConf
from nemo.collections.speechlm_generation.utils import get_object_list_from_config
from nemo.collections.speechlm_generation.data import T5SpeechGenerationDataModule
from nemo.collections.speechlm_generation.models.megatron_t5_speechlm import SpeechT5Config, MegatronT5SpeechLMModel
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceSpeechLLMTTSTokenizer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def load_task_templates(config: DictConfig):
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
    # TODO @xueyang: remove this section and `new_tasks` and `existing_tasks` if no other places applied them. Probably legacy codes.
    if config.new_tasks:
        new_task_name = config.new_tasks[0]
        total_new_task_virtual_tokens = task_templates_dict[new_task_name]["total_virtual_tokens"]

        assert all(
            task_templates_dict[taskname]["total_virtual_tokens"] == total_new_task_virtual_tokens
            for taskname in config.new_tasks
        ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."

    return {"task_templates": task_templates_dict, "task_id_num_to_name": task_id_num_to_name, "max_virtual_tokens": max_virtual_tokens, }


def build_tokenizer(config: DictConfig):
    if hasattr(config, "sentencepiece_legacy"):
        legacy = config.sentencepiece_legacy
    else:
        legacy = True if config.library == 'sentencepiece' else False

    if config.library == "sentencepiece":
        tokenizer = SentencePieceSpeechLLMTTSTokenizer(model_path=config.get('model', None), legacy=legacy)
    else:
        tokenizer = get_nmt_tokenizer(
            library=config.library,
            model_name=config.get("type", None),
            tokenizer_model=config.get('model', None),
            vocab_file=config.get('vocab_file', None),
            merges_file=config.get('merge_file', None),
            use_fast=config.get('use_fast', False),
            delimiter=config.get('delimiter', None),
            special_tokens=config.get('special_tokens', None),
            trust_remote_code=config.get('trust_remote_code', False),
            legacy=legacy,
            chat_template=getattr(config, "chat_template", None),
        )

    if config.get('additional_special_tokens', None) is not None:
        tokens_list = OmegaConf.to_object(config.additional_special_tokens)
        tokenizer.add_special_tokens(tokens_list)

    return tokenizer


def speech_generation_llm_train(cfg: DictConfig):
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 1. build model
    ## retain common settings in data config
    data_config = deepcopy(cfg['data'])
    for key in ['train_ds', 'validation_ds', 'test_ds']:
        if key in data_config:
            data_config.pop(key)

    tokenizer = build_tokenizer(cfg["tokenizer"])
    model_config = SpeechT5Config(
        task_templates=cfg["task_templates"],

    )
    model = MegatronT5SpeechLMModel(config=model_config, tokenizer=tokenizer)

    # 2. set up datasets
    # TODO @xueyang: there are so many params shared by both dataset and model. During model initialization, it loads
    #   data's config and assign to the same name of variables. So what is the better way to organize the common params?
    #   example params: speech_codebook_size, num_speech_codebooks, speech_offsets
    data = T5SpeechGenerationDataModule(
        config=cfg["data"],
        tokenizer=tokenizer,
        virtual_prompt_source=model_config.virtual_prompt_source,
        task_templates=model_config.task_templates,
        pseudo_tokens=model_config.pseudo_tokens,
        pad_token_id=model_config.pad_token_id,
        lm_vocab_size=model_config.lm_vocab_size,
        seq_pattern=model_config.seq_pattern,
        english_only_model=model_config.english_only_model,
        context_conditioning=model_config.context_conditioning,
        use_beta_binomial_interpolator=model_config.use_beta_binomial_interpolator,
    )

    # 3. update model's phoneme_tokenizer aligning with dataset's.
    if model_config.phoneme_tokenizer is None:
        model_config.phoneme_tokenizer = data._train_ds.phoneme_tokenizer

    # 3. initialize strategy

    # 4. set up the optimizer
    optim = Serialization.from_config_dict(cfg['optim'])

    # 5. set up the trainer
    trainer = nl.Trainer(
        plugins=get_object_list_from_config(cfg["plugins"]),
        strategy=Serialization.from_config_dict(cfg["strategy"]),
        callbacks=get_object_list_from_config(cfg["callbacks"]),
        **cfg["trainer"],
    )

    # 6. set up the logger and auto-resume
    resume = Serialization.from_config_dict(cfg['resume'])
    logger = Serialization.from_config_dict(cfg['logger'])

    # 7. train the model
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        resume=resume,
        optim=optim,
        tokenizer=data.tokenizer,
        model_transform=,
    )

def speech_generation_llm_finetune(cfg: DictConfig):
    raise NotImplementedError("Finetune function is not implemented yet.")

def speech_generation_llm_inference(cfg: DictConfig):
    raise NotImplementedError("Inference function is not implemented yet.")