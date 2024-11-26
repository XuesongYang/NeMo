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
from nemo.collections.speechlm_generation.speech_generation_tokenizer import build_tokenizer


def speech_generation_llm_train(cfg: DictConfig):
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # ## retain common settings in data config
    # data_config = deepcopy(cfg['data'])
    # for key in ['train_ds', 'validation_ds', 'test_ds']:
    #     if key in data_config:
    #         data_config.pop(key)

    ############### 1. build static basic tokenizer #########################
    tokenizer, phoneme_tokenizer = build_tokenizer(cfg)

    ############### 2. set up data module #########################
    # TODO @xueyang: there are so many params shared by both dataset and model. During model initialization, it loads
    #   data's config and assign to the same name of variables. So what is the better way to organize the common params?
    #   example params: speech_codebook_size, num_speech_codebooks, speech_offsets
    # TODO @xueyang: 11/25, decoupled tokenizer changes into section 1 from models. Need to check if any further changes done in dataloader.
    #   we are still missing where virtual tokens were added in tokenizer.
    #   1. phoneme_tokenizer is defined if applying english only model. do we have to merge it to tokenizer??
    # Note:
    #   - "taskname" in manifest should match "taskname" in task_templates. We don't need this entry anymore, but it would be great to keep them as a sanity-check.
    #
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

    ############### 2. build model #########################
    model_config = SpeechT5Config(
        task_templates=cfg["task_templates"],

    )
    model = MegatronT5SpeechLMModel(config=model_config, tokenizer=tokenizer)

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