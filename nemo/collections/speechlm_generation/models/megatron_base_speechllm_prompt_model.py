import pytorch_lightning as pl
from nemo.lightning import io
from nemo.collections.llm import fn
from nemo.utils.app_state import AppState
import uuid
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Optional

"""
copy and paste and modify codes:
nemo.core.classes.modelPT.ModelPT 
--> nemo.collections.nlp.models.nlp_model.NLPModel 
--> nemo.collections.nlp.models.language_modeling.megatron_base_model.MegatronBaseModel
--> nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model.MegatronBaseSpeechLM
--> nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model.MegatronT5SpeechLMModel
"""

class MegatronBaseSpeechLM(pl.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(self):
        super().__init__()

        # set global vars in AppState
        app_state = AppState()

        self._set_model_guid()
        self._cfg = OmegaConf.create({})
        # Set device_id in AppState
        if torch.cuda.is_available() and torch.cuda.current_device() is not None:
            app_state.device_id = torch.cuda.current_device()

        # Create list of lists for val and test outputs to support multiple dataloaders
        # Initialize an empty list as sometimes self._validation_dl can be None at this stage
        self._validation_step_outputs = None

        # Initialize an empty list as sometimes self._test_dl can be None at this stage
        self._test_step_outputs = None

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    @cfg.setter
    def cfg(self, cfg: DictConfig):
        self._cfg = cfg

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit, validate, test, or predict.
        This is called on every process when using DDP.

        Args:
            stage: fit, validate, test or predict
        """
        self.propagate_model_guid()
        if stage == 'fit':
            train_deferred_setup = (
                'train_ds' in self._cfg
                and self._cfg.train_ds is not None
                and self._cfg.train_ds.get('defer_setup', False)
            )
            if self.train_dataloader() is None and train_deferred_setup:
                self.setup_training_data(self._cfg.train_ds)

        if stage in ('fit', 'validate'):
            val_deferred_setup = (
                'validation_ds' in self._cfg
                and self._cfg.validation_ds is not None
                and self._cfg.validation_ds.get('defer_setup', False)
            )
            if self.val_dataloader() is None and val_deferred_setup:
                self.setup_multiple_validation_data(val_data_config=self._cfg.validation_ds)

        if stage == 'test':
            test_deferred_setup = (
                'test_ds' in self._cfg
                and self._cfg.test_ds is not None
                and self._cfg.test_ds.get('defer_setup', False)
            )
            if self.test_dataloader() is None and test_deferred_setup:
                self.setup_multiple_test_data(test_data_config=self._cfg.test_ds)

    def _set_model_guid(self):
        if not hasattr(self, 'model_guid'):
            appstate = AppState()

            # Generate a unique uuid for the instance
            # also determine if the model is being restored or not, and preserve the path
            self.model_guid = str(uuid.uuid4())
            appstate.register_model_guid(self.model_guid)

    def propagate_model_guid(self):
        """
        Propagates the model GUID to all submodules, recursively.
        """

        def recursively_propagate_guid(module: "NeuralModule"):
            module.model_guid = self.model_guid
            for child in module.children():
                recursively_propagate_guid(child)

        for _, _, module in self.named_nemo_modules():
            module.model_guid = self.model_guid
            recursively_propagate_guid(module)