import os

import numpy as np
import soundfile as sf
import torch
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.tts.data.text_to_speech_dataset import DatasetSample, T5TTSDataset
from nemo.collections.tts.models import T5TTS_Model

############  Checkpoints Paths #################################
# Checkpoint and Hparams Paths
ckpt_root = "/home/xueyang/pretrained_model/t5_tts/ckpts/icml2025_base_checkpoints"
hparams_file = f"{ckpt_root}/decodercontext_small_sp_ks3CorrectWithPrior_onlyphoneme_hparams.yaml"
checkpoint_file = f"{ckpt_root}/decodercontext_small_sp_ks3CorrectWithPrior_onlyphoneme_epoch161.ckpt"
codecmodel_path = "/home/xueyang/pretrained_model/t5_tts/pretrained_models/AudioCodec_21Hz_no_eliz.nemo"

# Temp out dir for saving audios
out_dir = "/home/xueyang/workspace/experiments/t5_tts/icml25/21hz/inference_finalizedtransformer"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


############  Load Models #################################
model_cfg = OmegaConf.load(hparams_file).cfg

with open_dict(model_cfg):
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None


model = T5TTS_Model(cfg=model_cfg)
print("Loading weights from checkpoint")
ckpt = torch.load(checkpoint_file)
model.load_state_dict(ckpt['state_dict'])
print("Loaded weights.")

model.use_kv_cache_for_inference = True

model.cuda()
model.eval()


#################### Initialize Dataset class and helper functions  ###############################
test_dataset = T5TTSDataset(
    dataset_meta={},
    sample_rate=model_cfg.sample_rate,
    min_duration=0.5,
    max_duration=20,
    codec_model_downsample_factor=model_cfg.codec_model_downsample_factor,
    bos_id=model.bos_id,
    eos_id=model.eos_id,
    context_audio_bos_id=model.context_audio_bos_id,
    context_audio_eos_id=model.context_audio_eos_id,
    audio_bos_id=model.audio_bos_id,
    audio_eos_id=model.audio_eos_id,
    num_audio_codebooks=model_cfg.num_audio_codebooks,
    prior_scaling_factor=None,
    load_cached_codes_if_available=True,
    dataset_type='test',
    tokenizer_config=None,
    load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
    use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
    pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
    context_duration_min=model.cfg.get('context_duration_min', 5.0),
    context_duration_max=model.cfg.get('context_duration_max', 5.0),
)
test_dataset.text_tokenizer, test_dataset.text_conditioning_tokenizer = model._setup_tokenizers(model.cfg, mode='test')


def get_audio_duration(file_path):
    with sf.SoundFile(file_path) as audio_file:
        # Calculate the duration
        duration = len(audio_file) / audio_file.samplerate
        return duration


def create_record(text, context_audio_filepath=None, context_text=None):
    dummy_audio_fp = os.path.join(out_dir, "dummy_audio.wav")
    _ = sf.write(dummy_audio_fp, np.zeros(22050 * 3), 22050)  # 3 seconds of silence
    record = {
        'audio_filepath': dummy_audio_fp,
        'duration': 3.0,
        'text': text,
        'speaker': "dummy",
    }
    if context_text is not None:
        assert context_audio_filepath is None
        record['context_text'] = context_text
    else:
        assert context_audio_filepath is not None
        record['context_audio_filepath'] = context_audio_filepath
        record['context_audio_duration'] = get_audio_duration(context_audio_filepath)

    return record


############################ Set transcript and context pairs to test ###################
# Change sample text and prompt audio/text here
audio_base_dir = "/"
test_entries = [
    create_record(
        text="This is a sample sentence to test the speed of my text to speech synthesis model. Call nine eight nine, two four seven, nine nine eight three.",
        # context_audio_filepath="/home/xueyang/datasets/RIVA-TTS/en/Lindy/44khz/WIZWIKI/LINDY_WIZWIKI_001254.wav", # Supply either context_audio_filepath or context_text, not both
        context_audio_filepath="/home/xueyang/datasets/zh-tw/241209/mount/src/NeMo/ASR/TechOrange_tp1/241209/preprocessed/trial/processed_audio_segment_202_473.wav",
        # context_text="Speaker and Emotion: | Language:en Dataset:Riva Speaker:Lindy_WIZWIKI |",
    ),
]

data_samples = []
for entry in test_entries:
    dataset_sample = DatasetSample(
        dataset_name="sample",
        manifest_entry=entry,
        audio_dir=audio_base_dir,
        feature_dir=audio_base_dir,
        text=entry['text'],
        speaker=None,
        speaker_index=0,
    )
    data_samples.append(dataset_sample)

test_dataset.data_samples = data_samples

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, num_workers=0, shuffle=False
)


#####################################  Generate  #####################################
item_idx = 0
for bidx, batch in enumerate(test_data_loader):
    print("Processing batch {} out of {}".format(bidx, len(test_data_loader)))
    model.t5_decoder.reset_cache(use_cache=True)
    batch_cuda = {}
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch_cuda[key] = batch[key].cuda()
        else:
            batch_cuda[key] = batch[key]
    import time

    st = time.time()
    predicted_audio, predicted_audio_lens, _, _ = model.infer_batch(
        batch_cuda, max_decoder_steps=500, temperature=0.6, topk=80, use_cfg=True, cfg_scale=2.5
    )
    print("generation time", time.time() - st)
    for idx in range(predicted_audio.size(0)):
        predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
        predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
        audio_path = os.path.join(out_dir, f"predicted_audio_{item_idx}.wav")
        sf.write(audio_path, predicted_audio_np, model.cfg.sample_rate)
        item_idx += 1
