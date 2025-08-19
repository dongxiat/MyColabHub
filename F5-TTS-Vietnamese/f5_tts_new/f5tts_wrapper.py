import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from safetensors.torch import load_file
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from importlib.resources import files
from pydub import AudioSegment, silence

from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import chunk_text, load_vocoder, transcribe, initialize_asr_pipeline

class F5TTSWrapper:
    def __init__(self, model_name: str = "F5TTS_v1_Base", **kwargs):
        self.device = kwargs.get('device') or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sample_rate = kwargs.get('target_sample_rate', 24000)
        self.n_mel_channels = kwargs.get('n_mel_channels', 100)
        self.hop_length = kwargs.get('hop_length', 256)
        self.win_length = kwargs.get('win_length', 1024)
        self.n_fft = kwargs.get('n_fft', 1024)
        self.vocoder_name = kwargs.get('vocoder_name', 'vocos')
        self.ode_method = kwargs.get('ode_method', 'euler')
        
        initialize_asr_pipeline(device=self.device)
        
        ckpt_path = kwargs.get('ckpt_path')
        config_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
        model_cfg = OmegaConf.load(config_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        vocab_file = kwargs.get('vocab_file')
        self.vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
        
        self.model = CFM(
            transformer=model_cls(**model_cfg.model.arch, text_num_embeds=vocab_size, mel_dim=self.n_mel_channels),
            mel_spec_kwargs=dict(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, n_mel_channels=self.n_mel_channels, target_sample_rate=self.target_sample_rate, mel_spec_type=self.vocoder_name),
            odeint_kwargs=dict(method=self.ode_method),
            vocab_char_map=self.vocab_char_map
        ).to(self.device)
        self._load_checkpoint(self.model, ckpt_path, use_ema=kwargs.get('use_ema', True))

        vocoder_path = kwargs.get('vocoder_path') or ("../checkpoints/vocos-mel-24khz" if self.vocoder_name == "vocos" else "../checkpoints/bigvgan_v2_24khz_100band_256x")
        self.vocoder = load_vocoder(vocoder_name=self.vocoder_name, is_local=kwargs.get('use_local_vocoder', False), local_path=vocoder_path, device=self.device, hf_cache_dir=kwargs.get('hf_cache_dir'))

        self.ref_audio_processed = None
        self.ref_text = None
        self.ref_audio_len = None
        self.last_processed_audio_path = None
        
        self.target_rms = 0.1
        self.cross_fade_duration = 0.15
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.speed = 1.0

    def _load_checkpoint(self, model, ckpt_path, use_ema=True):
        dtype = torch.float32 if self.vocoder_name == "bigvgan" else None
        model = model.to(dtype)
        state_dict = load_file(ckpt_path, device=self.device) if ckpt_path.endswith(".safetensors") else torch.load(ckpt_path, map_location=self.device, weights_only=True)
        
        if use_ema:
            state_dict = state_dict.get("ema_model_state_dict", state_dict)
            state_dict = {k.replace("ema_model.", ""): v for k, v in state_dict.items() if k not in ["initted", "step"]}
        else:
            state_dict = state_dict.get("model_state_dict", state_dict)
            
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            state_dict.pop(key, None)
            
        model.load_state_dict(state_dict)
        del state_dict; torch.cuda.empty_cache()

    def _remove_silence_edges(self, audio, silence_threshold=-42):
        start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
        audio = audio[start_idx:]
        end_duration = audio.duration_seconds
        for ms in reversed(audio):
            if ms.dBFS > silence_threshold: break
            end_duration -= 0.001
        return audio[:int(end_duration * 1000)]

    def preprocess_reference(self, ref_audio_path: str, ref_text: str = "", clip_short: bool = True):
        aseg = AudioSegment.from_file(ref_audio_path)
        if clip_short and len(aseg) > 12000:
            aseg = aseg[:12000]
            print("Audio > 12s, đã tự động cắt ngắn.")

        temp_dir = "temp_processed_audio"
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_audio_path = os.path.join(temp_dir, f"ref_{timestamp}.wav")
        final_aseg = self._remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        final_aseg.export(processed_audio_path, format="wav")
        
        if not ref_text.strip():
            ref_text = transcribe(processed_audio_path)
        
        if ref_text and not ref_text.strip().endswith(tuple(".?!。」")):
             ref_text += "."

        audio, sr = torchaudio.load(processed_audio_path)
        if audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True)
        
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms > 0 and rms < self.target_rms:
            audio = audio * self.target_rms / rms
        
        if sr != self.target_sample_rate: 
            audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
        
        self.ref_audio_processed = audio.to(self.device)
        self.ref_text = ref_text.strip()
        self.ref_audio_len = audio.shape[-1] // self.hop_length
        self.last_processed_audio_path = processed_audio_path
        
        return processed_audio_path, self.ref_text

    def generate(self, text: str, **kwargs):
        if self.ref_audio_processed is None: raise ValueError("Reference audio not preprocessed.")
        
        speed = kwargs.get('speed', self.speed)
        target_rms = kwargs.get('target_rms', self.target_rms)
        cross_fade_duration = kwargs.get('cross_fade_duration', self.cross_fade_duration)

        audio_duration_s = self.ref_audio_processed.shape[-1] / self.target_sample_rate
        max_chars = int(len(self.ref_text.encode("utf-8")) / audio_duration_s * (22 - audio_duration_s)) if audio_duration_s > 0 else 50
        
        text_batches = chunk_text(text, max_chars=max_chars)
        generated_waves = []

        for i, text_batch in enumerate(text_batches):
            if kwargs.get('progress_callback') is not None:
                kwargs['progress_callback'](i / len(text_batches), desc=f"Đang tạo phần {i+1}/{len(text_batches)}")
            
            ref_text_len = len(self.ref_text.encode("utf-8"))
            gen_text_len = len(text_batch.encode("utf-8"))
            duration = self.ref_audio_len + int(self.ref_audio_len / ref_text_len * gen_text_len / speed) if ref_text_len > 0 else self.ref_audio_len + int(10 * gen_text_len / speed)

            with torch.inference_mode():
                generated, _ = self.model.sample(cond=self.ref_audio_processed, text=[self.ref_text + " " + text_batch], duration=duration, steps=kwargs.get('nfe_step', self.nfe_step), cfg_strength=kwargs.get('cfg_strength', self.cfg_strength), sway_sampling_coef=-1.0)
                generated = generated.to(torch.float32)[:, self.ref_audio_len:, :].permute(0, 2, 1)
                wave = self.vocoder.decode(generated) if self.vocoder_name == "vocos" else self.vocoder(generated)
                
                rms = torch.sqrt(torch.mean(torch.square(wave)))
                if rms > 0:
                    wave = wave * (target_rms / rms)

                generated_waves.append(wave.squeeze().cpu().numpy())
        
        if len(generated_waves) > 1 and cross_fade_duration > 0:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave, next_wave = final_wave, generated_waves[i]
                fade_samples = int(cross_fade_duration * self.target_sample_rate)
                fade_samples = min(fade_samples, len(prev_wave), len(next_wave))
                if fade_samples > 0:
                    fade_out = np.linspace(1, 0, fade_samples)
                    fade_in = np.linspace(0, 1, fade_samples)
                    cross_faded = prev_wave[-fade_samples:] * fade_out + next_wave[:fade_samples] * fade_in
                    final_wave = np.concatenate([prev_wave[:-fade_samples], cross_faded, next_wave[fade_samples:]])
                else:
                    final_wave = np.concatenate([prev_wave, next_wave])
        else:
            final_wave = np.concatenate(generated_waves)

        if kwargs.get('output_path'):
            torchaudio.save(kwargs['output_path'], torch.tensor(final_wave).unsqueeze(0), self.target_sample_rate)
            return kwargs['output_path']
        return final_wave, self.target_sample_rate