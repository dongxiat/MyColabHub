import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Callable
import tempfile
import types
from datetime import datetime

from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from importlib.resources import files
from pydub import AudioSegment, silence

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)
from f5_tts.infer.utils_infer import (
    chunk_text,
    load_vocoder,
    transcribe,
    initialize_asr_pipeline,
)
from viphoneme import vi2IPA

class F5TTSWrapper:
    """
    A wrapper class for F5-TTS that preprocesses reference audio once 
    and allows for repeated TTS generation.
    """

    def __init__(
        self, 
        model_name: str = "F5TTS_v1_Base", 
        ckpt_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        vocoder_name: str = "vocos",
        use_local_vocoder: bool = False,
        vocoder_path: Optional[str] = None,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        target_sample_rate: int = 24000,
        n_mel_channels: int = 100,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        ode_method: str = "euler",
        use_ema: bool = True,
        dur_predictor_ckpt: Optional[str] = None,
        use_phoneme_durations: bool = True,
    ):
        if device is None:
            self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.target_sample_rate = target_sample_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.mel_spec_type = vocoder_name
        self.ode_method = ode_method
        initialize_asr_pipeline(device=self.device)
        if ckpt_path is None:
            ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
        config_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
        model_cfg = OmegaConf.load(config_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        if vocab_file is None:
            vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
        self.vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
        self.model = CFM(
            transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
            mel_spec_kwargs=dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate, mel_spec_type=vocoder_name),
            odeint_kwargs=dict(method=ode_method),
            vocab_char_map=self.vocab_char_map
        ).to(self.device)
        self._load_checkpoint(self.model, ckpt_path, dtype=torch.float32 if vocoder_name == "bigvgan" else None, use_ema=use_ema)
        if vocoder_path is None:
            vocoder_path = "../checkpoints/vocos-mel-24khz" if vocoder_name == "vocos" else "../checkpoints/bigvgan_v2_24khz_100band_256x"
        self.vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=use_local_vocoder, local_path=vocoder_path, device=self.device, hf_cache_dir=hf_cache_dir)
        self.ref_audio_processed = None
        self.ref_text = None
        self.ref_audio_len = None
        self.target_rms = 0.1
        self.cross_fade_duration = 0.15
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        self.fix_duration = None
        self.use_phoneme_durations = use_phoneme_durations
        self.phoneme_map = {}
        self.dur_predictor_loaded = False
        if dur_predictor_ckpt is not None and hasattr(self.model, 'duration_predictor'):
            try:
                duration_checkpoint = torch.load(dur_predictor_ckpt, map_location=self.device)
                if 'duration_predictor' in duration_checkpoint:
                    self.model.duration_predictor.load_state_dict(duration_checkpoint['duration_predictor'])
                    print(f"Loaded duration predictor from {dur_predictor_ckpt}")
                    self.dur_predictor_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load duration predictor checkpoint: {e}")
        # Thêm biến trạng thái để lưu đường dẫn file đã xử lý
        self.last_processed_audio_path = None

    def preprocess_reference(self, ref_audio_path: str, ref_text: str = "", clip_short: bool = True):
        print("Converting and processing reference audio...")
        aseg = AudioSegment.from_file(ref_audio_path)
        original_duration = len(aseg)
        was_clipped = False

        if clip_short and original_duration > 12000:
            was_clipped = True
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000: break
                non_silent_wave += non_silent_seg
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10)
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000: break
                    non_silent_wave += non_silent_seg
            if len(non_silent_wave) > 12000: non_silent_wave = non_silent_wave[:12000]
            aseg = non_silent_wave

        temp_dir = "temp_processed_audio"
        os.makedirs(temp_dir, exist_ok=True)
        
        if was_clipped or not ref_audio_path.startswith(temp_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_audio_path = os.path.join(temp_dir, f"ref_{timestamp}.wav")
            final_aseg = self._remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
            final_aseg.export(processed_audio_path, format="wav")
            print(f"Saved processed audio to: {processed_audio_path}")
        else:
            processed_audio_path = ref_audio_path
            print(f"Audio is short and already processed, using existing path: {processed_audio_path}")

        if not ref_text.strip():
            print("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(processed_audio_path)
        else:
            print("Using custom reference text...")
        
        if not ref_text.endswith((". ", "。")):
            ref_text += ". " if ref_text.endswith(".") else ". "
        
        print("\nReference text:", ref_text)
        
        audio, sr = torchaudio.load(processed_audio_path)
        if audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms: audio = audio * self.target_rms / rms
        if sr != self.target_sample_rate: audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
        
        self.ref_audio_processed = audio.to(self.device)
        self.ref_text = ref_text.strip()
        self.ref_audio_len = audio.shape[-1] // self.hop_length
        self.last_processed_audio_path = processed_audio_path
        
        return processed_audio_path, ref_text

    def _load_checkpoint(self, model, ckpt_path, dtype=None, use_ema=True):
        if dtype is None:
            dtype = (torch.float16 if "cuda" in self.device and torch.cuda.get_device_properties(self.device).major >= 7 and not torch.cuda.get_device_name().endswith("[ZLUDA]") else torch.float32)
        model = model.to(dtype)
        ckpt_type = ckpt_path.split(".")[-1]
        if ckpt_type == "safetensors":
            from safetensors.torch import load_file
            checkpoint = load_file(ckpt_path, device=self.device)
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if use_ema:
            if ckpt_type == "safetensors": checkpoint = {"ema_model_state_dict": checkpoint}
            checkpoint["model_state_dict"] = {k.replace("ema_model.", ""): v for k, v in checkpoint["ema_model_state_dict"].items() if k not in ["initted", "step"]}
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]: del checkpoint["model_state_dict"][key]
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            if ckpt_type == "safetensors": checkpoint = {"model_state_dict": checkpoint}
            model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint; torch.cuda.empty_cache(); return model.to(self.device)

    def _remove_silence_edges(self, audio, silence_threshold=-42):
        start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
        audio = audio[start_idx:]
        end_duration = audio.duration_seconds
        for ms in reversed(audio):
            if ms.dBFS > silence_threshold: break
            end_duration -= 0.001
        return audio[: int(end_duration * 1000)]

    def generate(self, text: str, output_path: Optional[str] = None, nfe_step: Optional[int] = None, cfg_strength: Optional[float] = None, sway_sampling_coef: Optional[float] = None, speed: Optional[float] = None, fix_duration: Optional[float] = None, cross_fade_duration: Optional[float] = None, use_duration_predictor: Optional[bool] = None, return_numpy: bool = False, return_spectrogram: bool = False, progress_callback: Optional[Callable] = None):
        if self.ref_audio_processed is None: raise ValueError("Reference audio not preprocessed.")
        nfe_step = nfe_step or self.nfe_step
        cfg_strength = cfg_strength or self.cfg_strength
        sway_sampling_coef = sway_sampling_coef or self.sway_sampling_coef
        speed = speed or self.speed
        fix_duration = fix_duration or self.fix_duration
        cross_fade_duration = cross_fade_duration or self.cross_fade_duration
        can_use_predictor = (use_duration_predictor if use_duration_predictor is not None else self.use_phoneme_durations) and hasattr(self.model, 'duration_predictor') and self.dur_predictor_loaded
        audio_len = self.ref_audio_processed.shape[-1] / self.target_sample_rate
        max_chars = int(len(self.ref_text.encode("utf-8")) / audio_len * (22 - audio_len)) if audio_len > 0 else 50
        text_batches = chunk_text(text, max_chars=max_chars)
        generated_waves = []
        for i, text_batch in enumerate(text_batches):
            if progress_callback is not None:
                try: progress_callback(i / len(text_batches), desc=f"Đang tạo phần {i+1}/{len(text_batches)}")
                except: pass
            local_speed = speed if len(text_batch.encode("utf-8")) >= 10 else 0.3
            text_list = [self.ref_text + text_batch]
            final_text_list = convert_char_to_pinyin(text_list)
            duration = None
            if fix_duration is not None: duration = int(fix_duration * self.target_sample_rate / self.hop_length)
            elif can_use_predictor:
                try:
                    phoneme_seq = vi2IPA(text_batch)
                    phoneme_indices = self._phoneme_to_indices(phoneme_seq)
                    phoneme_tensor = torch.tensor([phoneme_indices], dtype=torch.long, device=self.device)
                    phoneme_mask = torch.ones_like(phoneme_tensor).int()
                    with torch.no_grad():
                        log_durations = self.model.duration_predictor(phoneme_tensor, phoneme_mask)
                        durations = torch.exp(log_durations).round().int()
                        total_duration = durations.sum().item()
                    duration = self.ref_audio_len + int(total_duration / local_speed)
                except Exception as e: print(f"Error in phoneme duration prediction: {e}")
            if duration is None:
                ref_text_len = len(self.ref_text.encode("utf-8"))
                gen_text_len = len(text_batch.encode("utf-8"))
                if ref_text_len > 0: duration = self.ref_audio_len + int(self.ref_audio_len / ref_text_len * gen_text_len / local_speed)
                else: duration = self.ref_audio_len + int(10 * gen_text_len / local_speed)
            with torch.inference_mode():
                generated, _ = self.model.sample(cond=self.ref_audio_processed, text=final_text_list, duration=duration, steps=nfe_step, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef)
                generated = generated.to(torch.float32)[:, self.ref_audio_len:, :].permute(0, 2, 1)
                generated_wave = self.vocoder.decode(generated) if self.mel_spec_type == "vocos" else self.vocoder(generated)
                rms = torch.sqrt(torch.mean(torch.square(self.ref_audio_processed)))
                if rms > 0 and rms < self.target_rms: generated_wave = generated_wave * rms / self.target_rms
                generated_waves.append(generated_wave.squeeze().cpu().numpy())
        final_wave = np.concatenate(generated_waves) if len(generated_waves) == 1 or cross_fade_duration <= 0 else self._crossfade_waves(generated_waves, cross_fade_duration)
        if output_path is not None: torchaudio.save(output_path, torch.tensor(final_wave).unsqueeze(0), self.target_sample_rate)
        if return_numpy: return final_wave, self.target_sample_rate
        return output_path

    def _crossfade_waves(self, waves, duration):
        final_wave = waves[0]
        for i in range(1, len(waves)):
            prev_wave, next_wave = final_wave, waves[i]
            fade_samples = int(duration * self.target_sample_rate)
            fade_samples = min(fade_samples, len(prev_wave), len(next_wave))
            if fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)
            cross_faded = prev_wave[-fade_samples:] * fade_out + next_wave[:fade_samples] * fade_in
            final_wave = np.concatenate([prev_wave[:-fade_samples], cross_faded, next_wave[fade_samples:]])
        return final_wave

    def _phoneme_to_indices(self, phoneme_seq):
        for p in phoneme_seq:
            if p not in self.phoneme_map: self.phoneme_map[p] = len(self.phoneme_map) + 1
        return [self.phoneme_map.get(p, 0) for p in phoneme_seq]