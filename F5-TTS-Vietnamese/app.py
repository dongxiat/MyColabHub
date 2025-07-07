import spaces
import os
import sys
import gradio as gr
import json
import re
import numpy as np
from datetime import datetime
from huggingface_hub import login
from cached_path import cached_path
import tempfile
import soundfile as sf
import contextlib

from gradio.themes.utils import colors, fonts

# --- Theme (Không đổi) ---
latte = gr.themes.Citrus(
    primary_hue=colors.yellow,
    secondary_hue=colors.rose,
    neutral_hue=colors.gray,
    font=fonts.GoogleFont("Inter"),
    font_mono=fonts.GoogleFont("JetBrains Mono")
).set(
    body_background_fill="#eff1f5",
    body_text_color="#4c4f69",
    border_color_primary="#bcc0cc",
    block_background_fill="#ffffff",
    button_primary_background_fill="#df8e1d",
    button_primary_text_color="#ffffff",
    link_text_color="#dc8a78",
)

# --- Phần tải model (Không đổi) ---
MODEL_TYPE = os.getenv('MODEL_TYPE', 'new')
CKPT_PATH = os.getenv('CKPT_HF_PATH')
VOCAB_PATH = os.getenv('VOCAB_HF_PATH')
WORD_LIMIT = int(os.getenv('WORD_LIMIT', 1000))
DISPLAY_NAME = os.getenv('DISPLAY_NAME', 'Unknown Model')
INIT_PARAMS_JSON = os.getenv('INIT_PARAMS_JSON', '{}')
INIT_PARAMS = json.loads(INIT_PARAMS_JSON)

PATH_TO_OLD_F5_REPO = os.path.abspath('f5_tts_old')
PATH_TO_NEW_F5_REPO = os.path.abspath('f5_tts_new')

if not CKPT_PATH or not VOCAB_PATH:
    raise ValueError("Lỗi: CKPT_PATH hoặc VOCAB_PATH chưa được thiết lập.")

tts_instance = None
print(f"Nhận diện loại model: {MODEL_TYPE}")

if MODEL_TYPE == 'old':
    print("Đang tải model theo kiến trúc F5-TTS Base...")
    sys.path.insert(0, PATH_TO_OLD_F5_REPO)
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_vocoder, load_model, chunk_text
    vocoder = load_vocoder()
    model = load_model(
        DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path(CKPT_PATH)), vocab_file=str(cached_path(VOCAB_PATH)),
    )
    tts_instance = {"model": model, "vocoder": vocoder}
    sys.path.pop(0)
    print("✅ Tải model CŨ thành công.")
elif MODEL_TYPE == 'new':
    print(f"Đang tải model với các tham số: {INIT_PARAMS}")
    sys.path.insert(0, PATH_TO_NEW_F5_REPO)
    from f5tts_wrapper import F5TTSWrapper
    INIT_PARAMS['ckpt_path'] = str(cached_path(CKPT_PATH))
    INIT_PARAMS['vocab_file'] = str(cached_path(VOCAB_PATH))
    tts_instance = F5TTSWrapper(**INIT_PARAMS)
    sys.path.pop(0)
    print("✅ Tải model MỚI thành công.")

# Cache riêng cho model cũ
ref_audio_processed_old, ref_text_processed_old = None, None

# Helper để ẩn output không cần thiết
@contextlib.contextmanager
def suppress_outputs(target_path):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.path.insert(0, target_path)
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout, sys.stderr = devnull, devnull
            yield
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        sys.path.pop(0)

@spaces.GPU
def process_reference_audio(ref_audio_orig: str, progress=gr.Progress()):
    if not ref_audio_orig: return None, ""
    
    progress(0, desc="Đang xử lý giọng mẫu...")
    gr.Info("Đã nhận audio mẫu. Bắt đầu xử lý...")
    
    processed_path = ref_audio_orig
    transcribed_text = ""

    if MODEL_TYPE == 'new':
        with suppress_outputs(PATH_TO_NEW_F5_REPO):
            processed_path, transcribed_text = tts_instance.preprocess_reference(ref_audio_path=ref_audio_orig, ref_text="", clip_short=True)
    else:
        with suppress_outputs(PATH_TO_OLD_F5_REPO):
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            global ref_audio_processed_old, ref_text_processed_old
            ref_audio_processed_old, ref_text_processed_old = preprocess_ref_audio_text(ref_audio_orig, "")
            transcribed_text = ref_text_processed_old
            processed_path = ref_audio_orig 

    progress(1, desc="Xử lý giọng mẫu hoàn tất!")
    gr.Info("Xử lý giọng mẫu hoàn tất! Vui lòng kiểm tra và sửa lại văn bản phiên âm nếu cần.")
    print(f"Xử lý giọng mẫu hoàn tất. Văn bản nhận dạng: '{transcribed_text}'")
    
    return processed_path, transcribed_text

@spaces.GPU
def infer_tts(ref_audio_path: str, ref_text_ui: str, gen_text: str, speed: float, cfg_strength: float, nfe_step: int, force_reprocess: bool, progress=gr.Progress()):
    global ref_audio_processed_old, ref_text_processed_old
    if not ref_audio_path: raise gr.Error("Lỗi: Không tìm thấy âm thanh mẫu. Vui lòng tải lên.")
    if not gen_text.strip(): raise gr.Error("Vui lòng nhập nội dung văn bản.")
    
    try:
        ref_text_to_use = ref_text_ui.strip()

        if MODEL_TYPE == 'new':
            text_provided_and_changed = (ref_text_to_use != "" and ref_text_to_use != tts_instance.ref_text)
            should_reprocess_new = (force_reprocess or tts_instance.ref_audio_processed is None or text_provided_and_changed)

            if should_reprocess_new:
                progress(0.1, desc="Đang xử lý lại giọng mẫu...")
                print(f"Đang xử lý lại giọng mẫu với văn bản: '{ref_text_to_use if ref_text_to_use else 'Sẽ chạy ASR'}'")
                with suppress_outputs(PATH_TO_NEW_F5_REPO):
                    tts_instance.preprocess_reference(ref_audio_path=ref_audio_path, ref_text=ref_text_to_use, clip_short=True)
                print("Xử lý lại giọng mẫu hoàn tất.")
            else:
                print("Sử dụng giọng mẫu đã được cache bên trong model. Bỏ qua bước xử lý lại.")
        
        else: # Logic cho model cũ
            should_reprocess_old = (force_reprocess or ref_audio_processed_old is None or (ref_text_to_use != "" and ref_text_to_use != ref_text_processed_old))
            if should_reprocess_old:
                progress(0.1, desc="Đang xử lý lại giọng mẫu...")
                with suppress_outputs(PATH_TO_OLD_F5_REPO):
                    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
                    ref_audio_processed_old, ref_text_processed_old = preprocess_ref_audio_text(ref_audio_path, ref_text_to_use)
                print("Xử lý lại giọng mẫu hoàn tất.")
        
        print(f"Bắt đầu tạo audio cho toàn bộ văn bản...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        if MODEL_TYPE == 'old':
            with suppress_outputs(PATH_TO_OLD_F5_REPO):
                from vinorm import TTSnorm
                from f5_tts.infer.utils_infer import infer_process, chunk_text
                sentences = chunk_text(TTSnorm(gen_text).lower())
                audio_chunks = []
                final_sr = 24000
                for i, sentence in enumerate(sentences):
                    progress(i / len(sentences), desc=f"Đang tạo chunk {i+1}/{len(sentences)}")
                    wave, sr, _ = infer_process(ref_audio_processed_old, ref_text_processed_old.lower(), sentence, tts_instance["model"], tts_instance["vocoder"], speed=speed, nfe_step=nfe_step)
                    audio_chunks.append(wave)
                    final_sr = sr
                full_audio = np.concatenate(audio_chunks)
                sf.write(tmp_path, full_audio, final_sr)
        else: # Model new
            with suppress_outputs(PATH_TO_NEW_F5_REPO):
                from vinorm import TTSnorm
                final_text = TTSnorm(gen_text)
                tts_instance.generate(text=final_text, output_path=tmp_path, nfe_step=nfe_step, cfg_strength=cfg_strength, speed=speed, progress_callback=progress)

        final_wave, final_sr = sf.read(tmp_path)
        os.remove(tmp_path)
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_path = os.path.join(output_dir, f"generated_audio_{timestamp}.wav")
        sf.write(final_output_path, final_wave, final_sr)
        print(f"✅ Âm thanh hoàn chỉnh đã được lưu tại: {final_output_path}")
        
        progress(1, desc="Hoàn thành!")
        
        # <<< SỬA LỖI: Luôn lấy đường dẫn từ "nguồn chân lý" trong wrapper >>>
        processed_ref_path_for_ui = tts_instance.last_processed_audio_path if MODEL_TYPE == 'new' else ref_audio_path

        return (final_sr, final_wave), None, processed_ref_path_for_ui

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Lỗi khi tạo giọng nói: {e}")

# --- Giao diện Gradio ---
with gr.Blocks(theme=latte) as demo:
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech")
    gr.Markdown(f"### Model: **{DISPLAY_NAME}** | Kiến trúc: **{MODEL_TYPE.upper()}** | Giới hạn: **{WORD_LIMIT} từ**")
    with gr.Row():
        with gr.Column(scale=1):
            ref_audio_ui = gr.Audio(label="1. Tải lên âm thanh mẫu", type="filepath")
            ref_text_ui = gr.Textbox(label="2. Sửa lại văn bản (nếu cần)", placeholder="Nội dung audio mẫu sẽ tự động xuất hiện ở đây sau khi bạn tải file lên.", lines=5, interactive=True)
        with gr.Column(scale=2):
            gen_text_ui = gr.Textbox(label="3. Nhập văn bản cần tạo giọng nói", placeholder="Nhập văn bản dài vào đây...", lines=11)
    with gr.Row():
        speed_slider = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
        cfg_strength_slider = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="🗣️ Độ bám sát giọng mẫu")
    with gr.Accordion("🛠️ Cài đặt nâng cao", open=False):
        nfe_step_slider = gr.Slider(minimum=16, maximum=64, value=32, step=2, label="🔍 Số bước khử nhiễu (NFE)", info="Cao hơn = chậm hơn nhưng có thể chất lượng tốt hơn. Thấp hơn = nhanh hơn.")
        force_reprocess_ui = gr.Checkbox(label="Bắt buộc xử lý lại giọng mẫu", value=False, info="Tick vào ô này nếu bạn muốn model học lại giọng mẫu từ đầu.")
    btn = gr.Button("🔥 4. Tạo giọng nói", variant="primary")
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram (Chỉ có ở model cũ)", visible=False)

    ref_audio_ui.upload(process_reference_audio, inputs=[ref_audio_ui], outputs=[ref_audio_ui, ref_text_ui])
    
    btn.click(
        infer_tts, 
        inputs=[ref_audio_ui, ref_text_ui, gen_text_ui, speed_slider, cfg_strength_slider, nfe_step_slider, force_reprocess_ui], 
        outputs=[output_audio, output_spectrogram, ref_audio_ui]
    )

demo.queue().launch(share=True)