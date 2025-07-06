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

latte = gr.themes.Citrus(
    primary_hue=colors.yellow,              # Gần giống màu "yellow" của Latte
    secondary_hue=colors.rose,              # Gần giống màu "rosewater"
    neutral_hue=colors.gray,
    font=fonts.GoogleFont("Inter"),
    font_mono=fonts.GoogleFont("JetBrains Mono")
).set(
    body_background_fill="#eff1f5",          # Latte base
    body_text_color="#4c4f69",               # Text
    border_color_primary="#bcc0cc",          # Border
    block_background_fill="#ffffff",         # Panel (surface)
    button_primary_background_fill="#df8e1d", # Yellow
    button_primary_text_color="#ffffff",
    link_text_color="#dc8a78",               # Rosewater
)


# --- PHẦN 1 & 2: KHÔNG THAY ĐỔI ---
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

# ... (Logic tải model không đổi) ...
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

# <<< CÁC BIẾN CACHE TOÀN CỤC >>>
last_processed_ref_path = None
last_processed_ref_text = None
ref_audio_processed_old, ref_text_processed_old = None, None # Cache riêng cho model cũ

@spaces.GPU
def process_reference_audio(ref_audio_orig, progress=gr.Progress()):
    """Hàm này được kích hoạt NGAY KHI người dùng tải audio lên."""
    global last_processed_ref_path, last_processed_ref_text
    if not ref_audio_orig: return None, ""
    
    progress(0, desc="Đang xử lý giọng mẫu...")
    gr.Info("Đã nhận audio mẫu. Bắt đầu xử lý...")
    
    clipped_audio_path = ref_audio_orig
    transcribed_text = ""

    if MODEL_TYPE == 'new':
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            sys.path.insert(0, PATH_TO_NEW_F5_REPO)
            _, transcribed_text = tts_instance.preprocess_reference(ref_audio_path=ref_audio_orig, ref_text="", clip_short=True)
            processed_audio_tensor = tts_instance.ref_audio_processed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                clipped_audio_path = tmp_wav.name
            sf.write(clipped_audio_path, processed_audio_tensor.squeeze().cpu().numpy(), tts_instance.target_sample_rate)
            sys.path.pop(0)
    else:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            sys.path.insert(0, PATH_TO_OLD_F5_REPO)
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            _, transcribed_text = preprocess_ref_audio_text(ref_audio_orig, "")
            sys.path.pop(0)

    last_processed_ref_path = clipped_audio_path
    last_processed_ref_text = transcribed_text
    
    progress(1, desc="Xử lý giọng mẫu hoàn tất!")
    gr.Info("Xử lý giọng mẫu hoàn tất! Vui lòng kiểm tra và sửa lại văn bản phiên âm nếu cần.")
    print(f"Xử lý giọng mẫu hoàn tất. Văn bản nhận dạng: '{transcribed_text}'")
    
    return clipped_audio_path, transcribed_text

@spaces.GPU
def infer_tts(ref_audio_ui: str, ref_text_ui: str, gen_text: str, speed: float, cfg_strength: float, nfe_step: int, force_reprocess: bool, progress=gr.Progress()):
    global last_processed_ref_path, last_processed_ref_text, ref_audio_processed_old, ref_text_processed_old
    if not ref_audio_ui: raise gr.Error("Lỗi: Không tìm thấy âm thanh mẫu. Vui lòng tải lên.")
    if not gen_text.strip(): raise gr.Error("Vui lòng nhập nội dung văn bản.")
    
    try:
        ref_text_to_use = ref_text_ui.strip()
        should_reprocess = (
            force_reprocess or
            ref_audio_ui != last_processed_ref_path or
            ref_text_to_use != last_processed_ref_text
        )

        if should_reprocess:
            progress(0, desc="Phát hiện thay đổi hoặc yêu cầu. Đang xử lý lại giọng mẫu...")
            print(f"Đang xử lý lại giọng mẫu với văn bản: '{ref_text_to_use if ref_text_to_use else 'Sẽ chạy ASR'}'")

            if MODEL_TYPE == 'new':
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    sys.path.insert(0, PATH_TO_NEW_F5_REPO)
                    tts_instance.preprocess_reference(ref_audio_path=ref_audio_ui, ref_text=ref_text_to_use, clip_short=True)
                    sys.path.pop(0)
            else: # <<< THAY ĐỔI LỚN: Thêm logic xử lý lại cho model cũ >>>
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    sys.path.insert(0, PATH_TO_OLD_F5_REPO)
                    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
                    # Chạy lại và cập nhật cache của model cũ
                    ref_audio_processed_old, ref_text_processed_old = preprocess_ref_audio_text(ref_audio_ui, ref_text_to_use)
                    sys.path.pop(0)

            last_processed_ref_path = ref_audio_ui
            last_processed_ref_text = ref_text_to_use
            print("Xử lý lại giọng mẫu hoàn tất.")
        else:
            print("Sử dụng giọng mẫu đã được cache. Bỏ qua bước xử lý lại.")

        print(f"Bắt đầu tạo audio cho toàn bộ văn bản...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        if MODEL_TYPE == 'old':
            # <<< THAY ĐỔI LỚN: Dùng kết quả đã cache, không chạy lại preprocess >>>
            sys.path.insert(0, PATH_TO_OLD_F5_REPO)
            from vinorm import TTSnorm
            from f5_tts.infer.utils_infer import infer_process, chunk_text
            sentences = chunk_text(TTSnorm(gen_text).lower())
            audio_chunks = []
            final_sr = 24000
            for i, sentence in enumerate(sentences):
                progress(i / len(sentences), desc=f"Đang tạo chunk {i+1}/{len(sentences)}")
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    wave, sr, _ = infer_process(ref_audio_processed_old, ref_text_processed_old.lower(), sentence, tts_instance["model"], tts_instance["vocoder"], speed=speed, nfe_step=nfe_step)
                audio_chunks.append(wave)
                final_sr = sr
            full_audio = np.concatenate(audio_chunks)
            sf.write(tmp_path, full_audio, final_sr)
            sys.path.pop(0)
        else:
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                sys.path.insert(0, PATH_TO_NEW_F5_REPO)
                from vinorm import TTSnorm
                final_text = TTSnorm(gen_text)
                tts_instance.generate(text=final_text, output_path=tmp_path, nfe_step=nfe_step,
                    cfg_strength=cfg_strength, speed=speed, sway_sampling_coef=-1, cross_fade_duration=0.15)
                sys.path.pop(0)

        progress(0.9, desc="Đang đọc kết quả...")
        final_wave, final_sr = sf.read(tmp_path)
        os.remove(tmp_path)
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_path = os.path.join(output_dir, f"generated_audio_{timestamp}.wav")
        sf.write(final_output_path, final_wave, final_sr)
        print(f"✅ Âm thanh hoàn chỉnh đã được lưu tại: {final_output_path}")
        
        progress(1, desc="Hoàn thành!")
        return (final_sr, final_wave), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Lỗi khi tạo giọng nói: {e}")

# --- Giao diện Gradio (Không thay đổi) ---
with gr.Blocks(theme=latte) as demo:
    # (Toàn bộ phần giao diện giữ nguyên)
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech")
    gr.Markdown(f"### Model: **{DISPLAY_NAME}** | Kiến trúc: **{MODEL_TYPE.upper()}** | Giới hạn: **{WORD_LIMIT} từ**")
    with gr.Row():
        with gr.Column(scale=1):
            ref_audio_ui = gr.Audio(label="1. Tải lên âm thanh mẫu", type="filepath")
            ref_text_ui = gr.Textbox(
                label="2. Sửa lại văn bản (nếu cần)",
                placeholder="Nội dung audio mẫu sẽ tự động xuất hiện ở đây sau khi bạn tải file lên.",
                lines=5, interactive=True)
        with gr.Column(scale=2):
            gen_text_ui = gr.Textbox(label="3. Nhập văn bản cần tạo giọng nói", placeholder="Nhập văn bản dài vào đây...", lines=11)
    with gr.Row():
        speed_slider = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
        cfg_strength_slider = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="🗣️ Độ bám sát giọng mẫu")
    with gr.Accordion("🛠️ Cài đặt nâng cao", open=False):
        nfe_step_slider = gr.Slider(
            minimum=16, maximum=64, value=32, step=2,
            label="🔍 Số bước khử nhiễu (NFE)",
            info="Cao hơn = chậm hơn nhưng có thể chất lượng tốt hơn. Thấp hơn = nhanh hơn.")
        force_reprocess_ui = gr.Checkbox(
            label="Bắt buộc xử lý lại giọng mẫu", value=False,
            info="Tick vào ô này nếu bạn muốn model học lại giọng mẫu từ đầu.")
    btn = gr.Button("🔥 4. Tạo giọng nói", variant="primary")
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram (Chỉ có ở model cũ)")
    ref_audio_ui.upload(
        process_reference_audio,
        inputs=[ref_audio_ui],
        outputs=[ref_audio_ui, ref_text_ui]
    )
    btn.click(
        infer_tts, 
        inputs=[ref_audio_ui, ref_text_ui, gen_text_ui, speed_slider, cfg_strength_slider, nfe_step_slider, force_reprocess_ui], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)