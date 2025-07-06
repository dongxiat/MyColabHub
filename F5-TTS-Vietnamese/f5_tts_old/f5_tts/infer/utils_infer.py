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
import contextlib # Cần để tắt tiếng log thư viện

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
last_ref_audio_path = None

print(f"Nhận diện loại model: {MODEL_TYPE}")

# ... (Logic tải model không đổi) ...
if MODEL_TYPE == 'old':
    print("Đang tải model theo kiến trúc F5-TTS Base...")
    sys.path.insert(0, PATH_TO_OLD_F5_REPO)
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_vocoder, load_model
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

def intelligent_chunker(text: str, min_words=3, max_words=25):
    # (Hàm này không đổi)
    sentences = re.split(r'([.!?…])', text)
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') for i in range(0, len(sentences), 2)]
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return []
    final_chunks, processed_chunks = [], []
    for s in sentences:
        if len(s.split()) < min_words and final_chunks: final_chunks[-1] += " " + s
        else: final_chunks.append(s)
    for chunk in final_chunks:
        if len(chunk.split()) > max_words:
            parts = [p.strip() for p in chunk.split(',')]
            buffer = ""
            for part in parts:
                if not buffer: buffer = part
                elif len(buffer.split()) + len(part.split()) <= max_words: buffer += ", " + part
                else:
                    processed_chunks.append(buffer)
                    buffer = part
            if buffer: processed_chunks.append(buffer)
        else: processed_chunks.append(chunk)
    return [c for c in processed_chunks if c]

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float, cfg_strength: float, nfe_step: int, progress=gr.Progress()):
    global last_ref_audio_path
    if not ref_audio_orig: raise gr.Error("Vui lòng tải lên một tệp âm thanh mẫu.")
    if not gen_text.strip(): raise gr.Error("Vui lòng nhập nội dung văn bản.")
    if len(gen_text.strip().split()) > WORD_LIMIT: raise gr.Error(f"Văn bản quá dài. Giới hạn là {WORD_LIMIT} từ.")
    
    try:
        # <<< THAY ĐỔI 1: Xử lý giọng mẫu TRƯỚC vòng lặp >>>
        ref_audio_processed, ref_text_processed = None, None
        if ref_audio_orig != last_ref_audio_path:
            progress(0, desc="Phát hiện giọng mẫu mới. Đang xử lý...")
            if MODEL_TYPE == 'new':
                sys.path.insert(0, PATH_TO_NEW_F5_REPO)
                tts_instance.preprocess_reference(ref_audio_path=ref_audio_orig, ref_text="", clip_short=True)
                sys.path.pop(0)
                print(f"Xử lý giọng mẫu (mới) thành công.")
            else: # MODEL_TYPE == 'old'
                sys.path.insert(0, PATH_TO_OLD_F5_REPO)
                from f5_tts.infer.utils_infer import preprocess_ref_audio_text
                # Tắt tiếng một lần để không hiện log xử lý ref audio
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio_orig, "")
                sys.path.pop(0)
                # In ra log sạch theo ý bạn
                print(f"Xử lý giọng mẫu (cũ) thành công.")
                print(f"ref_text: {ref_text_processed}")

            last_ref_audio_path = ref_audio_orig

        print("Đang tách văn bản thành các chunk thông minh...")
        sentences = intelligent_chunker(gen_text)
        if not sentences: raise gr.Error("Không tìm thấy câu hợp lệ nào trong văn bản.")
        print(f"Đã tách thành {len(sentences)} chunk. Bắt đầu tạo audio...")

        audio_chunks = []
        final_sr = 24000
        
        for i, sentence in enumerate(sentences):
            progress(i / len(sentences), desc=f"Đang tạo chunk {i+1}/{len(sentences)}")
            # In ra log theo format bạn muốn
            print(f"gen_text {i}: {sentence[:80]}...")
            
            if MODEL_TYPE == 'old':
                # Tắt tiếng hàm infer_process để log được sạch
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    sys.path.insert(0, PATH_TO_OLD_F5_REPO)
                    from vinorm import TTSnorm
                    from f5_tts.infer.utils_infer import infer_process
                    final_text = TTSnorm(sentence).lower()
                    # Sử dụng lại ref audio đã xử lý, không cần gọi lại preprocess_ref_audio_text
                    wave, sr, _ = infer_process(ref_audio_processed, ref_text_processed.lower(), final_text, tts_instance["model"], tts_instance["vocoder"], speed=speed, nfe_step=nfe_step)
                    sys.path.pop(0)
            else: 
                sys.path.insert(0, PATH_TO_NEW_F5_REPO)
                from vinorm import TTSnorm
                final_text = TTSnorm(sentence)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    tmp_path = tmp_wav.name
                tts_instance.generate(
                    text=final_text, output_path=tmp_path, nfe_step=nfe_step,
                    cfg_strength=cfg_strength, speed=speed, sway_sampling_coef=-1, cross_fade_duration=0.15
                )
                wave, sr = sf.read(tmp_path)
                os.remove(tmp_path)
                sys.path.pop(0)

            audio_chunks.append(wave)
            final_sr = sr 

        print("Đang ghép các file audio...")
        progress(1, desc="Đang ghép nối và hoàn tất...")
        full_audio = np.concatenate(audio_chunks)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_path = os.path.join(output_dir, f"generated_audio_{timestamp}.wav")
        sf.write(final_output_path, full_audio, final_sr)
        print(f"✅ Âm thanh hoàn chỉnh đã được lưu tại: {final_output_path}")

        yield (final_sr, full_audio), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Lỗi khi tạo giọng nói: {e}")

# --- Giao diện Gradio (Không thay đổi) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech")
    gr.Markdown(f"### Model: **{DISPLAY_NAME}** | Kiến trúc: **{MODEL_TYPE.upper()}** | Giới hạn: **{WORD_LIMIT} từ**")
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Âm thanh mẫu", type="filepath")
        gen_text = gr.Textbox(label="📝 Văn bản", placeholder="Nhập văn bản dài vào đây...", lines=5)
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
        cfg_strength_slider = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="🗣️ Độ bám sát giọng mẫu")
    with gr.Accordion("🛠️ Cài đặt nâng cao", open=False):
        nfe_step_slider = gr.Slider(
            minimum=16, maximum=64, value=32, step=2,
            label="🔍 Số bước khử nhiễu (NFE)",
            info="Cao hơn = chậm hơn nhưng có thể chất lượng tốt hơn. Thấp hơn = nhanh hơn."
        )
    btn = gr.Button("🔥 Tạo giọng nói", variant="primary")
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram (Chỉ có ở model cũ)")
    btn.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, cfg_strength_slider, nfe_step_slider], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)