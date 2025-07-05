import spaces
import os
import sys
import gradio as gr
from huggingface_hub import login
from cached_path import cached_path
import tempfile
import soundfile as sf # Cần để đọc file wav được tạo ra

# --- PHẦN 1: ĐỌC CẤU HÌNH TỪ BIẾN MÔI TRƯỜNG ---
MODEL_TYPE = os.getenv('MODEL_TYPE', 'new')
CKPT_PATH = os.getenv('CKPT_HF_PATH')
VOCAB_PATH = os.getenv('VOCAB_HF_PATH')
WORD_LIMIT = int(os.getenv('WORD_LIMIT', 1000))

# Đường dẫn đến các thư mục thư viện
PATH_TO_OLD_F5_REPO = os.path.abspath('f5_tts_old')
# Đường dẫn repo mới phải trỏ đến thư mục cha của 'f5_tts' và 'f5tts_wrapper.py'
PATH_TO_NEW_F5_REPO = os.path.abspath('f5_tts_new')

if not CKPT_PATH or not VOCAB_PATH:
    raise ValueError("Lỗi: CKPT_PATH hoặc VOCAB_PATH chưa được thiết lập.")

# --- PHẦN 2: TẢI MODEL & CACHING LOGIC ---
tts_instance = None
last_ref_audio_path = None # Biến cache để lưu đường dẫn file mẫu cuối cùng

print(f"Nhận diện loại model: {MODEL_TYPE}")

if MODEL_TYPE == 'old':
    print("Đang tải model theo kiến trúc F5-TTS Base...")
    sys.path.insert(0, PATH_TO_OLD_F5_REPO)
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_vocoder, load_model
    
    vocoder = load_vocoder()
    model = load_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path(CKPT_PATH)),
        vocab_file=str(cached_path(VOCAB_PATH)),
    )
    tts_instance = {"model": model, "vocoder": vocoder}
    sys.path.pop(0)
    print("✅ Tải model CŨ thành công.")

elif MODEL_TYPE == 'new':
    print("Đang tải model theo kiến trúc F5-TTS v1...")
    sys.path.insert(0, PATH_TO_NEW_F5_REPO)
    from f5tts_wrapper import F5TTSWrapper

    tts_instance = F5TTSWrapper(
        vocoder_name="vocos",
        ckpt_path=str(cached_path(CKPT_PATH)),
        vocab_file=str(cached_path(VOCAB_PATH)),
        use_ema=False,
    )
    sys.path.pop(0)
    print("✅ Tải model MỚI thành công.")

def post_process(text):
    text = " " + text.replace('"', "") + " "
    text = text.replace(" . . ", " . ").replace(" .. ", " . ")
    text = text.replace(" , , ", " , ").replace(" ,, ", " , ")
    return " ".join(text.split())

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0):
    global last_ref_audio_path # Sử dụng biến cache toàn cục
    if not ref_audio_orig:
        raise gr.Error("Vui lòng tải lên một tệp âm thanh mẫu.")
    if not gen_text.strip():
        raise gr.Error("Vui lòng nhập nội dung văn bản.")
    if len(gen_text.strip().split()) > WORD_LIMIT:
        raise gr.Error(f"Văn bản quá dài. Giới hạn là {WORD_LIMIT} từ.")
    
    try:
        if MODEL_TYPE == 'old':
            # Logic cho model cũ không thay đổi
            sys.path.insert(0, PATH_TO_OLD_F5_REPO)
            from vinorm import TTSnorm
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process, save_spectrogram

            final_text = post_process(TTSnorm(gen_text)).lower()
            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
            
            final_wave, final_sr, spectrogram = infer_process(
                ref_audio, ref_text.lower(), final_text, tts_instance["model"], tts_instance["vocoder"], speed=speed
            )
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                save_spectrogram(spectrogram, tmp_file.name)
                spectrogram_path = tmp_file.name
            
            sys.path.pop(0)
            return (final_sr, final_wave), spectrogram_path

        elif MODEL_TYPE == 'new':
            sys.path.insert(0, PATH_TO_NEW_F5_REPO)
            from vinorm import TTSnorm
            
            # --- LOGIC CACHING CHO WRAPPER ---
            # Kiểm tra xem file mẫu có phải là file mới không
            if ref_audio_orig != last_ref_audio_path:
                print(f"Phát hiện âm thanh mẫu mới. Đang xử lý: {ref_audio_orig}")
                # Nếu mới, chạy preprocess_reference
                # Wrapper này không cần văn bản tham chiếu, nó có thể tự dùng Whisper
                tts_instance.preprocess_reference(
                    ref_audio_path=ref_audio_orig,
                    ref_text="", # Để trống để wrapper tự dùng ASR
                    clip_short=True
                )
                # Cập nhật cache
                last_ref_audio_path = ref_audio_orig
                print("Xử lý giọng mẫu thành công.")
            else:
                print("Sử dụng giọng mẫu đã được xử lý trước đó.")

            final_text = post_process(TTSnorm(gen_text)).lower()
            
            # Wrapper lưu file trực tiếp, nên ta cần tạo file tạm
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                output_path = tmp_wav.name

            # Chạy generate, nó sẽ lưu kết quả vào output_path
            tts_instance.generate(
                text=final_text,
                output_path=output_path,
                speed=speed
            )
            
            # Đọc lại file wav vừa tạo để trả về cho Gradio
            final_sr, final_wave = sf.read(output_path)
            
            sys.path.pop(0)
            # Wrapper không trả về spectrogram, nên ta trả về None
            return (final_sr, final_wave), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Lỗi khi tạo giọng nói: {e}")

# --- Giao diện Gradio không đổi ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech")
    display_name = os.getenv('CKPT_HF_PATH', 'N/A').split('/')[1]
    gr.Markdown(f"### Model: **{display_name}** | Kiến trúc: **{MODEL_TYPE.upper()}** | Giới hạn: **{WORD_LIMIT} từ**")

    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Âm thanh mẫu", type="filepath")
        gen_text = gr.Textbox(label="📝 Văn bản", placeholder="Nhập văn bản...", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
    btn = gr.Button("🔥 Tạo giọng nói", variant="primary")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram (Chỉ có ở model cũ)")

    btn.click(infer_tts, inputs=[ref_audio, gen_text, speed], outputs=[output_audio, output_spectrogram])

demo.queue().launch(share=True)