import spaces
import os
import sys
import gradio as gr
import json
from huggingface_hub import login
from cached_path import cached_path
import tempfile
import soundfile as sf

# --- PHẦN 1: ĐỌC CẤU HÌNH TỪ BIẾN MÔI TRƯỜNG ---
MODEL_TYPE = os.getenv('MODEL_TYPE', 'new')
CKPT_PATH = os.getenv('CKPT_HF_PATH')
VOCAB_PATH = os.getenv('VOCAB_HF_PATH')
WORD_LIMIT = int(os.getenv('WORD_LIMIT', 1000))
DISPLAY_NAME = os.getenv('DISPLAY_NAME', 'Unknown Model')
INIT_PARAMS_JSON = os.getenv('INIT_PARAMS_JSON', '{}')
INIT_PARAMS = json.loads(INIT_PARAMS_JSON)

# Đường dẫn đến các thư mục thư viện
PATH_TO_OLD_F5_REPO = os.path.abspath('f5_tts_old')
PATH_TO_NEW_F5_REPO = os.path.abspath('f5_tts_new')

if not CKPT_PATH or not VOCAB_PATH:
    raise ValueError("Lỗi: CKPT_PATH hoặc VOCAB_PATH chưa được thiết lập.")

# --- PHẦN 2: TẢI MODEL & CACHING LOGIC ---
tts_instance = None
last_ref_audio_path = None

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
    print(f"Đang tải model với các tham số: {INIT_PARAMS}")
    sys.path.insert(0, PATH_TO_NEW_F5_REPO)
    from f5tts_wrapper import F5TTSWrapper

    # Thêm các tham số bắt buộc vào từ điển
    INIT_PARAMS['ckpt_path'] = str(cached_path(CKPT_PATH))
    INIT_PARAMS['vocab_file'] = str(cached_path(VOCAB_PATH))
    
    # Dùng unpacking để truyền tất cả tham số từ INIT_PARAMS
    # Logic này vẫn đúng sau khi đã sửa cấu hình trong Colab
    tts_instance = F5TTSWrapper(**INIT_PARAMS)
    
    sys.path.pop(0)
    print("✅ Tải model MỚI thành công.")

def post_process(text):
    text = " " + text.replace('"', "") + " "
    text = text.replace(" . . ", " . ").replace(" .. ", " . ")
    text = text.replace(" , , ", " , ").replace(" ,, ", " , ")
    return " ".join(text.split())

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float, cfg_strength: float):
    global last_ref_audio_path
    if not ref_audio_orig:
        raise gr.Error("Vui lòng tải lên một tệp âm thanh mẫu.")
    if not gen_text.strip():
        raise gr.Error("Vui lòng nhập nội dung văn bản.")
    if len(gen_text.strip().split()) > WORD_LIMIT:
        raise gr.Error(f"Văn bản quá dài. Giới hạn là {WORD_LIMIT} từ.")
    
    try:
        if MODEL_TYPE == 'old':
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
            
            if ref_audio_orig != last_ref_audio_path:
                print(f"Phát hiện âm thanh mẫu mới. Đang xử lý: {ref_audio_orig}")
                tts_instance.preprocess_reference(
                    ref_audio_path=ref_audio_orig,
                    ref_text="",
                    clip_short=True
                )
                last_ref_audio_path = ref_audio_orig
                print(f"Xử lý giọng mẫu thành công. Độ dài sử dụng: {tts_instance.get_current_audio_length():.2f} giây")
            else:
                print("Sử dụng giọng mẫu đã được xử lý trước đó.")
            
            final_text = TTSnorm(gen_text)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                output_path = tmp_wav.name
            
            print(f"Đang tạo giọng nói với cfg_strength={cfg_strength}, speed={speed}...")
            
            tts_instance.generate(
                text=final_text,
                output_path=output_path,
                nfe_step=32,
                cfg_strength=cfg_strength,
                speed=speed,
                cross_fade_duration=0.15,
                sway_sampling_coef=-1,
            )
            print("Tạo giọng nói thành công.")
            
            # <<< SỬA LỖI Ở ĐÂY >>>
            # Đổi thứ tự final_wave và final_sr để khớp với output của sf.read
            final_wave, final_sr = sf.read(output_path)
            
            sys.path.pop(0)

            # Bây giờ (final_sr, final_wave) sẽ ở đúng định dạng (int, numpy_array)
            return (final_sr, final_wave), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Lỗi khi tạo giọng nói: {e}")

# --- Giao diện Gradio không đổi ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech")
    gr.Markdown(f"### Model: **{DISPLAY_NAME}** | Kiến trúc: **{MODEL_TYPE.upper()}** | Giới hạn: **{WORD_LIMIT} từ**")

    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Âm thanh mẫu", type="filepath")
        gen_text = gr.Textbox(label="📝 Văn bản", placeholder="Nhập văn bản...", lines=3)
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
        cfg_strength_slider = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="🗣️ Độ bám sát giọng mẫu", info="Giá trị cao hơn sẽ bắt chước giọng mẫu mạnh hơn. Mặc định ~2.0-3.0")
        
    btn = gr.Button("🔥 Tạo giọng nói", variant="primary")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram (Chỉ có ở model cũ)")

    btn.click(infer_tts, inputs=[ref_audio, gen_text, speed, cfg_strength_slider], outputs=[output_audio, output_spectrogram])

demo.queue().launch(share=True)