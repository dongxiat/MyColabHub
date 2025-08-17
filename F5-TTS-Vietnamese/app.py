import spaces
import os
import sys
import gradio as gr
import json
import numpy as np
from datetime import datetime
from huggingface_hub import login
from cached_path import cached_path
import tempfile
import soundfile as sf
import contextlib

from gradio.themes.utils import colors, fonts

# --- Theme ---
latte = gr.themes.Citrus(primary_hue=colors.yellow, secondary_hue=colors.rose, neutral_hue=colors.gray, font=fonts.GoogleFont("Inter"), font_mono=fonts.GoogleFont("JetBrains Mono")).set(body_background_fill="#eff1f5", body_text_color="#4c4f69", border_color_primary="#bcc0cc", block_background_fill="#ffffff", button_primary_background_fill="#df8e1d", button_primary_text_color="#ffffff", link_text_color="#dc8a78")

# --- Tải Giọng Mẫu Có Sẵn ---
SAMPLES_DIR = "samples"
SAMPLES_CONFIG = os.path.join(SAMPLES_DIR, "samples.json")

def load_samples():
    sample_names = ["(Tự tải lên giọng của bạn)"]
    sample_lookup = {}
    if os.path.exists(SAMPLES_CONFIG):
        try:
            with open(SAMPLES_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for sample in data:
                if 'name' in sample and 'audio_path' in sample and 'ref_text' in sample:
                    sample_names.append(sample['name'])
                    sample_lookup[sample['name']] = sample
            print(f"✅ Đã tải thành công {len(sample_lookup)} giọng mẫu.")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file samples.json: {e}")
    else:
        print("⚠️ Không tìm thấy file samples.json. Bỏ qua việc tải giọng mẫu có sẵn.")
    return sample_names, sample_lookup

sample_names, sample_lookup = load_samples()

# --- Tải Model ---
MODEL_TYPE = os.getenv('MODEL_TYPE', 'new')
CKPT_PATH = os.getenv('CKPT_HF_PATH')
VOCAB_PATH = os.getenv('VOCAB_HF_PATH')
WORD_LIMIT = int(os.getenv('WORD_LIMIT', 1000))
DISPLAY_NAME = os.getenv('DISPLAY_NAME', 'Unknown Model')
INIT_PARAMS_JSON = os.getenv('INIT_PARAMS_JSON', '{}')
INIT_PARAMS = json.loads(INIT_PARAMS_JSON)

PATH_TO_OLD_F5_REPO = os.path.abspath('f5_tts_old')
PATH_TO_NEW_F5_REPO = os.path.abspath('f5_tts_new')

tts_instance = None
print(f"Nhận diện loại model: {MODEL_TYPE}")

if MODEL_TYPE == 'old':
    print("Đang tải model theo kiến trúc F5-TTS Base...")
    sys.path.insert(0, PATH_TO_OLD_F5_REPO)
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_vocoder, load_model
    vocoder = load_vocoder()
    model = load_model(
        model_cls=DiT, 
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
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
    from safetensors.torch import load_file
    INIT_PARAMS['ckpt_path'] = str(cached_path(CKPT_PATH))
    INIT_PARAMS['vocab_file'] = str(cached_path(VOCAB_PATH))
    tts_instance = F5TTSWrapper(**INIT_PARAMS)
    sys.path.pop(0)
    print("✅ Tải model MỚI thành công.")

ref_audio_path_old, ref_text_processed_old = None, None

# --- CÁC HÀM XỬ LÝ LOGIC ---
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

def handle_preprocess(audio_path, text, clip_short, progress):
    progress(0, desc="Đang xử lý giọng mẫu...")
    gr.Info("Đã nhận audio mẫu. Bắt đầu xử lý...")
    
    if MODEL_TYPE == 'new':
        with suppress_outputs(PATH_TO_NEW_F5_REPO):
            processed_path, transcribed_text = tts_instance.preprocess_reference(
                ref_audio_path=audio_path, ref_text=text, clip_short=clip_short
            )
    else: # Model 'old'
        global ref_audio_path_old, ref_text_processed_old
        with suppress_outputs(PATH_TO_OLD_F5_REPO):
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            processed_path, transcribed_text = preprocess_ref_audio_text(audio_path, text, clip_short=clip_short)
            ref_audio_path_old = processed_path
            ref_text_processed_old = transcribed_text
    
    progress(1, desc="Xử lý giọng mẫu hoàn tất!")
    gr.Info("Xử lý giọng mẫu hoàn tất!")
    print(f"Xử lý giọng mẫu hoàn tất.\n Đường dẫn: {processed_path},\n Văn bản của giọng mẫu: '{transcribed_text}'")
    return processed_path, transcribed_text

@spaces.GPU
def select_sample(sample_name, progress=gr.Progress()):
    if not sample_name or sample_name == sample_names[0]:
        return None, ""
    sample_data = sample_lookup[sample_name]
    audio_path = os.path.join(SAMPLES_DIR, sample_data['audio_path'])
    return handle_preprocess(audio_path, sample_data['ref_text'], clip_short=False, progress=progress)

@spaces.GPU
def process_manual_upload(ref_audio_orig, progress=gr.Progress()):
    if not ref_audio_orig: return None, "", sample_names[0]
    processed_path, transcribed_text = handle_preprocess(ref_audio_orig, "", clip_short=True, progress=progress)
    return processed_path, transcribed_text, sample_names[0]

@spaces.GPU
def infer_tts(ref_audio_path, ref_text_from_ui, gen_text, speed, cfg_strength, nfe_step, progress=gr.Progress()):
    is_ready = (tts_instance.ref_audio_processed is not None) if MODEL_TYPE == 'new' else (ref_audio_path_old is not None)
    if not is_ready:
        raise gr.Error("Lỗi: Vui lòng chọn hoặc tải lên một giọng mẫu trước khi tạo giọng nói.")
    if not gen_text.strip():
        raise gr.Error("Vui lòng nhập nội dung văn bản.")
    
    try:
        text_from_ui = ref_text_from_ui.strip()
        
        if MODEL_TYPE == 'new':
            if text_from_ui and text_from_ui != tts_instance.ref_text:
                handle_preprocess(tts_instance.last_processed_audio_path, text_from_ui, clip_short=False, progress=progress)
        else:
            if text_from_ui and text_from_ui != ref_text_processed_old:
                handle_preprocess(ref_audio_path, text_from_ui, clip_short=False, progress=progress)
        
        print(f"Bắt đầu tạo audio cho văn bản...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        spectrogram_path = None
        if MODEL_TYPE == 'new':
            with suppress_outputs(PATH_TO_NEW_F5_REPO):
                from vinorm import TTSnorm
                final_text = TTSnorm(gen_text)
                tts_instance.generate(text=final_text, output_path=tmp_path, nfe_step=nfe_step, cfg_strength=cfg_strength, speed=speed, progress_callback=progress)
        else:
            with suppress_outputs(PATH_TO_OLD_F5_REPO):
                from vinorm import TTSnorm
                from f5_tts.infer.utils_infer import infer_process, save_spectrogram
                final_wave, final_sr, spectrogram = infer_process(
                    ref_audio=ref_audio_path_old, 
                    ref_text=ref_text_processed_old, 
                    gen_text=TTSnorm(gen_text),
                    model_obj=tts_instance['model'], 
                    vocoder=tts_instance['vocoder'], 
                    speed=speed, 
                    nfe_step=nfe_step,
                    progress=progress # <<< SỬA LỖI: TRUYỀN ĐÚNG ĐỐI TƯỢNG PROGRESS
                )
                sf.write(tmp_path, final_wave, final_sr)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spec:
                    spectrogram_path = tmp_spec.name
                    save_spectrogram(spectrogram, spectrogram_path)

        final_wave, final_sr = sf.read(tmp_path)
        os.remove(tmp_path)
        
        return (final_sr, final_wave), spectrogram_path
    except Exception as e:
        import traceback; traceback.print_exc()
        raise gr.Error(f"Lỗi khi tạo giọng nói: {e}")

# --- Giao diện Gradio ---
with gr.Blocks(theme=latte) as demo:
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech")
    gr.Markdown(f"### Model: **{DISPLAY_NAME}** | Kiến trúc: **{MODEL_TYPE.upper()}** | Giới hạn: **{WORD_LIMIT} từ**")

    with gr.Row():
        sample_dropdown = gr.Dropdown(choices=sample_names, value=sample_names[0], label="👇 Chọn một giọng mẫu có sẵn", interactive=True)
    
    with gr.Row():
        with gr.Column(scale=1):
            ref_audio_ui = gr.Audio(label="1. Tải lên hoặc chọn âm thanh mẫu", type="filepath")
            ref_text_ui = gr.Textbox(label="2. Văn bản của giọng mẫu", placeholder="Nội dung audio mẫu sẽ tự động xuất hiện ở đây.", lines=5, interactive=True)
        with gr.Column(scale=2):
            gen_text_ui = gr.Textbox(label="3. Nhập văn bản cần tạo giọng nói", placeholder="Nhập văn bản dài vào đây...", lines=11)
    
    with gr.Row():
        speed_slider = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
        cfg_strength_slider = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="🗣️ Độ bám sát giọng mẫu")
    
    with gr.Accordion("🛠️ Cài đặt nâng cao", open=False):
        nfe_step_slider = gr.Slider(minimum=16, maximum=64, value=32, step=2, label="🔍 Số bước khử nhiễu (NFE)", info="Cao hơn = chậm hơn nhưng có thể chất lượng tốt hơn. Thấp hơn = nhanh hơn.")

    btn = gr.Button("🔥 4. Tạo giọng nói", variant="primary")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram (Chỉ có ở model cũ)", visible=(MODEL_TYPE == 'old'))

    def reset_to_upload():
        return None, "", sample_names[0]

    sample_dropdown.change(fn=select_sample, inputs=[sample_dropdown], outputs=[ref_audio_ui, ref_text_ui], show_progress="full")
    ref_audio_ui.upload(fn=process_manual_upload, inputs=[ref_audio_ui], outputs=[ref_audio_ui, ref_text_ui, sample_dropdown], show_progress="full")
    ref_audio_ui.clear(fn=reset_to_upload, outputs=[ref_audio_ui, ref_text_ui, sample_dropdown])
    
    btn.click(
        fn=infer_tts, 
        inputs=[ref_audio_ui, ref_text_ui, gen_text_ui, speed_slider, cfg_strength_slider, nfe_step_slider], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch()