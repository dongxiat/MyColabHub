import gradio as gr
import torch
import torchaudio
import gc
import os
import subprocess

# Cố gắng import resemble_enhance, nếu không được thì in hướng dẫn
try:
    from resemble_enhance.enhancer.inference import denoise, enhance
except ImportError:
    print("Could not import from resemble_enhance. Please ensure the package is installed correctly.")
    exit()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def clear_gpu_cash():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _fn(path, solver, nfe, tau, chunk_seconds, chunks_overlap, denoising):
    """Processes a single audio file."""
    if path is None:
        return None, None, "Please provide an input audio file."

    try:
        solver = solver.lower()
        nfe = int(nfe)
        lambd = 0.9 if denoising else 0.1

        dwav, sr = torchaudio.load(path)
        dwav = dwav.mean(dim=0)

        # Thực hiện enhance và denoise
        wav_denoised, new_sr_denoised = denoise(dwav, sr, device)
        wav_enhanced, new_sr_enhanced = enhance(dwav=dwav, sr=sr, device=device, nfe=nfe, chunk_seconds=chunk_seconds, chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)

        # Chuyển đổi sang numpy array để Gradio có thể hiển thị
        wav_denoised_np = wav_denoised.cpu().numpy()
        wav_enhanced_np = wav_enhanced.cpu().numpy()

        clear_gpu_cash()
        
        return (new_sr_denoised, wav_denoised_np), (new_sr_enhanced, wav_enhanced_np), "Processing successful!"
    except Exception as e:
        clear_gpu_cash()
        return None, None, f"An error occurred: {str(e)}"


def process_folder(in_dir, out_dir, denoise_only):
    """Processes a batch of files in a folder using the command-line tool."""
    if not in_dir or not out_dir:
        return "Input and Output directories cannot be empty."
    
    if not os.path.isdir(in_dir):
        return f"Error: Input directory does not exist: {in_dir}"

    os.makedirs(out_dir, exist_ok=True)

    try:
        yield f"Starting to process folder '{in_dir}'..."
        
        command = [
            "resemble_enhance",
            in_dir,
            out_dir,
            "--device", device,
        ]

        if denoise_only:
            command.append("--denoise-only")
        
        # Chạy lệnh bên trong môi trường ảo của uv
        process_command = ["uv", "run", "--"] + command
        process = subprocess.Popen(process_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        full_log = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                full_log += output
                yield full_log

        process.wait()

        if process.returncode == 0:
            yield full_log + f"\n\nSuccess! Processed files are in '{out_dir}'"
        else:
            yield full_log + f"\n\nAn error occurred during batch processing."

    except FileNotFoundError:
        yield "Error: 'uv' or 'resemble_enhance' command not found."
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"


def main():
    # Giao diện cho xử lý từng file
    with gr.Blocks() as single_file_interface:
        gr.Markdown("## Resemble Enhance - Single File")
        gr.Markdown("Upload an audio file to enhance its quality.")
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(type="filepath", label="Input Audio")
                denoise_before_enhance = gr.Checkbox(value=False, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")
                
                with gr.Accordion("Advanced Settings", open=False):
                    solver = gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver")
                    nfe = gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations")
                    tau = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature")
                    chunk_seconds = gr.Slider(minimum=1, maximum=40, value=10, step=1, label="Chunk seconds (more seconds more VRAM usage)")
                    chunks_overlap = gr.Slider(minimum=0, maximum=5, value=1, step=0.5, label="Chunk overlap")
                
                process_btn = gr.Button("Start Processing", variant="primary")

            with gr.Column():
                output_denoised = gr.Audio(label="Output Denoised Audio")
                output_enhanced = gr.Audio(label="Output Enhanced Audio")
                status_label_single = gr.Label(label="Status")

        process_btn.click(
            fn=_fn,
            inputs=[input_audio, solver, nfe, tau, chunk_seconds, chunks_overlap, denoise_before_enhance],
            outputs=[output_denoised, output_enhanced, status_label_single],
        )

    # Giao diện cho xử lý hàng loạt
    with gr.Blocks() as batch_folder_interface:
        gr.Markdown("## Resemble Enhance - Batch Processing")
        gr.Markdown("Specify input and output directories to process all files in bulk.")
        
        in_dir = gr.Textbox(label="Input Directory", placeholder="/content/drive/MyDrive/input_audios")
        out_dir = gr.Textbox(label="Output Directory", placeholder="/content/drive/MyDrive/output_audios")
        
        denoise_only = gr.Checkbox(value=False, label="Denoise Only (runs with --denoise-only flag)")
        
        process_folder_btn = gr.Button("Start Batch Processing", variant="primary")
        status_label_batch = gr.Textbox(label="Status Log", lines=10, interactive=False)

        process_folder_btn.click(
            fn=process_folder,
            inputs=[in_dir, out_dir, denoise_only],
            outputs=[status_label_batch]
        )

    # Kết hợp 2 giao diện bằng Tab
    interface = gr.TabbedInterface(
        [single_file_interface, batch_folder_interface],
        ["Single File Processing", "Batch Folder Processing"]
    )
    
    interface.launch()


if __name__ == "__main__":
    main()