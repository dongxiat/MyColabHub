# %%
import sys
import torch
import torchaudio
from resemble_enhance.enhancer.train import Enhancer, HParams
from resemble_enhance.inference import inference

import pathlib
from pathlib import Path
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
def load_enhancer(device, run_dir='resemble_enhance/model_repo/enhancer_stage2'):
    run_dir = Path(run_dir)
    hp = HParams.load(run_dir)
    enhancer = Enhancer(hp)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu")["module"]
    enhancer.load_state_dict(state_dict)
    enhancer.eval()
    enhancer.to(device)
    return enhancer

# %%
enhancer = load_enhancer(device)

def denoise(dwav, sr, device):
    return inference(model=enhancer.denoiser, dwav=dwav, sr=sr, device=device)

def enhance(chunk_seconds, chunks_overlap,dwav, sr, device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5):
    assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
    assert solver in ("midpoint", "rk4", "euler"), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
    assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
    assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"
    enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    return inference(model=enhancer, chunk_seconds=chunk_seconds, overlap_seconds=chunks_overlap, dwav=dwav, sr=sr, device=device)

# %%
path = Path("./test.wav")
nfe = 90 #CFM Number of Function Evaluations [1, 128]
tau = 0.65 #CFM Prior Temperature [0, 1]
solver = "Midpoint".lower() #"Midpoint", "RK4", "Euler"
chunk_seconds = 999
chunks_overlap = 1
denoising = True

lambd = 0.9 if denoising else 0.1

# %%
dwav, sr = torchaudio.load(path)
dwav = dwav.mean(dim=0)

#wav_denoise, new_sr = denoise(dwav, sr, device)
wav_enhanced, new_sr = enhance(dwav = dwav, sr = sr, device = device, nfe=nfe,chunk_seconds=chunk_seconds,chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)

# %%
downsampled_wav_enhanced = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sr)(wav_enhanced) #back to original sr

# %%
import os
import soundfile as sf

# Define output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save Original audio
sf.write(os.path.join(output_dir, 'original.wav'), dwav, sr)

# Save Enhanced audio
sf.write(os.path.join(output_dir, 'enhanced.wav'), wav_enhanced, new_sr)

# Save Downsampled Enhanced audio
sf.write(os.path.join(output_dir, 'downsampled_enhanced.wav'), downsampled_wav_enhanced, sr)
