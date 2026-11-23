"""Download the VoxCPM2 model (about 4 GB) and say one test sentence.
Afterwards the voices page can run offline: HF_HUB_OFFLINE=1 python voxcpm_voices.py"""
import os
import soundfile as sf
from voxcpm import VoxCPM

home = os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")
os.makedirs(home, exist_ok=True)
print("Downloading the VoxCPM2 model (about 4 GB)...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
print("Model loaded. Making a test sentence...")
wav = model.generate(text="Hello, VoxCPM2 is working on this machine.", cfg_value=2.0, inference_timesteps=10)
out = os.path.join(home, "test_tts.wav")
sf.write(out, wav, model.tts_model.sample_rate)
print(f"Done, listen to {out}")
