# src/prepare_audio.py
# Convert raw .wav audio → spectrogram PNGs

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # <--- FIX: Added missing import
from pathlib import Path
from src.config import AUDIO_RAW, AUDIO_SPECS

os.makedirs(os.path.join(AUDIO_SPECS, "real"), exist_ok=True)
os.makedirs(os.path.join(AUDIO_SPECS, "fake"), exist_ok=True)

def wav_to_spec(in_path, out_path):
    try:
        y, sr = librosa.load(in_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3,3))
        librosa.display.specshow(S_db, sr=sr, cmap="magma")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"❌ Failed {in_path}: {e}")

def process_folder(label):
    in_dir = os.path.join(AUDIO_RAW, label)
    out_dir = os.path.join(AUDIO_SPECS, label)
    print(f"Processing {len(os.listdir(in_dir))} files in {in_dir}...")
    for fname in os.listdir(in_dir):
        if not fname.endswith(".wav"):
            continue
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, Path(fname).stem + ".png")
        if os.path.exists(out_path):
            continue
        wav_to_spec(in_path, out_path)

if __name__ == "__main__":
    for lbl in ["real", "fake"]:
        print(f">> Processing {lbl} audio...")
        process_folder(lbl)
    print("\n✅ All spectrograms saved in:", AUDIO_SPECS)