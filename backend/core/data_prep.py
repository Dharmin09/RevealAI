# sort_asvspoof_auto.py
# 1. Downloads ffmpeg if not installed
# 2. Converts ASVspoof FLAC -> WAV
# 3. Sorts into real/fake folders

import os
import shutil
import glob
import requests
import zipfile
from pathlib import Path
from pydub import AudioSegment

# -------------------------
# 1. Ensure ffmpeg exists
# -------------------------
FFMPEG_DIR = Path("C:/ffmpeg/bin")
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"

if not FFMPEG_EXE.exists():
    print("⚡ ffmpeg not found, downloading...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = Path("ffmpeg.zip")

    # Download zip
    with requests.get(url, stream=True) as r:
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("C:/")

    # Find extracted folder (e.g., ffmpeg-6.x-full_build)
    extracted_dir = next(Path("C:/").glob("ffmpeg-*"))
    if not FFMPEG_DIR.parent.exists():
        extracted_dir.rename("C:/ffmpeg")
    print("✅ ffmpeg installed at C:/ffmpeg/bin")

# Force pydub to use this ffmpeg
AudioSegment.converter = str(FFMPEG_EXE)

# -------------------------
# 2. Paths for dataset
# -------------------------
ROOT = r"N:\Datasets\LA"
PROTOCOL_DIR = os.path.join(ROOT, "ASVspoof2019_LA_cm_protocols")

SRC_DIRS = {
    "train": os.path.join(ROOT, "ASVspoof2019_LA_train", "flac"),
    "dev": os.path.join(ROOT, "ASVspoof2019_LA_dev", "flac"),
    "eval": os.path.join(ROOT, "ASVspoof2019_LA_eval", "flac"),
}

DST_REAL = r"N:\Datasets\audio\raw\real"
DST_FAKE = r"N:\Datasets\audio\raw\fake"

os.makedirs(DST_REAL, exist_ok=True)
os.makedirs(DST_FAKE, exist_ok=True)

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

# -------------------------
# 3. Copy & convert
# -------------------------
def copy_and_convert(split, protocol_file):
    src_dir = SRC_DIRS[split]
    protocol_path = os.path.join(PROTOCOL_DIR, protocol_file)

    print(f"\nProcessing {split} split using {protocol_file}")

    with open(protocol_path, "r") as f:
        lines = f.readlines()

    real_count, fake_count = 0, 0
    for line in lines:
        parts = line.strip().split()
        filename, label = parts[0], parts[-1]  # file ID, label

        matches = glob.glob(os.path.join(src_dir, "**", filename + ".flac"), recursive=True)
        if not matches:
            continue
        src = matches[0]

        dst_dir = DST_REAL if label == "bonafide" else DST_FAKE
        dst_wav = os.path.join(dst_dir, filename + ".wav")

        if os.path.exists(dst_wav):
            continue

        try:
            audio = AudioSegment.from_file(src, format="flac")
            audio.export(dst_wav, format="wav")
            if label == "bonafide":
                real_count += 1
            else:
                fake_count += 1
        except Exception as e:
            print(f"❌ Error converting {src}: {e}")

    print(f"✔ {split} done → {real_count} real, {fake_count} fake files converted")

if __name__ == "__main__":
    for split, proto_file in PROTOCOLS.items():
        copy_and_convert(split, proto_file)

    print("\n✅ All done! Files are in:")
    print(f" - {DST_REAL}")
    print(f" - {DST_FAKE}")