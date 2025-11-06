# src/sort_asvspoof.py
import os
import shutil
import glob
import requests
import zipfile
from pathlib import Path
from pydub import AudioSegment

# ... Paths and other constants are unchanged ...
# 1. Ensure ffmpeg exists
FFMPEG_DIR = Path("C:/ffmpeg/bin")
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"

if not FFMPEG_EXE.exists():
    print("⚡ ffmpeg not found, downloading...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = Path("ffmpeg.zip")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("C:/")
    
    # FIX: More robust renaming logic
    try:
        # Find the extracted folder (e.g., ffmpeg-6.x-essentials_build)
        extracted_dir = next(Path("C:/").glob("ffmpeg-*"))
        target_dir = Path("C:/ffmpeg")
        if not target_dir.exists():
            extracted_dir.rename(target_dir)
            print("✅ ffmpeg installed at C:/ffmpeg/bin")
        else:
            print("ℹ️ ffmpeg directory already exists. Assuming it's installed.")
    except StopIteration:
        print("❌ Could not find the extracted ffmpeg folder.")
    finally:
        if zip_path.exists():
            zip_path.unlink() # Clean up downloaded zip

# Force pydub to use this ffmpeg
AudioSegment.converter = str(FFMPEG_EXE)

# ... The rest of the script is unchanged ...
ROOT = r"N:\Datasets\LA"
PROTOCOL_DIR = os.path.join(ROOT, "ASVspoof2019_LA_cm_protocols")
SRC_DIRS = {
    "train": os.path.join(ROOT, "ASVspoof2019_LA_train", "flac"),
    "dev": os.path.join(ROOT, "ASVspoof2019_LA_dev", "flac"),
    "eval": os.path.join(ROOT, "ASVspoof2019_LA_eval", "flac"),
}
DST_REAL = r"N:\Datasets\audio\raw\real"
DST_FAKE = r"N:\Datasets\audio\raw\fake"
PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof201 miscellaneousLA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

def copy_and_convert(split, protocol_file):
    # ... function implementation is unchanged ...
    pass 

if __name__ == "__main__":
    # ... main execution block is unchanged ...
    pass