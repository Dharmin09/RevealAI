# src/make_split.py
# Create train/val/test splits for both audio and video datasets.
# Expects:
#   N:\Datasets\data\audio\specs\real, N:\Datasets\data\audio\specs\fake   (spectrograms)
#   N:\Datasets\data\video\frames\real, N:\Datasets\data\video\frames\fake (frames)
# It will generate text files listing file paths with numeric labels.

import os
import random
from glob import glob
from src.config import AUDIO_SPECS, VIDEO_FRAMES, SPLITS_DIR

os.makedirs(SPLITS_DIR, exist_ok=True)

def write_split(items, out_file):
    with open(out_file, "w") as f:
        for path, label in items:
            f.write(f"{path} {label}\n")
    print(f"✔ Wrote {len(items)} items → {out_file}")

def make_splits(data_dir, exts, prefix):
    if not os.path.exists(data_dir):
        print(f"❌ Missing directory: {data_dir}")
        return

    items = []
    for label, lbl_num in (("real", 0), ("fake", 1)):
        subdir = os.path.join(data_dir, label)
        if not os.path.exists(subdir):
            print(f"⚠️ Skipping missing folder: {subdir}")
            continue
        for ext in exts:
            files = glob(os.path.join(subdir, f"**/*{ext}"), recursive=True)
            for f in files:
                items.append((f, lbl_num))

    if not items:
        print(f"⚠️ No files found in {data_dir}")
        return

    random.shuffle(items)
    n = len(items)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)

    train_items = items[:n_train]
    val_items   = items[n_train:n_train+n_val]
    test_items  = items[n_train+n_val:]

    write_split(train_items, os.path.join(SPLITS_DIR, f"{prefix}_train.txt"))
    write_split(val_items, os.path.join(SPLITS_DIR, f"{prefix}_val.txt"))
    write_split(test_items, os.path.join(SPLITS_DIR, f"{prefix}_test.txt"))

def main():
    print(">> Creating splits...")
    make_splits(AUDIO_SPECS, [".png"], "audio")           # spectrograms
    make_splits(VIDEO_FRAMES, [".jpg", ".png"], "video")  # frames
    print("\n✅ All splits created in:", SPLITS_DIR)

if __name__ == "__main__":
    main()
