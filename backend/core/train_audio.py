# src/train_audio.py
# Train CNN on spectrogram images using predefined splits (.txt files)

import argparse
import json
import os
import sys

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
from tqdm.keras import TqdmCallback

sys.path.append('n:/Datasets/src')

from core.config import AUDIO_MODEL_PATH, IMG_SIZE_AUDIO, MODELS_DIR, SPLITS_DIR

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 12


def build_audio_model(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(), MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(), MaxPool2D(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(), MaxPool2D(),
        Flatten(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_split(split_file):
    if not os.path.exists(split_file):
        print(f"Error: Split file not found at {split_file}")
        return []
    items = []
    with open(split_file, "r") as f:
        for line in tqdm(f, desc=f"Loading split file {os.path.basename(split_file)}"):
            path, label = line.strip().split()
            label = 0 if label.lower() in ["real", "bonafide", "0"] else 1
            items.append((path, label))
    return items


def make_generator(items, target_size, batch_size, shuffle=True):
    n = len(items)
    while True:
        if shuffle:
            np.random.shuffle(items)
        for i in range(0, n, batch_size):
            batch = items[i:i + batch_size]
            X, y = [], []
            for path, label in batch:
                img = load_img(path, target_size=target_size)
                img_arr = img_to_array(img)
                X.append(img_arr / 255.0)
                onehot = np.zeros(2)
                onehot[label] = 1
                y.append(onehot)
            yield np.array(X), np.array(y)


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio classifier with optional correction data")
    parser.add_argument("--corrections-json", type=str, default=None,
                        help="Path to corrections JSON produced by the continuous learning system")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size")
    return parser.parse_args()


def load_corrections(json_path):
    if not json_path:
        return []
    if not os.path.exists(json_path):
        print(f"⚠️ Corrections file not found: {json_path}")
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except Exception as exc:
        print(f"⚠️ Failed to load corrections file {json_path}: {exc}")
        return []

    corrections = []
    for item in entries:
        if item.get("type") not in (None, "audio", "unknown"):
            continue
        file_path = item.get("file")
        label = item.get("label")
        if not file_path or not os.path.exists(file_path):
            continue
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        corrections.append((file_path, label_int))
    if corrections:
        print(f"🔁 Loaded {len(corrections)} correction samples for audio retraining")
    return corrections


def main(cli_args=None):
    args = cli_args or parse_args()

    if not os.path.exists(SPLITS_DIR):
        print(f"Error: Splits directory not found at {SPLITS_DIR}")
        return

    train_items = load_split(os.path.join(SPLITS_DIR, "audio_train.txt"))
    val_items = load_split(os.path.join(SPLITS_DIR, "audio_val.txt"))
    test_items = load_split(os.path.join(SPLITS_DIR, "audio_test.txt"))

    if not train_items or not val_items or not test_items:
        print("Error: Could not load split files. Aborting.")
        return

    correction_items = load_corrections(args.corrections_json)
    if correction_items:
        train_items.extend(correction_items)

    batch_size = max(1, args.batch_size)
    train_gen = make_generator(train_items, IMG_SIZE_AUDIO, batch_size, shuffle=True)
    val_gen = make_generator(val_items, IMG_SIZE_AUDIO, batch_size, shuffle=False)
    test_gen = make_generator(test_items, IMG_SIZE_AUDIO, batch_size, shuffle=False)

    steps_train = max(1, len(train_items) // batch_size)
    steps_val = max(1, len(val_items) // batch_size)
    steps_test = max(1, len(test_items) // batch_size)

    model = build_audio_model(input_shape=(IMG_SIZE_AUDIO[0], IMG_SIZE_AUDIO[1], 3))
    os.makedirs(MODELS_DIR, exist_ok=True)

    chk = ModelCheckpoint(AUDIO_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    model.fit(train_gen,
              validation_data=val_gen,
              steps_per_epoch=steps_train,
              validation_steps=steps_val,
              epochs=max(1, args.epochs),
              callbacks=[chk, rlr, TqdmCallback(verbose=2)],
              verbose=0)

    print("\n✅ Training finished. Best model saved to:", AUDIO_MODEL_PATH)

    print("\n🔎 Evaluating on test set...")
    loss, acc = model.evaluate(test_gen, steps=steps_test, verbose=1)
    print(f"\nTest accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
