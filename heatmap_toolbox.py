#!/usr/bin/env python3
"""Unified utilities for RevealAI heatmap diagnostics."""
import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "COLAB_GPU" in os.environ:
    del os.environ["COLAB_GPU"]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

from backend.core.infer import find_last_conv_layer, infer_video, load_models
from backend.core.utils import create_thermal_colormap, make_gradcam_heatmap, overlay_heatmap_on_image
from backend.core.config import IMG_SIZE_VIDEO

DEFAULT_API_URL = "http://localhost:5000/api/analyze-video"
DEFAULT_REPORT_URL = "http://localhost:5000/api/generate-report"


def resolve_video(path_hint: Optional[Path]) -> Path:
    if path_hint:
        candidate = path_hint.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Video file not found: {candidate}")
        return candidate

    search_roots = [
        PROJECT_ROOT / "video" / "raw" / "real",
        PROJECT_ROOT / "video" / "raw" / "fake",
    ]
    for root in search_roots:
        if root.exists():
            matches = sorted(root.glob("*.mp4"))
            if matches:
                return matches[0]
    raise FileNotFoundError("No MP4 test video found in video/raw/{real,fake}.")


def decode_base64_image(payload: str) -> np.ndarray:
    if not payload:
        raise ValueError("Empty base64 payload")
    data = payload.split(",")[-1]
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw))
    return np.array(img)


def summarize_heatmaps(name: str, frames: Iterable[np.ndarray]) -> None:
    frames = list(frames)
    print(f"[{name}] count={len(frames)}")
    for idx, frame in enumerate(frames, start=1):
        if frame is None:
            print(f"  #{idx}: missing")
            continue
        arr = np.array(frame)
        shape = arr.shape
        dtype = arr.dtype
        rng = (arr.min(), arr.max()) if arr.size else (None, None)
        print(f"  #{idx}: shape={shape} dtype={dtype} range={rng}")
        if arr.ndim == 3 and arr.shape[2] >= 3:
            red = arr[..., 0]
            green = arr[..., 1]
            blue = arr[..., 2]
            hotspot = np.sum((red > 180) & (green < 120))
            cool = np.sum((blue > 160) & (red < 120))
            total = arr.shape[0] * arr.shape[1]
            if total:
                print(
                    f"    red~: {hotspot/total:.1%} blue~: {cool/total:.1%}"
                )


def fetch_api_payload(api_url: str, video_path: Path, heatmap_frames: int) -> Tuple[dict, float]:
    with open(video_path, "rb") as handle:
        files = {"file": handle}
        data = {"heatmap_frames": str(heatmap_frames)}
        start = time.time()
        response = requests.post(api_url, files=files, data=data, timeout=120)
    elapsed = time.time() - start
    response.raise_for_status()
    return response.json(), elapsed


def command_colormap(_: argparse.Namespace) -> None:
    test_cases = [
        ("uniform_low", np.full((100, 100), 0.2, dtype=np.float32)),
        ("uniform_high", np.full((100, 100), 0.85, dtype=np.float32)),
        ("gradient", np.tile(np.linspace(0, 1, 128, dtype=np.float32), (128, 1))),
    ]
    print("[colormap] running synthetic checks...")
    for name, grid in test_cases:
        colorized = create_thermal_colormap(grid)
        print(f"  {name}: input=({grid.min():.3f},{grid.max():.3f}) output_shape={colorized.shape}")
        summarize_heatmaps(name, [colorized])


def command_gradcam(args: argparse.Namespace) -> None:
    import cv2

    video_model, _ = load_models()
    if video_model is None:
        raise RuntimeError("Video model unavailable. Ensure models/video_xception.h5 exists.")

    img_size = IMG_SIZE_VIDEO if isinstance(IMG_SIZE_VIDEO, tuple) else (IMG_SIZE_VIDEO, IMG_SIZE_VIDEO)
    height, width = img_size
    frame = np.random.rand(height, width, 3).astype(np.float32)
    batch = np.expand_dims(frame, axis=0)

    last_conv = find_last_conv_layer(video_model)
    heatmap = make_gradcam_heatmap(batch, video_model, last_conv, pred_index=args.class_index)
    heatmap_resized = cv2.resize(heatmap, (width, height))

    frame_uint8 = (frame * 255).astype(np.uint8)
    heatmap_layer, overlay = overlay_heatmap_on_image(
        frame_uint8,
        heatmap_resized,
        alpha=args.alpha,
        cmap="thermal",
        return_layers=True,
    )

    summarize_heatmaps("gradcam-layer", [heatmap_layer])
    summarize_heatmaps("gradcam-overlay", [overlay])

    if args.save:
        output = Path(args.save).expanduser().resolve()
        Image.fromarray(overlay).save(output)
        print(f"[gradcam] overlay saved to {output}")


def command_inference(args: argparse.Namespace) -> None:
    video_model, _ = load_models()
    if video_model is None:
        raise RuntimeError("Video model unavailable. Ensure models/video_xception.h5 exists.")

    video_path = resolve_video(args.video)
    print(f"[inference] analyzing {video_path}")
    result = infer_video(
        video_model,
        str(video_path),
        every_n_frames=args.every,
        max_frames=args.max_frames,
        heatmap_frames=args.heatmap_frames,
    )

    print(f"  video_score={result.get('video_score', 0.0):.3f}")
    summarize_heatmaps("heatmaps", result.get("heatmaps", []))

    if args.save_dir:
        out_dir = Path(args.save_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(result.get("heatmaps", []), start=1):
            Image.fromarray(np.array(frame)).save(out_dir / f"heatmap_{idx:02d}.png")
        for idx, frame in enumerate(result.get("original_frames", []), start=1):
            Image.fromarray(np.array(frame)).save(out_dir / f"original_{idx:02d}.png")
        print(f"[inference] frames saved to {out_dir}")


def command_api(args: argparse.Namespace) -> None:
    video_path = resolve_video(args.video)
    print(f"[api] calling {args.url} with {video_path}")
    payload, elapsed = fetch_api_payload(args.url, video_path, args.heatmap_frames)
    print(f"  status=200 time={elapsed:.1f}s")

    required_fields = {"video_score", "heatmaps", "verdict"}
    missing = required_fields - payload.keys()
    if missing:
        raise RuntimeError(f"Missing fields in response: {sorted(missing)}")

    heatmaps = payload.get("heatmaps", [])
    print(f"  heatmaps={len(heatmaps)} video_score={payload.get('video_score', 0.0):.3f}")
    frames = []
    originals = []
    for item in heatmaps:
        try:
            frames.append(decode_base64_image(item.get("heatmap", "")))
        except Exception as exc:
            print(f"    heatmap decode failed: {exc}")
        try:
            originals.append(decode_base64_image(item.get("original_frame", "")))
        except Exception as exc:
            print(f"    original decode failed: {exc}")
    summarize_heatmaps("api_heatmaps", frames)

    if args.include_pdf and heatmaps:
        pdf_payload = {
            "filename": video_path.name,
            "video_score": payload.get("video_score"),
            "audio_score": payload.get("audio_score"),
            "final_score": payload.get("combined_score"),
            "verdict": payload.get("verdict"),
            "heatmaps": [item.get("heatmap") for item in heatmaps],
            "original_frames": [item.get("original_frame") for item in heatmaps],
            "findings": payload.get("frame_metadata", {}),
        }
        response = requests.post(args.report_url, json=pdf_payload, timeout=120)
        if response.status_code == 200 and response.headers.get("content-type") == "application/pdf":
            out_path = Path(args.save_pdf).expanduser().resolve() if args.save_pdf else None
            if out_path:
                out_path.write_bytes(response.content)
                print(f"[api] pdf saved to {out_path}")
            else:
                print(f"[api] pdf generated ({len(response.content)} bytes)")
        else:
            raise RuntimeError(f"PDF generation failed: {response.status_code}")


def command_colors(args: argparse.Namespace) -> None:
    video_path = resolve_video(args.video)
    payload, _ = fetch_api_payload(args.url, video_path, args.heatmap_frames)
    heatmaps = payload.get("heatmaps", [])
    frames = []
    for item in heatmaps:
        try:
            frames.append(decode_base64_image(item.get("heatmap", "")))
        except Exception as exc:
            print(f"[colors] decode failed: {exc}")
    summarize_heatmaps("color_check", frames)


def command_mock_model(args: argparse.Namespace) -> None:
    import tensorflow as tf

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for _ in range(3):
        x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(2048, 3, padding="same", name="block14_sepconv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.save(output_path)
    print(f"[mock-model] saved mock video model to {output_path}")


COMMANDS = {
    "colormap": command_colormap,
    "gradcam": command_gradcam,
    "inference": command_inference,
    "api": command_api,
    "colors": command_colors,
    "mock-model": command_mock_model,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RevealAI heatmap toolbox")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("colormap", help="Run synthetic thermal colormap checks.")

    gradcam_parser = subparsers.add_parser("gradcam", help="Generate Grad-CAM overlay on dummy frames.")
    gradcam_parser.add_argument("--class-index", type=int, default=1, help="Prediction index to visualize.")
    gradcam_parser.add_argument("--alpha", type=float, default=0.6, help="Overlay blend factor.")
    gradcam_parser.add_argument("--save", type=Path, help="Optional path to save overlay image.")

    inference_parser = subparsers.add_parser("inference", help="Run local inference with heatmap output.")
    inference_parser.add_argument("--video", type=Path, help="Video path for testing.")
    inference_parser.add_argument("--every", type=int, default=15, help="Frame sampling interval.")
    inference_parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to inspect.")
    inference_parser.add_argument("--heatmap-frames", type=int, default=2, help="Number of heatmap frames.")
    inference_parser.add_argument("--save-dir", type=Path, help="Directory to save overlays and originals.")

    api_parser = subparsers.add_parser("api", help="Call the REST API and validate payload.")
    api_parser.add_argument("--url", default=DEFAULT_API_URL, help="API endpoint for analyze-video.")
    api_parser.add_argument("--report-url", default=DEFAULT_REPORT_URL, help="API endpoint for PDF generation.")
    api_parser.add_argument("--video", type=Path, help="Video path for upload.")
    api_parser.add_argument("--heatmap-frames", type=int, default=2, help="Frames requested from API.")
    api_parser.add_argument("--include-pdf", action="store_true", help="Trigger PDF generation test.")
    api_parser.add_argument("--save-pdf", type=Path, help="Where to store generated PDF.")

    colors_parser = subparsers.add_parser("colors", help="Inspect thermal color distribution via API output.")
    colors_parser.add_argument("--url", default=DEFAULT_API_URL, help="API endpoint for analyze-video.")
    colors_parser.add_argument("--video", type=Path, help="Video path for upload.")
    colors_parser.add_argument("--heatmap-frames", type=int, default=2, help="Frames requested from API.")

    mock_parser = subparsers.add_parser("mock-model", help="Create a lightweight mock video model.")
    mock_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "video_xception_mock.h5", help="Destination for mock model.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = COMMANDS[args.command]
    handler(args)


if __name__ == "__main__":
    main()
