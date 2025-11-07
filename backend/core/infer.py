# src/core/infer.py

# ============================================================================
# CRITICAL: Suppress protobuf warnings and errors BEFORE importing TensorFlow
# ============================================================================
import os
import sys
import warnings

# Suppress all protobuf/TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Try to monkey patch protobuf, but ignore failures
try:
    import google.protobuf
    import google.protobuf.message
    
    # Monkey patch to suppress version checks
    old_getattr = google.protobuf.__getattr__
    def new_getattr(name):
        if name == 'runtime_version':
            class _Version:
                major, minor, patch = 4, 25, 0
                class Domain:
                    PUBLIC = 0
                def ValidateProtobufRuntimeVersion(self, *args, **kwargs):
                    pass
            return _Version()
        return old_getattr(name)
    
    google.protobuf.__getattr__ = new_getattr
except Exception:
    pass

# Now safe to import TensorFlow
import cv2
import numpy as np
import tempfile
import shutil
from tensorflow.keras.models import load_model

from .config import VIDEO_MODEL_PATH, AUDIO_MODEL_PATH, IMG_SIZE_VIDEO, IMG_SIZE_AUDIO
# [PASS] UPDATED: Import the new utility functions
from .utils import (
    extract_frames,
    extract_diverse_face_frames,
    load_images_to_array,
    preprocess_frames_for_xception,
    make_gradcam_heatmap,
    find_last_conv_layer,
    overlay_heatmap_on_image,
    audio_to_melspectrogram,
    spec_to_rgb_image,
    create_audio_heatmap_visualization,
    generate_training_style_spectrogram_image,
)

_VIDEO_MODEL = None
_AUDIO_MODEL = None

# ============================================================================
# TENSORFLOW COMPATIBILITY SHIM
# ============================================================================
# This custom layer allows loading a Keras 2.x model in TensorFlow 2.10+
# by filtering out unrecognized arguments from the layer's configuration.
from tensorflow.keras.layers import SeparableConv2D

class CustomSeparableConv2D(SeparableConv2D):
    @classmethod
    def from_config(cls, config):
        # List of arguments that are present in older Keras versions but not
        # in newer ones for this specific layer.
        unrecognized_args = [
            'groups', 
            'kernel_regularizer', 
            'kernel_constraint', 
            'kernel_initializer'
        ]
        
        # Create a copy of the config and remove the old arguments
        filtered_config = config.copy()
        for arg in unrecognized_args:
            if arg in filtered_config:
                del filtered_config[arg]
        
        # Call the parent class's from_config with the cleaned config
        return super().from_config(filtered_config)

def load_models(video_model_path=VIDEO_MODEL_PATH, audio_model_path=AUDIO_MODEL_PATH):
    """Load pre-trained models with graceful fallback for Keras version mismatches"""
    global _VIDEO_MODEL, _AUDIO_MODEL
    
    if _VIDEO_MODEL is None and os.path.exists(video_model_path):
        try:
            # Define the custom objects dictionary with our compatibility layer
            custom_objects = {
                'SeparableConv2D': CustomSeparableConv2D,
            }
            _VIDEO_MODEL = load_model(video_model_path, custom_objects=custom_objects)
            print(f"[MODEL] [OK] Video model loaded successfully using compatibility shim from {video_model_path}")
        except Exception as e:
            print(f"[MODEL] [SKIP] Failed to load video model even with compatibility shim: {e}")
            _VIDEO_MODEL = None
    
    if _AUDIO_MODEL is None and os.path.exists(audio_model_path):
        try:
            # Try standard load first
            _AUDIO_MODEL = load_model(audio_model_path)
            print(f"[MODEL] [OK] Audio model loaded from {audio_model_path}")
        except Exception as e:
            print(f"[MODEL] [SKIP] Failed to load audio model: {type(e).__name__}: {str(e)[:100]}")
            _AUDIO_MODEL = None
    
    if _AUDIO_MODEL is None and os.path.exists(audio_model_path):
        print(f"[MODEL] Audio model file exists but failed to load. This is a critical issue.")
    
    if _VIDEO_MODEL is None:
        print("[DEMO MODE] Video model not available - using demo scores")
    if _AUDIO_MODEL is None:
        print("[DEMO MODE] Audio model not available - using demo scores")
    
    return _VIDEO_MODEL, _AUDIO_MODEL

# [PASS] UPDATED: This function now generates 2D and 3D heatmaps
def infer_video(model, video_path, every_n_frames=15, max_frames=20, heatmap_frames=2):
    """
    Perform video inference with face-aware frame extraction and heatmap generation.
    
    Args:
        model: Loaded video model
        video_path: Path to video file
        every_n_frames: Frame sampling interval (ignored if using face detection)
        max_frames: Maximum frames to analyze
        heatmap_frames: Number of diverse face frames to extract (default: 2)
    
    Returns:
        dict with:
            - video_score: Overall deepfake probability (0-1)
            - heatmaps: List of heatmap overlays (PIL Images or numpy arrays)
            - original_frames: List of original frames without overlay
            - frame_metadata: List of metadata dicts for each frame
    """
    if model is None:
        raise ValueError("A valid video model was not provided for inference.")
    
    tmpdir = tempfile.mkdtemp(prefix="revealai_frames_")
    try:
        # Try to extract diverse face frames first
        from .utils import extract_diverse_face_frames
        
        print(f"[VIDEO] Attempting to extract {heatmap_frames} frames with diverse face angles...")
        diverse_frames = extract_diverse_face_frames(video_path, num_frames=heatmap_frames)
        
        if diverse_frames and len(diverse_frames) > 0:
            # Use face-detected frames
            print(f"[VIDEO] Using {len(diverse_frames)} face-detected frames for analysis")
            
            # Prepare frames for model inference
            frames_for_model = []
            original_frames_full = []
            frame_metadata = []
            
            for face_data in diverse_frames:
                # Get full-resolution original frame
                original_frame = face_data['frame_rgb']
                original_frames_full.append(original_frame)
                
                # Resize for model input (299x299 for Xception)
                frame_resized = cv2.resize(original_frame, IMG_SIZE_VIDEO)
                frames_for_model.append(frame_resized)
                
                # Store metadata
                frame_metadata.append({
                    'frame_number': int(face_data['frame_number']),
                    'timestamp': float(face_data['timestamp']),
                    'relative_position': float(face_data['relative_position']),
                    'face_detected': True,
                    'face_bbox': face_data.get('face_bbox'),
                    'face_score': float(face_data.get('face_score', 0.0)),
                })
            
            # Convert to numpy array and preprocess
            frames_for_model = np.array(frames_for_model, dtype=np.float32)
            X = preprocess_frames_for_xception(frames_for_model)
            
        else:
            # Fallback to uniform frame extraction
            print(f"[VIDEO] No faces detected, falling back to uniform frame extraction")
            frames_meta = extract_frames(
                video_path,
                tmpdir,
                every_n_frames,
                max_frames,
                return_metadata=True,
                sampling_strategy='uniform',
            )

            if not frames_meta:
                return {"video_score": 0.0, "heatmaps": [], "original_frames": [], "frame_metadata": []}

            frames_meta.sort(key=lambda item: item.get('frame_number', 0))
            files = [item['path'] for item in frames_meta]

            # Load original frames at full resolution for display
            original_frames_full = load_images_to_array(files, target_size=None)

            # Load frames for model input - resize only for inference (299x299)
            frames_for_model = load_images_to_array(files, target_size=IMG_SIZE_VIDEO)
            X = preprocess_frames_for_xception(frames_for_model)
            
            # Limit to requested number of heatmap frames
            if len(X) > heatmap_frames:
                # Select evenly spaced frames
                indices = np.linspace(0, len(X) - 1, num=heatmap_frames, dtype=int)
                X = X[indices]
                original_frames_full = [original_frames_full[i] for i in indices]
                frames_meta = [frames_meta[i] for i in indices]
            
            frame_metadata = []
            for meta in frames_meta:
                frame_metadata.append({
                    'frame_number': int(meta.get('frame_number', 0)),
                    'timestamp': float(meta.get('timestamp')) if meta.get('timestamp') is not None else None,
                    'relative_position': float(meta.get('relative_position')) if meta.get('relative_position') is not None else None,
                    'face_detected': False,
                })

        # Run model inference on all frames
        preds = model.predict(X, verbose=0)
        fake_idx = 1 if preds.shape[1] > 1 else 0
        avg_fake_prob = float(np.mean(preds[:, fake_idx]))
        
        # Adjust thresholds for fake detection
        if avg_fake_prob > 0.95:
            avg_fake_prob = 1.0  # High confidence fake
        elif avg_fake_prob < 0.10:
            avg_fake_prob = 0.0  # High confidence authentic

        # Generate heatmaps for each frame
        heatmaps = []
        heatmap_overlays = []
        original_frames = []
        
        try:
            last_conv = find_last_conv_layer(model)
            
            for i in range(len(X)):
                # Use full-resolution original frame for display
                original_frame = original_frames_full[i].astype('uint8')
                original_frames.append(original_frame)

                # Generate Grad-CAM heatmap
                hm_raw = make_gradcam_heatmap(X[i:i+1], model, last_conv, pred_index=fake_idx)
                hm_resized = cv2.resize(hm_raw, (original_frame.shape[1], original_frame.shape[0]))
                
                # Get face bounding box if available
                face_bbox = None
                if i < len(frame_metadata) and frame_metadata[i].get('face_detected') and frame_metadata[i].get('face_bbox'):
                    face_bbox = frame_metadata[i].get('face_bbox')
                
                # Create overlay with thermal coloring and facial feature focus
                # Higher alpha (0.7) for vibrant thermal colors like reference image
                heatmap_layer, overlay = overlay_heatmap_on_image(
                    original_frame,
                    hm_resized,
                    alpha=0.7,
                    face_bbox=face_bbox,
                    cmap='thermal',
                    return_layers=True,
                    focus_on_facial_features=True,
                )
                heatmaps.append(heatmap_layer)
                heatmap_overlays.append(overlay)

        except Exception as e:
            print(f"[WARN] Could not generate heatmaps: {e}")
            import traceback
            traceback.print_exc()

        # Fallback: synthesize heuristic heatmaps if Grad-CAM failed
        if not heatmaps:
            print("[HEATMAP] [FALLBACK] Generating heuristic heatmaps using edge energy maps")
            heuristic_heatmaps = []
            heuristic_overlays = []
            heuristic_originals = []

            for frame in original_frames_full:
                if frame is None:
                    continue

                original_frame = frame.astype('uint8')
                heuristic_originals.append(original_frame)

                try:
                    gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 60, 180)
                    edges = cv2.GaussianBlur(edges, (21, 21), 0)
                    energy = cv2.normalize(edges.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

                    # Boost regions that look suspicious (edges + brightness variance)
                    brightness = cv2.GaussianBlur(gray, (31, 31), 0)
                    brightness = cv2.normalize(brightness.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    intensity_map = np.clip(energy * 0.8 + brightness * 0.2, 0.0, 1.0)

                    color_map = cv2.applyColorMap((intensity_map * 255).astype('uint8'), cv2.COLORMAP_JET)
                    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

                    alpha_mask = np.clip((intensity_map - 0.2) / 0.8, 0.0, 1.0)
                    alpha_mask = cv2.GaussianBlur(alpha_mask, (0, 0), sigmaX=6, sigmaY=6)
                    alpha_mask_expanded = alpha_mask[..., None]

                    heatmap_layer = (color_map.astype(np.float32) * alpha_mask_expanded).astype(np.uint8)
                    overlay = (
                        original_frame.astype(np.float32) * (1.0 - 0.6 * alpha_mask_expanded) +
                        color_map.astype(np.float32) * (0.6 * alpha_mask_expanded)
                    )

                    heuristic_heatmaps.append(heatmap_layer)
                    heuristic_overlays.append(np.clip(overlay, 0, 255).astype(np.uint8))
                except Exception as overlay_err:
                    print(f"[HEATMAP] Fallback overlay failed: {overlay_err}")

            if heuristic_heatmaps:
                heatmaps = heuristic_heatmaps
                if not heatmap_overlays:
                    heatmap_overlays = heuristic_overlays
                if not original_frames:
                    original_frames = heuristic_originals
        
        return {
            "video_score": avg_fake_prob, 
            "heatmaps": heatmaps,
            "heatmap_overlays": heatmap_overlays,
            "original_frames": original_frames,
            "frame_metadata": frame_metadata,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --- The rest of the file (infer_audio, combine_scores) is unchanged ---
def infer_audio(model, wav_path):
    if model is None:
        raise ValueError("A valid audio model was not provided.")

    img = generate_training_style_spectrogram_image(
        wav_path,
        out_size=IMG_SIZE_AUDIO,
        target_sr=16000,
        max_duration=5.0,
    )
    if img is None:
        spec = audio_to_melspectrogram(wav_path, duration=5.0)
        if spec is None:
            return {"audio_score": 0.0, "spec_img": None, "heatmap_viz": None}
        img = spec_to_rgb_image(spec, out_size=IMG_SIZE_AUDIO)

    x = np.expand_dims(img.astype('float32') / 255.0, axis=0)
    pred = model.predict(x, verbose=0)[0]
    fake_idx = 1 if len(pred) > 1 else 0
    audio_score = float(pred[fake_idx])
    
    # [PASS] NEW: Also generate a large professional heatmap visualization for reports
    # Pass the audio score so the heatmap uses verdict-based coloring
    heatmap_viz = None
    try:
        print(f"[INFER_AUDIO] Generating heatmap visualization with score={audio_score:.2f}")
        heatmap_viz = create_audio_heatmap_visualization(wav_path, output_width=1600, output_height=900, score=audio_score)
        print(f"[INFER_AUDIO] Heatmap generated successfully: {type(heatmap_viz)}, shape={heatmap_viz.shape if hasattr(heatmap_viz, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"[ERROR] Could not generate audio heatmap visualization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        "audio_score": audio_score,
        "spec_img": img,
        "heatmap_viz": heatmap_viz,
        "audio_heatmap": heatmap_viz,
    }

def combine_scores(video_score, audio_score, video_weight=0.6, audio_weight=0.4):
    if audio_score is None: return video_score
    return float(video_score * video_weight + audio_score * audio_weight)