import sys
import os
try:
    from pkg_resources import get_distribution, DistributionNotFound
    protobuf_version = get_distribution('protobuf').version
    if not protobuf_version.startswith('3.20.'):
        # This is a fatal error. The program cannot continue.
        print("=" * 80, file=sys.stderr)
        print("[FATAL ENVIRONMENT ERROR]", file=sys.stderr)
        print("Your Python environment is not set up correctly.", file=sys.stderr)
        print("\nREASON:", file=sys.stderr)
        print("  The 'protobuf' library version is incorrect. Mediapipe, which is used for", file=sys.stderr)
        print("  face detection, requires version 3.20.x of this library.", file=sys.stderr)
        print(f"  You currently have version {protobuf_version} installed.", file=sys.stderr)
        print("\nSOLUTION:", file=sys.stderr)
        print("  Please run the following command in your terminal to install the correct version:", file=sys.stderr)
        
        # Construct the platform-specific command to be helpful
        # Assumes the .venv is in the parent directory of this file's parent directory (i.e., repo root)
        utils_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(utils_dir, '..', '..'))
        python_executable = os.path.join(repo_root, '.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(repo_root, '.venv', 'bin', 'python')

        print(f"\n    {python_executable} -m pip install protobuf==3.20.3\n", file=sys.stderr)
        print("After running this command, please restart the server.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1) # Exit the program
except (DistributionNotFound, ImportError):
    # If pkg_resources or protobuf is not installed, it will crash soon anyway.
    # This is a less common case than the version mismatch.
    pass

# --- OpenCV DNN face detector fallback ---
def opencv_dnn_face_bbox(img_rgb, conf_threshold=0.05):
    """
    Detect face using OpenCV DNN. Returns (x, y, w, h) or None if not found.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'opencv_models')
    proto = os.path.join(model_dir, 'deploy.prototxt')
    model = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    if not (os.path.exists(proto) and os.path.exists(model)):
        print('[DNN] Face model files not found.')
        return None
    net = cv2.dnn.readNetFromCaffe(proto, model)
    h, w = img_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x2, y2 = box.astype(int)
            print(f"[DNN] Face detected: conf={confidence:.2f}, bbox=({x},{y},{x2-x},{y2-y})")
            return (max(0, x), max(0, y), min(w, x2)-max(0, x), min(h, y2)-max(0, y))
    print("[DNN] No face detected by OpenCV DNN.")
    return None
    return None
 
# src/core/utils.py
import os
import cv2
import io
import tempfile
import subprocess
import shutil
import numpy as np
import librosa
from PIL import Image
import tensorflow as tf

# Lazy load TensorFlow to avoid blocking on import
_TF = None
_TF_LOADED = False
_FFMPEG_PATH = None

def _ensure_tf_loaded():
    global _TF, _TF_LOADED
    if not _TF_LOADED:
        try:
            from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
            _TF = (tf, xception_preprocess)
            _TF_LOADED = True
        except Exception as e:
            print(f"[WARNING] Could not import TensorFlow: {e}")
            _TF_LOADED = True  # Mark as loaded to avoid trying again

import matplotlib.cm as cm

# Optional imports - make them lazy to avoid breaking if not installed
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection if mp else None
    mp_face_mesh = mp.solutions.face_mesh if mp else None
    from .face_landmarks import (
        detect_facial_landmarks,
        create_facial_region_mask,
        get_facial_region_emphasis,
    )
    FACE_LANDMARKS_AVAILABLE = True
except ImportError:
    mp = None
    mp_face_detection = None
    mp_face_mesh = None
    FACE_LANDMARKS_AVAILABLE = False

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None

# --- All existing functions (extract_frames, load_images_to_array, etc.) remain unchanged ---
# (Keep all your previous functions here)

def get_video_metadata(video_path):
    """
    Extract metadata from video file (duration, resolution, FPS, codec, bitrate)
    Returns dict with video properties or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Calculate duration in seconds
        duration = frame_count / fps if fps > 0 else 0
        
        # Get file size in MB
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Calculate bitrate in Mbps
        bitrate = (file_size_mb * 8) / duration if duration > 0 else 0
        
        # Convert fourcc to codec string
        codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        metadata = {
            'duration': round(duration, 2),
            'resolution': f'{width}x{height}',
            'frame_rate': round(fps, 2),
            'codec': codec_str.strip() or 'Unknown',
            'bitrate': round(bitrate, 2),
            'file_size_mb': round(file_size_mb, 2)
        }
        
        return metadata
    except Exception as e:
        print(f"[WARN] Could not extract video metadata: {e}")
        return None

def extract_frames(
    video_path,
    out_dir,
    every_n_frames=10,
    max_frames=50,
    resize=None,
    return_metadata=False,
    sampling_strategy='uniform'
):
    """
    Extract frames from a video at FULL RESOLUTION.

    When sampling_strategy is "uniform" (default) this will select up to
    ``max_frames`` frames evenly spaced across the entire clip so the
    downstream heatmaps represent different points in the video rather than
    the first few frames only. Preprocessing for the model happens later.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video directly: {video_path}")
        print("[*] Attempting fallback codec approach...")
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("[ERROR] Video codec not supported")
                return [] if return_metadata else 0
        except Exception:
            print("[ERROR] Could not open video with any codec")
            return [] if return_metadata else 0

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[*] Video: {frame_count} frames @ {fps:.1f} FPS, {width}x{height}")

        targets = None
        if sampling_strategy == 'uniform' and frame_count > 0:
            target_total = min(max_frames, frame_count)
            targets = np.linspace(0, frame_count - 1, num=target_total, dtype=int)
            targets = np.unique(targets)
            target_total = len(targets)
        else:
            target_total = max_frames
        if targets is not None and len(targets) == 0:
            targets = None
            target_total = max_frames

        frames_meta = []
        idx = 0
        next_target_idx = 0
        current_target = targets[next_target_idx] if targets is not None and len(targets) > 0 else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            capture = False
            if targets is not None:
                if current_target is None:
                    break
                if idx == current_target:
                    capture = True
                    next_target_idx += 1
                    current_target = targets[next_target_idx] if next_target_idx < len(targets) else None
            else:
                step = max(1, every_n_frames)
                capture = (idx % step) == 0

            if capture and frame is not None and frame.size > 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{len(frames_meta)}.jpg"
                    out_path = os.path.join(out_dir, fname)
                    Image.fromarray(frame_rgb).save(out_path, format="JPEG", quality=95)

                    timestamp = (idx / fps) if fps and fps > 0 else None
                    relative_position = (idx / frame_count) if frame_count > 0 else None
                    frames_meta.append({
                        'path': out_path,
                        'frame_number': int(idx),
                        'timestamp': float(timestamp) if timestamp is not None else None,
                        'relative_position': float(relative_position) if relative_position is not None else None,
                    })

                    if len(frames_meta) >= target_total:
                        break
                except Exception as frame_error:
                    print(f"[WARN] Could not save frame {idx}: {frame_error}")

            idx += 1

        print(f"[OK] Extracted {len(frames_meta)} frames from video at native resolution")
        return frames_meta if return_metadata else len(frames_meta)

    finally:
        cap.release()

def load_images_to_array(file_list, target_size=(299,299)):
    """
    Load images from files into a numpy array.
    
    Args:
        file_list: List of image file paths
        target_size: Tuple (width, height) to resize to. If None, keeps original size.
    
    Returns:
        numpy array of images
    """
    arr = []
    for f in file_list:
        img = Image.open(f).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size)
        arr.append(np.array(img))
    
    if len(arr) == 0:
        if target_size is None:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros((0, target_size[0], target_size[1], 3), dtype=np.float32)
    
    return np.array(arr, dtype=np.float32)

def preprocess_frames_for_xception(frames):
    """
    Preprocess frames for Xception model
    
    Args:
        frames: List of numpy arrays representing video frames
    
    Returns:
        Preprocessed numpy array
    """
def preprocess_frames_for_xception(frames):
    """
    Preprocess frame data for Xception model
    NOTE: This requires TensorFlow - if it's not available, returns raw frames
    
    Args:
        frames: List of numpy arrays representing video frames
    
    Returns:
        Preprocessed numpy array
    """
    _ensure_tf_loaded()
    if _TF is not None:
        _, xception_preprocess = _TF
        return np.array([xception_preprocess(frame) for frame in frames])
    else:
        # If TensorFlow isn't available, return normalized frames
        print("[WARNING] TensorFlow not available - using raw frame preprocessing")
        return frames.astype('float32') / 255.0

def _resolve_ffmpeg_path():
    """Return the ffmpeg executable path if available."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is not None:
        return _FFMPEG_PATH

    candidates = []
    env_candidate = os.environ.get('FFMPEG_PATH')
    if env_candidate:
        candidates.append(env_candidate)

    which_candidate = shutil.which('ffmpeg')
    if which_candidate:
        candidates.append(which_candidate)

    if os.name == 'nt':
        windows_defaults = [
            'C:/ffmpeg/bin/ffmpeg.exe',
            'C:/Program Files/ffmpeg/bin/ffmpeg.exe',
            'C:/Program Files/FFmpeg/bin/ffmpeg.exe',
            'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe',
            'C:/ProgramData/chocolatey/bin/ffmpeg.exe',
        ]
        candidates.extend(windows_defaults)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            _FFMPEG_PATH = candidate
            print(f"[FFMPEG] Using executable at {_FFMPEG_PATH}")
            return _FFMPEG_PATH

    _FFMPEG_PATH = None
    print("[FFMPEG] Executable not found; video conversion features will be disabled.")
    return None


def convert_video_to_mp4(video_path, output_path=None):
    """Convert any video file to MP4 (H.264/AAC) without quality loss when possible."""
    if not video_path or not os.path.exists(video_path):
        print(f"[VIDEO] Conversion skipped; source not found: {video_path}")
        return None

    ext = os.path.splitext(video_path)[1].lower()
    if ext == '.mp4':
        return video_path

    ffmpeg_path = _resolve_ffmpeg_path()
    if not ffmpeg_path:
        print("[VIDEO] Cannot convert video because ffmpeg is unavailable.")
        return None

    if output_path is None:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', prefix='revealai_conv_')
        output_path = tmp_file.name
        tmp_file.close()

    copy_cmd = [
        ffmpeg_path,
        '-y',
        '-i', video_path,
        '-c', 'copy',
        '-movflags', '+faststart',
        output_path,
    ]

    try:
        subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[VIDEO] Converted to MP4 via stream copy: {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        print("[VIDEO] Stream copy failed; re-encoding to MP4 (H.264/AAC).")

    encode_cmd = [
        ffmpeg_path,
        '-y',
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        output_path,
    ]

    try:
        subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[VIDEO] Re-encoded to MP4: {output_path}")
        return output_path
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Video conversion to MP4 failed: {exc}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None


def audio_to_melspectrogram(wav_path, sr=16000, n_mels=128, hop_length=512, n_fft=2048, duration=5.0):
    """Fast audio to mel-spectrogram conversion - optimized for speed"""
    try:
        y, sr = librosa.load(wav_path, sr=sr, duration=duration, mono=True)
        if y is None or len(y) == 0:
            return None
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    except Exception as e:
        print(f"[WARNING] Error converting to mel-spectrogram: {e}")
        return None


def extract_audio_from_video(video_path, target_sr=16000, channels=1, max_duration=10.0):
    """Extract the audio track from a video file into a temporary WAV.

    Returns the path to the temporary WAV file or None if extraction fails.
    Caller is responsible for deleting the returned file.
    """
    if not os.path.exists(video_path):
        print(f"[WARNING] Video path not found for audio extraction: {video_path}")
        return None

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="revealai_audio_")
    tmp_path = tmp_file.name
    tmp_file.close()

    def _cleanup_tmp(path):
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass

    # Try moviepy first (pure Python, no external binary requirement)
    try:
        from moviepy.editor import VideoFileClip

        try:
            clip = VideoFileClip(video_path)
            audio_clip = clip.audio
            if audio_clip is None:
                clip.close()
            else:
                if max_duration is not None:
                    end_time = max_duration
                    audio_clip = audio_clip.subclip(0, end_time)
                audio_clip.write_audiofile(
                    tmp_path,
                    fps=target_sr,
                    nbytes=2,
                    codec='pcm_s16le',
                    ffmpeg_params=["-ac", str(channels)],
                    verbose=False,
                    logger=None,
                )
                audio_clip.close()
                clip.close()
                return tmp_path
        except Exception as moviepy_error:
            print(f"[INFO] MoviePy audio extraction failed ({moviepy_error}); falling back to ffmpeg.")
    except ImportError:
        # MoviePy not installed; proceed to ffmpeg fallback
        pass

    # Fallback to ffmpeg if available
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-ac', str(channels),
        '-ar', str(target_sr),
    ]
    if max_duration is not None:
        ffmpeg_cmd.extend(['-t', str(max_duration)])
    ffmpeg_cmd.extend(['-y', tmp_path])

    try:
        # Use DEVNULL for stderr to suppress verbose output, especially for common cases
        # like videos without an audio stream, which ffmpeg treats as an error.
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return tmp_path
    except FileNotFoundError:
        print("[WARNING] ffmpeg executable not found. Install ffmpeg or add it to PATH for audio extraction.")
    except subprocess.CalledProcessError:
        # This error now likely indicates a more serious problem (e.g., corrupted file)
        # since common "errors" like missing audio streams are silenced.
        print(f"[WARNING] ffmpeg failed to extract audio. The file may be corrupt or in an unsupported format.")

    _cleanup_tmp(tmp_path)
    return None

def get_audio_metadata(audio_path):
    """Extract audio metadata (duration, sample rate, channels, bitrate, format)"""
    try:
        import wave
        import struct
        
        # Load audio info with librosa
        y, sr = librosa.load(audio_path, sr=None, duration=None)
        duration = len(y) / sr if sr > 0 else 0
        
        # Try to get channel count from wav file
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Calculate bitrate (bits per second)
                bitrate = (n_frames * sample_width * 8 * frame_rate) // int(duration) if duration > 0 else 0
                
                return {
                    'duration': duration,
                    'sample_rate': sr,
                    'channels': n_channels,
                    'bitrate': bitrate,
                    'format': 'WAV',
                    'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                }
        except:
            # Fallback if wav reading fails
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1,
                'bitrate': 0,
                'format': 'AUDIO',
                'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            }
    except Exception as e:
        print(f"[WARNING] Could not extract audio metadata: {e}")
        return {
            'duration': 0,
            'sample_rate': 0,
            'channels': 0,
            'bitrate': 0,
            'format': 'UNKNOWN',
            'file_size': 0
        }

def spec_to_rgb_image(spec, out_size=(224, 224), cmap_name='magma'):
    """Convert mel-spectrogram to RGB using the same colormap as training."""
    if spec is None:
        return None

    spec_norm = spec - np.min(spec)
    spec_norm = spec_norm / (np.max(spec_norm) + 1e-8)

    cmap = cm.get_cmap(cmap_name)
    rgba_img = cmap(spec_norm)
    rgb_img = (rgba_img[..., :3] * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb_img)
    if out_size is not None:
        pil_img = pil_img.resize(out_size, Image.BICUBIC)

    return np.array(pil_img)


def generate_training_style_spectrogram_image(wav_path, out_size=(224, 224), target_sr=None, max_duration=5.0):
    """Replicate training spectrogram rendering for consistent inference."""
    try:
        # [PASS] CRITICAL: Set matplotlib backend BEFORE importing pyplot
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server environments
        
        import matplotlib.pyplot as plt
        import librosa.display

        y, sr = librosa.load(wav_path, sr=target_sr, mono=True, duration=max_duration)
        if y is None or len(y) == 0:
            return None

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(3, 3))
        librosa.display.specshow(S_db, sr=sr, cmap='magma', ax=ax)
        ax.axis('off')
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert('RGB')
        if out_size is not None:
            img = img.resize(out_size, Image.BICUBIC)
        return np.array(img)
    except Exception as e:
        print(f"[WARNING] Failed to render training-style spectrogram: {e}")
        return None

def create_audio_heatmap_visualization(audio_path, sr=16000, n_mels=256, hop_length=512, n_fft=2048, duration=10.0, output_width=1600, output_height=900, score=None):
    """
    Create a professional audio heatmap visualization with axes (like spectrogram).
    Large, readable format with time and frequency labels.
    This is used for visual evidence in reports, NOT for model inference.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel frequency bins (256 = fine granularity)
        hop_length: Hop length for STFT
        n_fft: FFT size
        duration: Duration to analyze in seconds
        output_width: Output image width in pixels (default 1600)
        output_height: Output image height in pixels (default 900)
        score: Audio deepfake score (0-1) for verdict-based coloring
    
    Returns:
        numpy array with heatmap visualization (RGB, 0-255)
    """
    try:
        # [PASS] CRITICAL: Set matplotlib backend BEFORE importing pyplot
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server environments
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        if y is None or len(y) == 0:
            print(f"[WARNING] Could not load audio: {audio_path}")
            return None
        
        # Create mel-spectrogram with power_to_db for better visualization
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        
        # Convert power to dB scale (relative to max) - gives better contrast
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create matplotlib figure with professional styling
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        fig.patch.set_facecolor('white')
        
        # [PASS] VERDICT-BASED COLORMAP SELECTION
        # Determine verdict and colormap based on score
        if score is None:
            cmap_name = 'jet'  # Default: blue->red
            verdict_text = 'ANALYSIS'
            verdict_color = '#6B7280'  # Gray
        elif score > 0.7:
            cmap_name = 'hot'  # Red/hot colormap for DEEPFAKE
            verdict_text = 'LIKELY FAKE'
            verdict_color = '#DC2626'  # Red
        elif score > 0.4:
            cmap_name = 'twilight'  # Purple/cyan twilight for UNCERTAIN
            verdict_text = 'SUSPICIOUS'
            verdict_color = '#F59E0B'  # Amber
        else:
            cmap_name = 'cool'  # Blue colormap for AUTHENTIC
            verdict_text = 'LIKELY REAL'
            verdict_color = '#10B981'  # Green
        
        # Display spectrogram with verdict-based coloring
        img = ax.imshow(
            S_db,
            aspect='auto',
            origin='lower',
            cmap=cmap_name,
            interpolation='bilinear'
        )
        
        # Add colorbar to show dB scale
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Intensity (dB)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        # Calculate time and frequency labels
        time_axis = np.linspace(0, len(y) / sr, S.shape[1])
        freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
        
        # Set x-axis (time) labels
        time_ticks = np.linspace(0, S.shape[1], 11)  # 11 time labels
        time_labels = [f'{time_axis[int(i)]:.1f}s' if i < S.shape[1] else f'{time_axis[-1]:.1f}s' for i in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, fontsize=10)
        
        # Set y-axis (frequency) labels - show mel scale
        freq_ticks = np.linspace(0, n_mels-1, 11)  # 11 frequency labels
        freq_labels = [f'{freq_axis[int(i)]/1000:.1f}k' if i < n_mels else 'kHz' for i in freq_ticks]
        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_labels, fontsize=10)
        
        # Add axis labels with bold font
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency (kHz)', fontsize=14, fontweight='bold')
        
        # [PASS] VERDICT-BASED TITLE
        if score is not None:
            title_text = f'Audio Frequency Analysis - Mel-Scale Spectrogram Heatmap\n{verdict_text} (Score: {score:.1%})\nRed/Hot = High energy | Blue = Low energy'
        else:
            title_text = 'Audio Frequency Analysis - Mel-Scale Spectrogram Heatmap\nRed/Hot = High energy (potential AI artifacts) | Blue = Low energy (natural speech)'
        
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Improve layout
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array using savefig (more reliable)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Load PNG from buffer
            X = np.array(Image.open(buf).convert('RGB'))
        except Exception as e:
            print(f"[HEATMAP] savefig method failed: {e}, trying canvas method...")
            # Fallback to canvas method
            try:
                fig.canvas.draw()
                raw_data = fig.canvas.buffer_rgba()
                X = np.frombuffer(raw_data, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                X = X[:, :, :3]  # Drop alpha channel, keep RGB
            except Exception as e2:
                print(f"[HEATMAP] Canvas method also failed: {e2}")
                plt.close(fig)
                return None
        
        # Resize to target dimensions if needed
        if X.shape != (output_height, output_width, 3):
            X = cv2.resize(X, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
        
        # Close figure to free memory
        plt.close(fig)
        
        print(f"[HEATMAP] Professional spectrogram generated:")
        print(f"          Shape: {X.shape}, Dtype: {X.dtype}")
        print(f"          Color range: [{X.min()}, {X.max()}]")
        print(f"          Mean intensity: {X.mean():.1f}")
        
        return X.astype(np.uint8)
    
    except Exception as e:
        print(f"[WARNING] Error creating audio heatmap visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_last_conv_layer(model):
    """Locate the best convolutional layer for Grad-CAM - prefer layers with good spatial resolution."""
    if model is None:
        raise ValueError("Model instance is required to locate convolutional layers.")

    total_layers = len(model.layers)
    print(f"[GRAD-CAM] Searching for convolutional layer in model with {total_layers} layers...")

    # Log the tail of the model for quick debugging insight (last 15 layers)
    for index, layer in enumerate(reversed(model.layers[-15:]), start=1):
        layer_type = type(layer).__name__
        output_shape = getattr(layer, 'output_shape', 'unknown')
        print(
            f"[GRAD-CAM]   Layer {total_layers - index + 1}: "
            f"{layer.name} ({layer_type}) - shape: {output_shape}"
        )

    # Find all convolutional layers with 4D output
    conv_layers = []
    for layer in reversed(model.layers):
        layer_type = type(layer).__name__
        is_conv = any(
            conv_name in layer_type
            for conv_name in ['Conv2D', 'SeparableConv2D', 'DepthwiseConv2D', 'Conv']
        )
        output_shape = getattr(layer, 'output_shape', None)

        dims = None
        if output_shape is not None and output_shape != 'unknown':
            try:
                dims = len(output_shape)
            except TypeError:
                pass

        if dims is None:
            # Fall back to graph tensor metadata (handles custom layers / lazy shapes)
            try:
                tensor = getattr(layer, 'output', None)
                if tensor is not None and hasattr(tensor, 'shape'):
                    dims = len(tensor.shape)
                    output_shape = tuple(int(d) if d is not None else -1 for d in tensor.shape)
            except Exception:
                dims = None

        has_4d_output = dims == 4

        if is_conv and has_4d_output:
            # Get spatial dimensions
            spatial_size = 1
            if output_shape and len(output_shape) >= 3:
                h, w = output_shape[1], output_shape[2]
                if h is not None and w is not None:
                    spatial_size = h * w
            
            conv_layers.append({
                'name': layer.name,
                'type': layer_type,
                'output_shape': output_shape,
                'spatial_size': spatial_size
            })

    if not conv_layers:
        print("[GRAD-CAM] ❌ No convolutional layers found!")
        raise ValueError("Could not find any convolutional layers for Grad-CAM.")

    # Prefer layers with larger spatial resolution (better for visualization)
    # but not too large (avoid early layers that might be too low-level)
    conv_layers.sort(key=lambda x: x['spatial_size'], reverse=True)
    
    # Skip layers that are too large (early layers) or too small (deep layers)
    # Aim for spatial size between 100 and 1000 pixels
    best_layer = None
    for layer_info in conv_layers:
        size = layer_info['spatial_size']
        if 100 <= size <= 1000:
            best_layer = layer_info
            break
    
    # Fallback to the largest spatial layer if no ideal one found
    if best_layer is None and conv_layers:
        best_layer = conv_layers[0]

    if best_layer:
        print(
            f"[GRAD-CAM] ✅ Selected layer: {best_layer['name']} "
            f"({best_layer['type']}) - shape: {best_layer['output_shape']} "
            f"(spatial size: {best_layer['spatial_size']})"
        )
        return best_layer['name']

    # If no convolutional layer was located, surface clear diagnostics
    print("[GRAD-CAM] ❌ No suitable convolutional layer found! Model summary:")
    print(f"[GRAD-CAM]   Total layers: {total_layers}")
    print(f"[GRAD-CAM]   Tail layer types: {[type(l).__name__ for l in model.layers[-5:]]}")
    raise ValueError("Could not find a suitable convolutional layer for Grad-CAM.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for model explainability with resilient tensor handling."""
    if model is None or last_conv_layer_name is None:
        return np.zeros((10, 10), dtype=np.float32), 0.0

    try:
        import tensorflow as tf

        # Convert incoming data to a TensorFlow tensor we can safely reuse
        if not isinstance(img_array, (tf.Tensor, np.ndarray)):
            img_array = np.array(img_array, dtype=np.float32)

        if isinstance(img_array, np.ndarray):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

        if len(img_array.shape) == 3:
            img_array = tf.expand_dims(img_array, axis=0)

        # Record fallback spatial dims in case we need to bail out
        fallback_height = int(img_array.shape[1]) if img_array.shape[1] is not None else 10
        fallback_width = int(img_array.shape[2]) if img_array.shape[2] is not None else 10

        # Retrieve the last convolutional layer defined earlier
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except Exception:
            print(f"[GRAD-CAM] ⚠️ Layer '{last_conv_layer_name}' not found")
            return np.zeros((fallback_height, fallback_width), dtype=np.float32), 0.0

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        def _ensure_tensor(value):
            """Pick the first tensor-like object from nested lists/tuples."""
            if isinstance(value, (list, tuple)):
                for item in value:
                    tensor = _ensure_tensor(item)
                    if tensor is not None:
                        return tensor
                return None
            if hasattr(value, "shape"):
                return value
            return None

        with tf.GradientTape() as tape:
            outputs = grad_model(img_array, training=False)

            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                conv_outputs_raw, predictions_raw = outputs
            else:
                conv_outputs_raw, predictions_raw = outputs

            conv_outputs = _ensure_tensor(conv_outputs_raw)
            predictions = _ensure_tensor(predictions_raw)

            if conv_outputs is None or predictions is None:
                print("[GRAD-CAM] ⚠️ Could not resolve tensors from grad model outputs")
                return np.zeros((fallback_height, fallback_width), dtype=np.float32), 0.0

            predictions = tf.convert_to_tensor(predictions)

            if len(predictions.shape) == 1:
                predictions = tf.expand_dims(predictions, axis=0)

            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))

            num_classes = predictions.shape[1]
            pred_index = min(int(pred_index), int(num_classes) - 1)

            loss = predictions[:, pred_index]
            confidence = float(predictions[0, pred_index])


        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print(f"[GRAD-CAM] ⚠️ Gradients are None for pred_index={pred_index}")
            # Only fallback if absolutely necessary
            return np.zeros((fallback_height, fallback_width), dtype=np.float32), confidence

        grad_mean = tf.reduce_mean(tf.abs(grads))
        if grad_mean < 1e-6:
            # Gradients are effectively zero, which means the model is not producing a
            # meaningful activation map for this frame. Return a blank heatmap.
            return np.zeros((fallback_height, fallback_width), dtype=np.float32), confidence


        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        pooled_mean = tf.reduce_mean(tf.abs(pooled_grads))
        if pooled_mean < 1e-6:
            print(f"[GRAD-CAM] ⚠️ Pooled gradients are effectively zero (mean={pooled_mean:.2e}), returning blank heatmap")
            return np.zeros((fallback_height, fallback_width), dtype=np.float32), confidence


        conv_outputs_squeezed = conv_outputs[0]
        heatmap = tf.tensordot(conv_outputs_squeezed, pooled_grads, axes=1)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            heatmap = tf.pow(heatmap, 0.7)

        result = heatmap.numpy().astype(np.float32)
        print(
            f"[GRAD-CAM] ✅ Heatmap generated - shape: {result.shape}, "
            f"min: {result.min():.3f}, max: {result.max():.3f}, mean: {result.mean():.3f}, pred_idx: {pred_index}, confidence: {confidence:.3f}"
        )
        return result, confidence

    except Exception as e:
        print(f"[GRAD-CAM] ❌ Error: {type(e).__name__}: {str(e)[:100]}")
        return np.zeros((10, 10), dtype=np.float32), 0.0

def create_thermal_colormap(heatmap_normalized, colormap=cv2.COLORMAP_JET, gamma=0.8):
    """Convert a 0-1 heatmap into an RGB thermal map with cool-to-hot colors.
    
    Uses JET colormap for authentic thermal imaging look:
    - Blue/Cyan: Low activation (cool regions)
    - Green/Yellow: Medium activation 
    - Orange/Red: High activation (hot regions)
    """
    if heatmap_normalized is None or heatmap_normalized.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    heatmap_norm = np.clip(heatmap_normalized, 0.0, 1.0).astype(np.float32)
    
    # Apply gamma correction for better color distribution
    # Lower gamma (0.8) gives more vivid colors like thermal cameras
    if gamma != 1.0:
        heatmap_norm = np.power(heatmap_norm, gamma)

    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    
    # JET colormap gives the classic thermal imaging look
    # Blue -> Cyan -> Green -> Yellow -> Red
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"[COLORMAP] Created thermal map - shape: {heatmap_rgb.shape}, unique_colors: {len(np.unique(heatmap_rgb.flatten()))}")
    return heatmap_rgb


def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.6, face_bbox=None, cmap='thermal', return_layers=False, focus_on_facial_features=True):
    """Create focused thermal overlays that highlight manipulation-prone facial features."""
    # Force alpha and colormap for exact effect
    alpha = 1.0
    cmap = 'thermal'

    if img_rgb is None:
        return None if return_layers else None

    if heatmap is None or heatmap.size == 0:
        base = img_rgb.astype(np.uint8)
        return (np.zeros_like(base), base) if return_layers else base

    img_h, img_w = img_rgb.shape[:2]
    img_rgb_uint8 = img_rgb.astype(np.uint8)

    # Normalize heatmap to 0-1
    heatmap_array = heatmap.astype(np.float32)
    if heatmap_array.max() > 1.0:
        heatmap_array = heatmap_array / (heatmap_array.max() + 1e-8)
    if heatmap_array.ndim > 2:
        heatmap_array = heatmap_array[:, :, 0]
    heatmap_resized = cv2.resize(heatmap_array, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Apply face region mask if available

    mask = np.zeros_like(heatmap_resized, dtype=np.float32)
    if face_bbox is not None:
        x, y, w, h = face_bbox
        print(f"[HEATMAP] Using face bbox: x={x}, y={y}, w={w}, h={h}")
        # Tighter mask, less blur for precision
        cv2.rectangle(mask, (x, y), (x + w, y + h), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)
        heatmap_boosted = np.clip(heatmap_resized, 0.0, 1.0)
        # Lower threshold for highlight, sharper mask
        highlight = (heatmap_boosted > 0.15).astype(np.float32)
        base = np.full_like(heatmap_boosted, 0.10)
        masked_heatmap = np.where(mask > 0.10, np.where(highlight, heatmap_boosted, base), 0.0)
        final_heatmap = masked_heatmap
    else:
        # Fallback: use elliptical, soft mask in upper-center region
        cx, cy = img_w // 2, int(img_h * 0.28)
        ax, ay = int(img_w * 0.22), int(img_h * 0.22)
        y_grid, x_grid = np.ogrid[:img_h, :img_w]
        ellipse_mask = (((x_grid - cx) / ax) ** 2 + ((y_grid - cy) / ay) ** 2) <= 1
        mask[ellipse_mask] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=18, sigmaY=18)
        heatmap_boosted = np.clip(heatmap_resized, 0.0, 1.0)
        highlight = (heatmap_boosted > 0.18).astype(np.float32)
        base = np.full_like(heatmap_boosted, 0.10)
        masked_heatmap = np.where(mask > 0.08, np.where(highlight, heatmap_boosted, base), 0.0)
        final_heatmap = masked_heatmap
    # Always show a visible heatmap, even if gradients are zero
    if np.all(final_heatmap == 0):
        final_heatmap = mask * 0.18  # faint fallback color

    # Direct JET colormap
    heatmap_uint8 = (final_heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay: alpha=1.0, so only heatmap is visible in colored regions
    overlay_float = (heatmap_colored.astype(np.float32) * (final_heatmap[..., None])) + (img_rgb_uint8.astype(np.float32) * (1.0 - final_heatmap[..., None]))
    overlay_visual = np.clip(overlay_float, 0, 255).astype(np.uint8)

    if return_layers:
        return heatmap_colored, overlay_visual
    return overlay_visual

def create_professional_comparison(original_frame, heatmap, frame_idx, alpha=0.3):
    """
    Create side-by-side comparison: Original + Heatmap Overlay
    Better for professional reports with lighter transparent heatmap
    """
    h, w = original_frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Create better heatmap overlay with lighter colors
    colormap = cm.get_cmap('jet')
    heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    
    # Apply transparency - lighter overlay so original image is visible
    heatmap_overlay = cv2.addWeighted(
        original_frame.astype(np.uint8),  # Original image
        1.0 - alpha,                      # Original visibility
        heatmap_colored,                  # Heatmap layer
        alpha,                            # Heatmap visibility (lighter)
        0                                  # No gamma
    )
    
    # Create side-by-side comparison
    combined = np.hstack([original_frame.astype(np.uint8), heatmap_overlay])
    
    # Add labels
    cv2.putText(combined, 'Original Frame', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Heatmap Overlay (Artifact Regions)', (w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, f'Frame {frame_idx}', (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined

def create_comparison_for_pdf(original_frame, heatmap, alpha=0.3):
    """
    Create professional comparison for PDF report
    Returns PIL Image that can be embedded in PDF
    """
    comparison = create_professional_comparison(original_frame, heatmap, 0, alpha)
    # Convert BGR to RGB for PIL
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
    return Image.fromarray(comparison_rgb)


def extract_diverse_face_frames(video_path, num_frames=2, min_face_confidence=0.2):
    """
    Extract frames from video with different face angles/poses for better heatmap analysis.
    Uses MediaPipe Face Detection to find frames with diverse face orientations.
    
    Args:
        video_path: Path to video file
        num_frames: Number of diverse frames to extract (default: 2)
        min_face_confidence: Minimum confidence for face detection (0-1)
    
    Returns:
        List of dicts with keys:
            - 'frame_number': Frame index in video
            - 'timestamp': Time in seconds
            - 'frame_rgb': RGB frame as numpy array (original resolution)
            - 'face_bbox': (x, y, width, height) of detected face
            - 'face_score': Detection confidence score
            - 'relative_position': Position in video (0-1)
    """
    if mp_face_detection is None:
        print("[WARN] MediaPipe not available, using OpenCV DNN fallback.")
        # Use OpenCV DNN to extract face frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames <= 0 or fps <= 0:
            cap.release()
            return []
        sample_interval = max(1, total_frames // 30)
        sampled_frames = []
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            bbox = opencv_dnn_face_bbox(frame_rgb, conf_threshold=0.2)
            if bbox:
                x, y, width, height = bbox
                face_area = width * height
                face_center_x = x + width / 2
                face_center_y = y + height / 2
                face_size_norm = face_area / (w * h)
                face_x_norm = face_center_x / w
                face_y_norm = face_center_y / h
                diversity_features = np.array([
                    face_x_norm,
                    face_y_norm,
                    face_size_norm,
                ])
                sampled_frames.append({
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps,
                    'frame_rgb': frame_rgb.copy(),
                    'face_bbox': (x, y, width, height),
                    'face_score': 1.0,
                    'relative_position': frame_idx / total_frames,
                    'diversity_features': diversity_features,
                    'face_area': face_area,
                })
            frame_idx += sample_interval
        cap.release()
        if len(sampled_frames) == 0:
            print("[WARN] No faces detected in video, using uniform frame extraction")
            return _extract_uniform_frames_fallback(video_path, num_frames)
        selected_frames = _select_diverse_frames(sampled_frames, num_frames)
        print(f"[FACE-DNN] Extracted {len(selected_frames)} frames with OpenCV DNN")
        return selected_frames
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or fps <= 0:
            cap.release()
            return []
        
        # Sample frames uniformly across video to find diverse faces
        sample_interval = max(1, total_frames // 30)  # Sample ~30 frames
        sampled_frames = []
        
        with mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model (0-5m distance)
            min_detection_confidence=min_face_confidence
        ) as face_detection:
            
            frame_idx = 0
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                
                # Detect faces
                results = face_detection.process(frame_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Get detection score
                        score = detection.score[0] if hasattr(detection, 'score') else 1.0
                        
                        # Calculate face metrics for diversity
                        # - Face size (larger faces = closer to camera)
                        # - Face position (different x/y positions = different angles)
                        face_area = width * height
                        face_center_x = x + width / 2
                        face_center_y = y + height / 2
                        
                        # Normalize metrics
                        face_size_norm = face_area / (w * h)
                        face_x_norm = face_center_x / w
                        face_y_norm = face_center_y / h
                        
                        # Calculate diversity score (combine position and size)
                        # We want frames with different face positions/angles
                        diversity_features = np.array([
                            face_x_norm,      # Horizontal position
                            face_y_norm,      # Vertical position
                            face_size_norm,   # Face size (distance from camera)
                        ])
                        
                        sampled_frames.append({
                            'frame_number': frame_idx,
                            'timestamp': frame_idx / fps,
                            'frame_rgb': frame_rgb.copy(),
                            'face_bbox': (x, y, width, height),
                            'face_score': float(score),
                            'relative_position': frame_idx / total_frames,
                            'diversity_features': diversity_features,
                            'face_area': face_area,
                        })
                        break  # Only use first face per frame
                
                frame_idx += sample_interval
        
        cap.release()
        
        if len(sampled_frames) == 0:
            print("[WARN] No faces detected in video, using uniform frame extraction")
            return _extract_uniform_frames_fallback(video_path, num_frames)
        
        # Select most diverse frames using k-means clustering in feature space
        selected_frames = _select_diverse_frames(sampled_frames, num_frames)
        
        print(f"[FACE] Extracted {len(selected_frames)} frames with diverse face angles")
        for i, frame_data in enumerate(selected_frames):
            print(f"       Frame {i+1}: #{frame_data['frame_number']} at {frame_data['timestamp']:.2f}s "
                  f"(confidence: {frame_data['face_score']:.2f})")
        
        return selected_frames
        
    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return _extract_uniform_frames_fallback(video_path, num_frames)


def _select_diverse_frames(sampled_frames, num_frames):
    """
    Select most diverse frames from sampled frames using feature-based diversity.
    Uses simple distance-based selection to find frames with different face angles.
    """
    if len(sampled_frames) <= num_frames:
        return sampled_frames
    
    # Extract diversity features
    features = np.array([f['diversity_features'] for f in sampled_frames])
    
    # Select frames by maximizing minimum pairwise distance
    selected_indices = []
    
    # Start with frame from middle of video (usually has good face angle)
    middle_idx = len(sampled_frames) // 2
    selected_indices.append(middle_idx)
    
    # Iteratively add frames that are most different from already selected frames
    while len(selected_indices) < num_frames:
        max_min_dist = -1
        best_idx = 0
        
        for i, feature in enumerate(features):
            if i in selected_indices:
                continue
            
            # Calculate minimum distance to any selected frame
            min_dist = min(
                np.linalg.norm(feature - features[j]) 
                for j in selected_indices
            )
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        
        selected_indices.append(best_idx)
    
    # Sort by frame number for temporal consistency
    selected_indices.sort()
    
    return [sampled_frames[i] for i in selected_indices]


def _extract_uniform_frames_fallback(video_path, num_frames):
    """
    Fallback: Extract frames uniformly spaced across video without face detection.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or fps <= 0:
            cap.release()
            return []
        
        # Select evenly spaced frames
        frame_indices = np.linspace(
            total_frames * 0.2,  # Start at 20% to skip intro
            total_frames * 0.8,  # End at 80% to skip outro
            num=num_frames,
            dtype=int
        )
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                
                frames.append({
                    'frame_number': int(frame_idx),
                    'timestamp': float(frame_idx / fps),
                    'frame_rgb': frame_rgb,
                    'face_bbox': None,
                    'face_score': 0.0,
                    'relative_position': float(frame_idx / total_frames),
                })
        
        cap.release()
        print(f"[FALLBACK] Extracted {len(frames)} uniform frames (no face detection)")
        return frames
        
    except Exception as e:
        print(f"[ERROR] Fallback frame extraction failed: {e}")
        return []
