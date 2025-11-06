# src/core/config.py
# Centralized configuration - Environment-agnostic and auto-detecting

import os
import sys
from pathlib import Path
import json

# ============================================================================
# ENVIRONMENT DETECTION - Auto-detect on ANY machine
# ============================================================================

IS_COLAB = "COLAB_GPU" in os.environ

def detect_project_root():
    """
    Auto-detect project root on any machine by:
    1. Checking PROJECT_ROOT environment variable
    2. Looking for .venv folder (virtual environment marker)
    3. Looking for src/core folder structure
    4. Using current script location
    """
    # Check env variable first
    if 'PROJECT_ROOT' in os.environ:
        return Path(os.environ['PROJECT_ROOT'])
    
    # Start from this file's location
    current = Path(__file__).resolve().parent.parent.parent  # core/config.py -> root
    
    # Check if we're in right structure
    if (current / 'src' / 'core').exists():
        return current
    
    # Search up directory tree for markers
    for level in range(6):
        search_path = Path(__file__).resolve().parents[level]
        if (search_path / '.venv').exists() or (search_path / 'src' / 'core').exists():
            return search_path
    
    # Final fallback
    return current

if IS_COLAB:
    # Google Colab paths
    PROJECT_ROOT = Path('/content/drive/MyDrive/revealai')
    DATA_ROOT = PROJECT_ROOT / "datasets"
else:
    # Local machine - auto-detect
    PROJECT_ROOT = detect_project_root()
    DATA_ROOT = PROJECT_ROOT

print(f"[CONFIG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[CONFIG] DATA_ROOT: {DATA_ROOT}")

# ============================================================================
# CREATE MISSING DIRECTORIES
# ============================================================================

REQUIRED_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "reports",
    DATA_ROOT / "video" / "raw",
    DATA_ROOT / "audio" / "raw",
    DATA_ROOT / "video" / "frames",
    DATA_ROOT / "audio" / "specs",
    DATA_ROOT / "splits"
]

for dir_path in REQUIRED_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA DIRECTORIES
# ============================================================================

VIDEO_RAW = DATA_ROOT / "video" / "raw"
AUDIO_RAW = DATA_ROOT / "audio" / "raw"
VIDEO_FRAMES = DATA_ROOT / "video" / "frames"
AUDIO_SPECS = DATA_ROOT / "audio" / "specs"
SPLITS_DIR = DATA_ROOT / "splits"

# ============================================================================
# ASSET DIRECTORIES
# ============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL PATHS
# ============================================================================

VIDEO_MODEL_PATH = MODELS_DIR / "video_xception.h5"

# Audio model - Freshly trained on your spectrograms!
# Trained: 2000 samples (1000 fake + 1000 real from your dataset)
# Accuracy: 81% validation
# Format: Modern .keras format (TensorFlow 2.13 compatible)
# Ready for production use
AUDIO_MODEL_PATH = MODELS_DIR / "audio_cnn.keras"

print(f"[CONFIG] Video model: {VIDEO_MODEL_PATH}")
print(f"[CONFIG] Audio model: {AUDIO_MODEL_PATH}")

# ============================================================================
# MODEL CONFIGURATION - MUST MATCH TRAINING
# ============================================================================

IMG_SIZE_VIDEO = (299, 299)   # Xception input size (required!)
IMG_SIZE_AUDIO = (224, 224)   # Spectrogram image size for CNN

# ============================================================================
# HEATMAP & VISUALIZATION SETTINGS
# ============================================================================

HEATMAP_ALPHA = 0.35  # Transparency (0.0 = fully transparent, 1.0 = opaque)
HEATMAP_COLORMAP = 'jet'  # 'jet', 'hot', 'cool', 'RdYlBu'
HEATMAP_FRAMES_COUNT = 5  # Number of frame heatmaps to generate

# ============================================================================
# SYSTEM REQUIREMENTS CHECK
# ============================================================================

def check_requirements():
    """Check if system has all required files and paths"""
    issues = []
    
    # Check models
    if not VIDEO_MODEL_PATH.exists():
        issues.append(f"[WARNING] Video model not found: {VIDEO_MODEL_PATH}")
    if not AUDIO_MODEL_PATH.exists():
        issues.append(f"[WARNING] Audio model not found: {AUDIO_MODEL_PATH}")
    
    # Check directories are writable
    try:
        test_file = TEMP_DIR / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        issues.append(f"[ERROR] Cannot write to {TEMP_DIR}: {e}")
    
    return issues

# ============================================================================
# PRINT CONFIGURATION ON IMPORT
# ============================================================================

def print_config():
    """Print configuration for debugging"""
    print("\n" + "="*80)
    print("REVEALAI CONFIGURATION")
    print("="*80)
    print(f"Environment: {'Google Colab' if IS_COLAB else 'Local Machine'}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Models: {MODELS_DIR}")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Temp: {TEMP_DIR}")
    print("-"*80)
    
    # Check for issues
    issues = check_requirements()
    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("All paths verified successfully!")
    print("="*80 + "\n")

# Call on import in debug mode (comment out in production)
# print_config()