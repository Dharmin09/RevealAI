#!/usr/bin/env python3
"""Direct test of audio heatmap generation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import tempfile
import os

# Create a test audio file
print("Creating test audio file...")
sr = 16000
duration = 3
t = np.linspace(0, duration, int(sr * duration))
# Synthetic speech-like signal
y = 0.3 * (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t))

tmpfd, tmp_path = tempfile.mkstemp(suffix='.wav')
os.close(tmpfd)
sf.write(tmp_path, y, sr)

print(f"Test audio saved to: {tmp_path}")

# Test heatmap generation
try:
    from core.utils import create_audio_heatmap_visualization
    
    print("Generating audio heatmap visualization...")
    heatmap = create_audio_heatmap_visualization(tmp_path, score=0.45)
    
    if heatmap is not None:
        print(f"✅ SUCCESS! Heatmap generated:")
        print(f"   Type: {type(heatmap)}")
        print(f"   Shape: {heatmap.shape}")
        print(f"   Dtype: {heatmap.dtype}")
    else:
        print(f"❌ FAILED: heatmap is None")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.remove(tmp_path)
    print("\nTest file cleaned up")
