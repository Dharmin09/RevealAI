# RevealAI Copilot Instructions

**RevealAI** is a deepfake detection system with dual video and audio analysis pipelines that produces PDF reports with visual evidence.

## Project Architecture

### Core Structure
- **`core/`** – ML inference and data processing engine
  - `infer.py` – Model loading and inference (video/audio)
  - `utils.py` – Computer vision utilities (Grad-CAM heatmaps, spectrograms, face detection via MediaPipe)
  - `config.py` – Centralized path management (auto-detects Colab vs local)
  - `generate_report.py` – PDF generation with heatmap overlays and spectrograms
  - `train_*.py` – Training scripts (video/audio models)

- **`app/`** – User-facing interface
  - `app_streamlit.py` – Main Streamlit web interface for file upload and inference
  - `download_ffpp.py` – FaceForensics++ dataset downloader

- **`Website/`** – Frontend (separate React/HTML project)

### Data Flow
1. User uploads video/audio via Streamlit (`app_streamlit.py`)
2. File temporarily saved, inference runs in `core/infer.py`
3. **Video path**: Frame extraction → Xception model (299×299 input) → Grad-CAM heatmaps (2D overlay + 3D plot)
4. **Audio path**: Audio → Mel-spectrogram → CNN model (224×224 input) → Raw spectrogram
5. Scores combined (60% video, 40% audio) → PDF report generated with evidence

## Key Patterns & Conventions

### Path Management (config.py)
- **Critical**: `config.py` auto-detects environment (`COLAB_GPU` env var vs local)
- Local paths: `N:\Datasets` root; Colab paths: `/content/drive/MyDrive/revealai`
- Always import paths from `config.py`, not hardcoded
- Data folders: `video/raw`, `audio/raw`, `video/frames`, `audio/specs`, `splits`
- Model paths: `models/video_xception.h5`, `models/audio_cnn.h5`

### Model Input Sizes
- **Video**: Xception expects (299, 299, 3) RGB
- **Audio**: CNN expects (224, 224, 3) RGB spectrogram image
- Preprocessing: Use `xception_preprocess()` for video, manual normalization (0-255 → 0-1) for audio

### Inference Pipeline (infer.py)
- `load_models()` uses module-level globals to cache loaded models (singleton pattern)
- `infer_video()` returns dict: `{"video_score": float, "heatmaps": [PIL Images]}`
- `infer_audio()` returns dict: `{"audio_score": float, "spec_img": np.array}`
- `combine_scores()` applies 60/40 weighting; handles None audio gracefully

### Heatmap Generation (utils.py)
- **Heatmaps**: `find_last_conv_layer()` → `make_gradcam_heatmap()` → `overlay_heatmap_on_image()`

### PDF Report Generation (generate_report.py)
- Use FPDF class; set font to "Arial" (built-in, no FileNotFoundError)
- Handle Windows temp file issues: create NamedTemporaryFile, immediately close, pass path to `Image.save()`
- Delete temp files in finally block to avoid permission errors
- Structure: cover page → summary (scores + verdict) → heatmaps page → spectrogram page

### Streamlit UI (app_streamlit.py)
- Single file upload with type filter `['mp4', 'wav']`
- Check `uploaded.type` (MIME) to branch logic: `"video/mp4"` vs `"audio/wav"`
- Use `st.checkbox()` for 3D heatmap toggle; fall back to 2D if unavailable
- Columns layout for multi-heatmap display: `cols = st.columns(len(heatmaps))`

## Common Tasks

### Adding a New Feature
1. If it's ML-related: add function to `utils.py` with proper input/output types (numpy arrays, PIL Images)
2. If it's inference logic: extend `infer.py` (don't modify model loading globals)
3. If it's UI: modify `app_streamlit.py` (use Streamlit widgets, not standard print/input)

### Debugging Inference
- Models are cached globally; restart Streamlit to reload (`CTRL+C` then rerun)
- Check `IMG_SIZE_VIDEO` and `IMG_SIZE_AUDIO` in `config.py` match model weights
- Verify temp directory cleanup in finally block of `infer_video()` to avoid disk bloat

### Handling Missing Dependencies
- Install via: `.venv\Scripts\Activate.ps1; pip install <package>`
- Core deps: tensorflow, opencv-python, librosa, streamlit, fpdf2, mediapipe, plotly
- Add to `requirements.txt` after testing

## File Naming Issues

**Fixed**: `core/__init__ .py` had a space; renamed to `core/__init__.py`  
**Convention**: All Python modules use standard names (no spaces in filenames)

## Integration Points

- **Streamlit → Core**: Pass file paths from temp upload, receive dict outputs
- **Core → TensorFlow**: Load `.h5` models; use `model.predict()` for inference
- **Utils → OpenCV/MediaPipe**: Frame extraction, face detection, heatmap overlay
- **Generate_report → FPDF**: Render images and text; always use temp files for Windows compatibility

---

**Last Updated**: Oct 2025  
For framework-specific questions (Streamlit, FPDF, TensorFlow API), prioritize docs over assumptions.
