# RevealAI

RevealAI is a dual-stream deepfake detection system that analyzes both video and audio evidence, generates PDF reports, and exposes an API plus a static web UI.

---

## Repository Layout

```
backend/
	api_production_v2.py    # Flask API serving inference and static assets
	core/                   # Inference, training, utilities, reporting
	app/                    # Data tools (e.g., FF++ downloader)
frontend/
	web/src/                # HTML/CSS/JS frontend served by the API
ops/
	STARTUP.ps1             # Production launcher for Windows
	setup_revealai.ps1      # Environment bootstrap script
samples/                  # Demo media files (safe to extend)
models/                   # Drop trained weights here (ignored by Git)
reports/                  # Generated PDF reports (.gitkeep preserved)
temp/                     # Working directory for heatmaps/spectrograms
```

Key points:
- `backend/core/config.py` centralizes path detection; import paths from there instead of hardcoding.
- `models/` must contain `video_xception.h5` and `audio_cnn.keras` before running production inference.
- `.gitignore` keeps runtime artifacts out of version control while retaining empty directories via `.gitkeep`.

---

## Prerequisites

- Python 3.10 or 3.11 (64-bit, installed from python.org)
- Git
- Windows PowerShell (for the provided scripts)
- Optional: Docker Desktop if you plan to containerize

---

## Initial Setup

1. **Create and activate a virtual environment**
	 ```powershell
	 cd E:\Datasets\src
	 python -m venv .venv
	 .\.venv\Scripts\Activate.ps1
	 ```

2. **Install backend dependencies**
	 ```powershell
	 pip install --upgrade pip
	 pip install -r backend\requirements.txt
	 ```

3. **Place trained models** inside `models/`
	 ```text
	 models/
		 audio_cnn.keras
		 video_xception.h5
	 ```
	 Download both weights from Google Drive (`https://drive.google.com/drive/folders/1bUa6TtVIlHxjz6KxQ7nXIj0CeHLcoUAs?usp=drive_link`) and copy them into this folder—the repository keeps the directory empty by default.

4. **(Optional) Populate sample media** in `samples/` for quick testing.

---

## Running the API

### Fast path (recommended)
```powershell
cd E:\Datasets\src
ops\STARTUP.ps1
```
The script activates the virtual environment, performs dependency and model checks, then launches `backend/api_production_v2.py` on `http://localhost:5000`.

### Manual run
```powershell
cd E:\Datasets\src
.\.venv\Scripts\Activate.ps1
python backend\api_production_v2.py --host 0.0.0.0 --port 5000
```

Once running, the API serves:
- `/` and `/guide.html` etc. from `frontend/web/src`
- `/api/analyze-video` and `/api/analyze-audio` for inference
- `/api/learning/*` endpoints for the continuous learning workflow

---

## Frontend Notes

- Static files live under `frontend/web/src`; add new pages, scripts, or styles there.
- The Flask app dynamically resolves these paths via `FRONTEND_*` helpers, so no additional configuration is required after editing assets.
- If you prefer local development of the frontend alone, open the HTML files directly or host them with a simple HTTP server (they do not require a build step).

---

## Development Tips

- `backend/core/infer.py` exposes the main inference functions; reuse them in new scripts instead of duplicating logic.
- Temporary files (frames, heatmaps, spectrograms) default to `temp/` and are cleaned up by callers—ensure your additions follow the same pattern.
- Continuous learning artifacts are written under `backend/learning_data/`; the directory is ignored by Git to protect sensitive feedback samples.
- When adding new modules, keep imports relative to `backend/` to avoid path issues when the API resolves the repository root.

---

## Testing

- Use `samples/` or your own media to exercise `/api/analyze-video` and `/api/analyze-audio`.
- `python backend\debug_audio_heatmap.py` and `python backend\final_check.py` are useful for sanity checks while iterating on model outputs.
- For PDF verification, call `backend/core/generate_report.py` directly with dummy inputs to confirm layout changes before wiring them into the API.

---

## Troubleshooting

- **TensorFlow import errors**: confirm the virtual environment is active and reinstall with `pip install tensorflow==2.13.0` (matching the models' training version).
- **Port already in use**: `Get-NetTCPConnection -LocalPort 5000 | Stop-Process -Id {$_.OwningProcess}` releases the port on Windows PowerShell.
- **Static assets missing**: verify the requested file exists under `frontend/web/src` and the name matches the route in `api_production_v2.py`.
- **Model file not found**: check `backend/core/config.py` logs; it prints the resolved model directory on startup.

---

## Contact

For internal questions, start with the inline comments in `backend/api_production_v2.py` and `backend/core/config.py`. If issues persist, file a ticket in the repo with reproduction steps and relevant logs.

---

_Last updated: October 2025_
