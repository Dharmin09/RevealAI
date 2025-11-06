# RevealAI Production Startup Script

Write-Host "RevealAI Deepfake Detection System" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

# Check if venv exists
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Create it with: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Verify dependencies
Write-Host "Verifying dependencies..." -ForegroundColor Green
python -c "import tensorflow; print(\"TensorFlow version: \" + tensorflow.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: TensorFlow not available. Continuing anyway." -ForegroundColor Yellow
} else {
    Write-Host "TensorFlow import check passed." -ForegroundColor Green
}

# Check models
Write-Host "Verifying models..." -ForegroundColor Green
if (-not (Test-Path "N:\Datasets\models\video_xception.h5")) {
    Write-Host "Warning: Video model not found at N:\Datasets\models\video_xception.h5" -ForegroundColor Yellow
} else {
    Write-Host "   Video model: OK" -ForegroundColor Green
}
if (-not (Test-Path "N:\Datasets\models\audio_cnn.keras")) {
    Write-Host "Warning: Audio model not found at N:\Datasets\models\audio_cnn.keras" -ForegroundColor Yellow
} else {
    Write-Host "   Audio model: OK" -ForegroundColor Green
}

# Start API
Write-Host "" -ForegroundColor Green
Write-Host "Starting RevealAI API Server..." -ForegroundColor Green
Write-Host "Open browser: http://localhost:5000" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

python api_production_v2.py
