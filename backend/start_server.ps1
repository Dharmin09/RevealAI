#!/usr/bin/env powershell
# Quick start script for RevealAI Grad-CAM API Server (PowerShell version)

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host " RevealAI Grad-CAM API Server - Quick Start (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running from correct directory
if (-not (Test-Path "api_production_v2.py")) {
    Write-Host "ERROR: Please run this script from the backend directory:" -ForegroundColor Red
    Write-Host "  cd e:\Datasets\src\backend" -ForegroundColor Yellow
    Write-Host "  .\start_server.ps1" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path "..\venv\Scripts\Activate.ps1") {
    Write-Host "[1] Activating virtual environment..." -ForegroundColor Green
    & "..\venv\Scripts\Activate.ps1"
} else {
    Write-Host "[!] No virtual environment found at ..\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "    Continuing with system Python..." -ForegroundColor Yellow
}

# Check if models exist
Write-Host ""
Write-Host "[2] Checking for trained models..." -ForegroundColor Green
if (Test-Path "..\models\video_xception.h5") {
    Write-Host "    ✓ Video model found" -ForegroundColor Green
} else {
    Write-Host "    ⚠ Video model NOT found at ..\models\video_xception.h5" -ForegroundColor Yellow
    Write-Host "    Using mock model for testing" -ForegroundColor Yellow
}

if (Test-Path "..\models\audio_cnn.keras") {
    Write-Host "    ✓ Audio model found" -ForegroundColor Green
} else {
    Write-Host "    ⚠ Audio model NOT found (optional)" -ForegroundColor Yellow
}

# Start the server
Write-Host ""
Write-Host "[3] Starting Flask API server..." -ForegroundColor Green
Write-Host ""
Write-Host "    Server will be available at:" -ForegroundColor Cyan
Write-Host "    - http://localhost:5000 (local machine)" -ForegroundColor Cyan
Write-Host "    - http://127.0.0.1:5000 (this computer only)" -ForegroundColor Cyan
Write-Host ""
Write-Host "    Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

python api_production_v2.py

Write-Host ""
Read-Host "Press Enter to exit"
