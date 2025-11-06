#!/usr/bin/env pwsh
# ============================================================================
# RevealAI Automated Setup Script
# Automatically configures Python environment and installs dependencies
# ============================================================================

$ErrorActionPreference = "Continue"

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           RevealAI Automated Setup Script                 ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

# STEP 1: Find Python
Write-Host "[STEP 1/5] Finding Python installation..." -ForegroundColor Yellow

$pythonPaths = @(
    "C:\Python310\python.exe",
    "C:\Python311\python.exe", 
    "C:\Python39\python.exe",
    (Get-Command python.exe -ErrorAction SilentlyContinue).Source,
    (Get-Command python -ErrorAction SilentlyContinue).Source
)

$pythonExe = $null
foreach ($path in $pythonPaths) {
    if ($path -and (Test-Path $path)) {
        $version = & $path --version 2>&1 3>$null
        if (($version -like "*3.9*") -or ($version -like "*3.10*") -or ($version -like "*3.11*") -or ($version -like "*3.12*")) {
            $pythonExe = $path
            Write-Host "  ✓ Found: $path ($version)" -ForegroundColor Green
            break
        }
    }
}

if (-not $pythonExe) {
    Write-Host "`n  ✗ ERROR: Python 3.9+ not found!" -ForegroundColor Red
    Write-Host "`n  How to fix:" -ForegroundColor Yellow
    Write-Host "  1. Download Python 3.10 from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  2. During install, CHECK 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host "  3. Run this script again" -ForegroundColor Yellow
    Write-Host "`n"
    exit 1
}

# ============================================================================
# STEP 2: Setup Virtual Environment
# ============================================================================

Write-Host "`n[STEP 2/5] Setting up virtual environment..." -ForegroundColor Yellow

$venvPath = ".\venv"
$venvPython = "$venvPath\Scripts\python.exe"
$venvPip = "$venvPath\Scripts\pip.exe"

# Remove old venv if it exists
if (Test-Path $venvPath) {
    Write-Host "  Removing old venv..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue | Out-Null
}

# Create new venv
Write-Host "  Creating venv..." -ForegroundColor Gray
& $pythonExe -m venv $venvPath | Out-Null

if (-not (Test-Path $venvPython)) {
    Write-Host "  ✗ Failed to create venv!" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ venv created at $venvPath" -ForegroundColor Green

# ============================================================================
# STEP 3: Install Dependencies
# ============================================================================

Write-Host "`n[STEP 3/5] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Gray

# Upgrade pip first
Write-Host "  - Upgrading pip..." -ForegroundColor Gray
& $venvPip install --upgrade pip setuptools wheel -q | Out-Null

# Install from compatible requirements
if (Test-Path ".\requirements-compatible.txt") {
    Write-Host "  - Installing from requirements-compatible.txt..." -ForegroundColor Gray
    & $venvPip install -r requirements-compatible.txt --no-cache-dir -q
} else {
    Write-Host "  - requirements-compatible.txt not found, using requirements.txt..." -ForegroundColor Gray
    & $venvPip install -r requirements.txt --no-cache-dir -q
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ⚠ Warning: Some packages may not have installed fully" -ForegroundColor Yellow
}

Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# ============================================================================
# STEP 4: Test TensorFlow
# ============================================================================

Write-Host "`n[STEP 4/5] Testing TensorFlow installation..." -ForegroundColor Yellow

$testScript = @"
import sys
import warnings
warnings.filterwarnings('ignore')

print("  Python:", sys.version.split()[0])

try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}", flush=True)
except Exception as e:
    print(f"  ✗ TensorFlow import failed: {e}", flush=True)
    sys.exit(1)

try:
    from google.protobuf import __version__ as pb_version
    print(f"  ✓ Protobuf {pb_version}", flush=True)
except Exception as e:
    print(f"  ✗ Protobuf failed: {e}", flush=True)

try:
    import numpy as np
    import cv2
    import flask
    print(f"  ✓ Core deps (numpy, cv2, flask)", flush=True)
except Exception as e:
    print(f"  ✗ Core deps failed: {e}", flush=True)
    sys.exit(1)

print("  ✓ All tests passed!", flush=True)
"@

$output = & $venvPython -c $testScript 2>&1
Write-Host $output

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ TensorFlow tests FAILED" -ForegroundColor Red
    Write-Host "`n  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Clear pip cache: $venvPip cache purge" -ForegroundColor Yellow
    Write-Host "  2. Reinstall: $venvPip install --no-cache-dir -r requirements-compatible.txt" -ForegroundColor Yellow
    Write-Host "  3. Check: $venvPython test_tensorflow.py" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# STEP 5: Summary & Options
# ============================================================================

Write-Host "`n[STEP 5/5] Setup Complete!" -ForegroundColor Green
Write-Host "`n"
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✓ RevealAI is Ready to Use!                  ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`nQuick Start:" -ForegroundColor Cyan
Write-Host ""

Write-Host "  Activate venv:" -ForegroundColor Yellow
Write-Host "    .\venv\Scripts\Activate.ps1`n" -ForegroundColor White

Write-Host "  Test everything:" -ForegroundColor Yellow
Write-Host "    python test_tensorflow.py`n" -ForegroundColor White

Write-Host "  Run production API:" -ForegroundColor Yellow
Write-Host "    python api_production_v2.py" -ForegroundColor White
Write-Host "    Then open: http://localhost:5000`n" -ForegroundColor Gray

Write-Host "  Or run demo API (if production fails):" -ForegroundColor Yellow
Write-Host "    python api_simple_working.py" -ForegroundColor White
Write-Host "    Then open: http://localhost:5000`n" -ForegroundColor Gray

# Optional: Ask to start API
Write-Host "Would you like to start the API now? (Y/N)" -ForegroundColor Cyan
$start = Read-Host

if ($start -eq "Y" -or $start -eq "y") {
    Write-Host "`nStarting API server..." -ForegroundColor Green
    Write-Host "Visit http://localhost:5000 in your browser`n" -ForegroundColor Gray
    & $venvPython api_production_v2.py
}

Write-Host "`n✓ Setup complete!`n" -ForegroundColor Green
