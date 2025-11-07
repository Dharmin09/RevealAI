@echo off
REM Quick start script for RevealAI Grad-CAM API Server

echo.
echo ================================================================================
echo  RevealAI Grad-CAM API Server - Quick Start
echo ================================================================================
echo.

REM Check if running from correct directory
if not exist "api_production_v2.py" (
    echo ERROR: Please run this script from the backend directory:
    echo   cd e:\Datasets\src\backend
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "..\venv\Scripts\activate.bat" (
    echo [1] Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo [!] No virtual environment found at ..\venv\Scripts\activate.bat
    echo     Continuing with system Python...
)

REM Check if models exist
echo.
echo [2] Checking for trained models...
if exist "..\models\video_xception.h5" (
    echo     ✓ Video model found
) else (
    echo     ⚠ Video model NOT found at ..\models\video_xception.h5
    echo     Using mock model for testing
)

if exist "..\models\audio_cnn.keras" (
    echo     ✓ Audio model found
) else (
    echo     ⚠ Audio model NOT found (optional)
)

REM Start the server
echo.
echo [3] Starting Flask API server...
echo.
echo     Server will be available at:
echo     - http://localhost:5000 (local machine)
echo     - http://127.0.0.1:5000 (this computer only)
echo.
echo     Press CTRL+C to stop the server
echo.
echo ================================================================================
echo.

python api_production_v2.py

pause
