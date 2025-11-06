@echo off
REM ============================================================================
REM RevealAI Setup - Batch Version (Windows)
REM Simpler alternative when PowerShell scripts won't run
REM ============================================================================

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║           RevealAI Setup - Batch Version                  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if Python exists anywhere
for /f "delims=" %%A in ('where python 2^>nul') do set PYTHON_EXE=%%A
if not defined PYTHON_EXE (
    for /f "delims=" %%A in ('where python3 2^>nul') do set PYTHON_EXE=%%A
)

if not defined PYTHON_EXE (
    echo ERROR: Python not found in system PATH
    echo.
    echo How to fix:
    echo 1. Download Python 3.10 from https://www.python.org/downloads/
    echo 2. During install, CHECK "Add Python to PATH"
    echo 3. Restart PowerShell/Command Prompt
    echo 4. Run this script again
    echo.
    pause
    exit /b 1
)

echo Found Python: %PYTHON_EXE%
echo.

REM Remove old venv
if exist venv (
    echo Removing old venv...
    rmdir /s /q venv
)

REM Create new venv
echo Creating virtual environment...
%PYTHON_EXE% -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)

REM Activate and install
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
if exist requirements-compatible.txt (
    python -m pip install -r requirements-compatible.txt
) else (
    python -m pip install -r requirements.txt
)

echo.
echo ✓ Setup complete!
echo.
echo Next steps:
echo   1. Activate: venv\Scripts\activate.bat
echo   2. Test: python test_complete_setup.py
echo   3. Run: python api_production_v2.py
echo   4. Open: http://localhost:5000
echo.
pause
