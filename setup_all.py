import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def run_command(cmd, cwd=None):
    """Run a shell command and relay output."""
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result.returncode


SUPPORTED_MINOR_VERSIONS = {10, 11}


def _determine_python_version(cmd: List[str]) -> Optional[Tuple[int, int, int]]:
    try:
        result = subprocess.run(
            cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    version_text = result.stdout.strip()
    try:
        major, minor, micro = (int(part) for part in version_text.split(".")[:3])
    except ValueError:
        return None
    return major, minor, micro


def _format_python_cmd(cmd: List[str]) -> str:
    return " ".join(cmd)


def _find_compatible_python() -> Tuple[List[str], Tuple[int, int, int]]:
    """Find a compatible Python 3.10 or 3.11 interpreter."""
    candidates = []
    if os.name == "nt":
        candidates.extend([["py", "-3.11"], ["py", "-3.10"]])
    else:
        candidates.extend([["python3.11"], ["python3.10"]])

    candidates.extend([["python3"], ["python"], [sys.executable]])

    for cmd in candidates:
        version = _determine_python_version(cmd)
        if version and version[0] == 3 and version[1] in SUPPORTED_MINOR_VERSIONS:
            print(f"[setup] Found compatible Python {version[0]}.{version[1]}.{version[2]} with command: {' '.join(cmd)}")
            return cmd, version

    # If no compatible Python is found, provide installation instructions.
    if os.name == "nt":
        install_cmd = "winget install Python.Python.3.11"
        message = f"Could not find Python 3.10 or 3.11. Please install it by running: \n{install_cmd}"
    else:
        message = "Could not find Python 3.10 or 3.11. Please install it using your system's package manager."
    
    raise RuntimeError(message)


def _ensure_venv(repo_root: Path) -> Path:
    """Ensure a virtual environment with a compatible Python version exists."""
    venv_name = ".venv"
    venv_path = repo_root / venv_name

    if venv_path.exists():
        python_exe = venv_path / ("Scripts" if os.name == "nt" else "bin") / "python"
        version = _determine_python_version([str(python_exe)])
        if version and version[0] == 3 and version[1] in SUPPORTED_MINOR_VERSIONS:
            print(f"[setup] Using existing compatible virtual environment '{venv_name}'.")
            return venv_path
        else:
            print(f"[setup] Existing virtual environment '{venv_name}' uses Python {version[0]}.{version[1]} (unsupported). It will be recreated.")
            shutil.rmtree(venv_path)

    # If we reach here, no compatible venv exists, so create one.
    python_cmd, version = _find_compatible_python()
    print(f"[setup] Creating new virtual environment '{venv_name}' with Python {version[0]}.{version[1]}.{version[2]}...")
    run_command(python_cmd + ["-m", "venv", str(venv_path)])
    return venv_path


def _ensure_winget():
    """Check if winget is installed and provide instructions if not."""
    if os.name == "nt":
        try:
            subprocess.run(["winget", "--version"], check=True, capture_output=True)
            print("[setup] winget is already installed.")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("[warning] winget is not installed. Please install it from the Microsoft Store: ms-windows-store://pdp/?id=9NBLGGH4NNS1")
            raise RuntimeError("winget is not installed.")

def main():
    if os.name == "nt":
        _ensure_winget()

    repo_root = Path(__file__).resolve().parent
    venv_path = _ensure_venv(repo_root)
    scripts_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
    python_in_venv = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    pip_in_venv = scripts_dir / ("pip.exe" if os.name == "nt" else "pip")


    # Step 2: Upgrade pip, setuptools, wheel for compatibility
    print("[setup] Upgrading pip, setuptools, wheel inside virtual environment...")
    run_command([str(python_in_venv), "-m", "pip", "install", "--upgrade", "--force-reinstall", "pip", "setuptools", "wheel"])

    # Step 2b: Check for system build tools (Windows)
    if os.name == "nt":
        import winreg
        def has_vs_build_tools():
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VS7")
                return True
            except OSError:
                return False
        if not has_vs_build_tools():
            print("[warning] Visual Studio Build Tools not detected. Some packages may fail to build native extensions.")
            print("[info] Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

    # Step 2c: Check for system build tools (Linux)
    if sys.platform.startswith("linux"):
        import shutil as _shutil
        if not _shutil.which("gcc"):
            print("[warning] GCC not found. Install build-essential: sudo apt-get install build-essential")


    # Step 2d: Install backend requirements with compatibility flags
    requirements_file = repo_root / "backend" / "requirements.txt"
    if requirements_file.exists():
        print("\n[setup] Installing backend requirements. Packages already installed will be skipped.")
        try:
            run_command([str(pip_in_venv), "install", "-r", str(requirements_file)])
        except subprocess.CalledProcessError:
            print("[warning] Pip install failed, retrying with --use-deprecated=legacy-resolver...")
            run_command([str(pip_in_venv), "install", "--use-deprecated=legacy-resolver", "-r", str(requirements_file)])
        # Extra: install system-level dependencies for ffmpeg, moviepy, etc.
        if os.name == "nt":
            print("[setup] Checking for ffmpeg (Windows)...")
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                print("[setup] Downloading ffmpeg for Windows...")
                print("[manual] Download and extract ffmpeg from https://www.gyan.dev/ffmpeg/builds/ and add to PATH.")
        else:
            print("[setup] Checking for ffmpeg (Linux/Mac)...")
            if not shutil.which("ffmpeg"):
                print("[manual] Install ffmpeg using your package manager, e.g. sudo apt-get install ffmpeg")
    else:
        print("[warning] backend/requirements.txt not found; skipping backend dependency install.")

    # Step 2e: Ensure all critical Python dependencies are present (redundant safety)
    print("\n[setup] Ensuring all critical Python dependencies are present...")
    run_command([str(pip_in_venv), "install", "protobuf==3.20.3", "numpy==1.26.0", "tensorflow==2.19.1", "opencv-python==4.8.1.78", "scipy==1.11.3", "scikit-learn==1.3.2", "matplotlib==3.8.1", "librosa==0.10.0", "soundfile==0.12.1", "Pillow==10.0.1", "plotly==5.17.0", "flask==2.3.3", "flask-cors==4.0.0", "requests==2.31.0", "fpdf2==2.7.0", "moviepy==1.0.3", "typing-extensions>=4.0.0"])
    if os.name == "nt":
        run_command([str(pip_in_venv), "install", "pywin32"])

    # Step 3: Node dependencies (frontend)
    npm_path = shutil.which("npm")
    node_status = "complete"
    frontend_dir = repo_root / "frontend" / "web"
    if npm_path:
        if (frontend_dir / "package.json").exists():
            print("[setup] Installing frontend dependencies with npm...")
            try:
                run_command([npm_path, "install"], cwd=str(frontend_dir))
            except subprocess.CalledProcessError:
                print("[warning] npm install failed. Try deleting node_modules and package-lock.json, then rerun.")
        else:
            print("[warning] frontend/web/package.json not found; skipping npm install.")
    else:
        node_status = "pending (install Node.js to enable npm)"
        print("[warning] npm executable not found. Install Node.js then rerun this script to install frontend dependencies.")

    # Step 3b: Ensure all frontend dependencies are present (redundant safety)
    if npm_path and (frontend_dir / "package.json").exists():
        print("[setup] Ensuring all frontend dependencies are present...")
        run_command([npm_path, "install", "@tailwindcss/cli@^4.1.14", "tailwindcss@^4.1.14"], cwd=str(frontend_dir))
        print("[setup] Checking for Node.js version compatibility...")
        try:
            result = subprocess.run([npm_path, "-v"], capture_output=True, text=True)
            node_version = result.stdout.strip()
            print(f"[setup] Detected npm version: {node_version}")
        except Exception:
            print("[warning] Could not determine npm version.")

    # Step 4: Final summary and troubleshooting tips
    print("\n" + "="*80)
    print("**" + " "*29 + "POST-SETUP COMMANDS" + " "*28 + "**")
    print("="*80)
    print("\n[summary] Frontend dependencies: installed" if node_status == "complete" else f"[summary] Frontend dependencies: {node_status}")

    if os.name == "nt":
        activate_cmd = f".\\{venv_path.name}\\Scripts\\Activate.ps1"
        api_cmd = f".\\{venv_path.name}\\Scripts\\python.exe backend\\api_production_v2.py"
    else:
        activate_cmd = f"source {venv_path.name}/bin/activate"
        api_cmd = f"{venv_path.name}/bin/python backend/api_production_v2.py"

    print("\n[summary] Activate the environment with:\n  " + activate_cmd)
    print("\n[summary] Launch the API with:\n  " + api_cmd)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
        print("\n✅ Setup completed successfully! The application is now runnable.")
    except subprocess.CalledProcessError as err:
        print(f"\n[error] Command failed with exit code {err.returncode}: {' '.join(err.cmd)}")
        print("Setup halted. Resolve the error above and rerun the script.")
        sys.exit(err.returncode)
    except RuntimeError as err:
        print(f"\n[error] {err}")
        sys.exit(1)
