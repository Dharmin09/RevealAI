import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and relay output."""
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result.returncode


def main():
    repo_root = Path(__file__).resolve().parent
    venv_path = repo_root / ".venv"
    scripts_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
    python_in_venv = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    pip_in_venv = scripts_dir / ("pip.exe" if os.name == "nt" else "pip")

    # Step 1: Create venv if missing
    if not venv_path.exists():
        print("[setup] Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])
    else:
        print("[setup] Virtual environment already exists.")

    # Step 2: Upgrade pip and install requirements
    print("[setup] Upgrading pip inside virtual environment...")
    run_command([str(python_in_venv), "-m", "pip", "install", "--upgrade", "pip"])

    requirements_file = repo_root / "backend" / "requirements.txt"
    if requirements_file.exists():
        print("[setup] Installing backend requirements...")
        run_command([str(pip_in_venv), "install", "-r", str(requirements_file)])
    else:
        print("[warning] backend/requirements.txt not found; skipping backend dependency install.")

    # Step 3: Node dependencies
    npm_path = shutil.which("npm")
    node_status = "complete"
    if npm_path:
        frontend_dir = repo_root / "frontend" / "web"
        if (frontend_dir / "package.json").exists():
            print("[setup] Installing frontend dependencies with npm...")
            run_command([npm_path, "install"], cwd=str(frontend_dir))
        else:
            print("[warning] frontend/web/package.json not found; skipping npm install.")
    else:
        node_status = "pending (install Node.js to enable npm)"
        print("[warning] npm executable not found. Install Node.js then rerun this script to install frontend dependencies.")

    # Step 4: Optional API start message
    api_cmd = f"{python_in_venv} backend\\api_production_v2.py"
    print("\nAll setup steps finished.")
    if node_status == "complete":
        print("[summary] Frontend dependencies: installed")
    else:
        print(f"[summary] Frontend dependencies: {node_status}")
    print("[summary] Backend environment ready. Launch API with:\n  "
          f"{api_cmd}")


if __name__ == "__main__":
    try:
        main()
        print("\nAll setup done.")
    except subprocess.CalledProcessError as err:
        print(f"\n[error] Command failed with exit code {err.returncode}: {' '.join(err.cmd)}")
        print("Setup halted. Resolve the error above and rerun the script.")
        sys.exit(err.returncode)
