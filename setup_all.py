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
    candidates: List[List[str]] = []

    if os.name == "nt":
        candidates.extend(( ["py", "-3.11"], ["py", "-3.10"] ))

    for exe_name in ("python3.11", "python3.10", "python3", "python"):
        exe_path = shutil.which(exe_name)
        if exe_path:
            candidates.append([exe_path])

    candidates.append([sys.executable])

    for cmd in candidates:
        version = _determine_python_version(cmd)
        if version and version[0] == 3 and version[1] in SUPPORTED_MINOR_VERSIONS:
            return cmd, version

    raise RuntimeError(
        "Could not find Python 3.10 or 3.11. Install one of those versions and rerun this script."
    )


def _ensure_venv(repo_root: Path) -> Path:
    """Locate or create a usable virtual environment under the repo root."""
    candidate_names = [".venv", "venv"]

    for name in candidate_names:
        path = repo_root / name
        cfg = path / "pyvenv.cfg"
        python_exe = path / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

        if cfg.exists() and python_exe.exists():
            version = _determine_python_version([str(python_exe)])
            if version and version[0] == 3 and version[1] in SUPPORTED_MINOR_VERSIONS:
                print(f"[setup] Using existing virtual environment '{name}' (Python {version[0]}.{version[1]}.{version[2]}).")
                return path

            if version:
                print(
                    f"[setup] Existing virtual environment '{name}' uses Python {version[0]}.{version[1]}.{version[2]} (unsupported). "
                    "It will be ignored."
                )
            else:
                print(f"[setup] Existing virtual environment '{name}' is unreadable. It will be ignored.")
        elif path.exists():
            print(f"[setup] Removing incomplete virtual environment folder '{name}'.")
            shutil.rmtree(path)

    target = repo_root / ".venv"
    if target.exists():
        print("[setup] Removing stale '.venv' before recreating...")
        shutil.rmtree(target)

    python_cmd, version = _find_compatible_python()
    version_text = f"{version[0]}.{version[1]}.{version[2]}"
    print(f"[setup] Creating virtual environment '.venv' with Python {version_text} using `{_format_python_cmd(python_cmd)}`...")
    run_command(python_cmd + ["-m", "venv", str(target)])
    return target


def main():
    repo_root = Path(__file__).resolve().parent
    venv_path = _ensure_venv(repo_root)
    scripts_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
    python_in_venv = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    pip_in_venv = scripts_dir / ("pip.exe" if os.name == "nt" else "pip")

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
    print("\nAll setup steps finished.")
    if node_status == "complete":
        print("[summary] Frontend dependencies: installed")
    else:
        print(f"[summary] Frontend dependencies: {node_status}")

    if os.name == "nt":
        activate_cmd = f".\\{venv_path.name}\\Scripts\\Activate.ps1"
        api_cmd = f".\\{venv_path.name}\\Scripts\\python.exe backend\\api_production_v2.py"
    else:
        activate_cmd = f"source {venv_path.name}/bin/activate"
        api_cmd = f"{venv_path.name}/bin/python backend/api_production_v2.py"

    print("[summary] Activate the environment with:\n  " + activate_cmd)
    print("[summary] Launch the API with:\n  " + api_cmd)


if __name__ == "__main__":
    try:
        main()
        print("\nAll setup done.")
    except subprocess.CalledProcessError as err:
        print(f"\n[error] Command failed with exit code {err.returncode}: {' '.join(err.cmd)}")
        print("Setup halted. Resolve the error above and rerun the script.")
        sys.exit(err.returncode)
    except RuntimeError as err:
        print(f"\n[error] {err}")
        sys.exit(1)
