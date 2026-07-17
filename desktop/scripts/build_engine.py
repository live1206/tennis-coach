"""
Build the Python analysis engine into a standalone executable using PyInstaller.
Run from the project root: python desktop/scripts/build_engine.py
"""
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIST_DIR = PROJECT_ROOT / "dist-engine"


def build(entry_name: str, executable_name: str, collect_foundry: bool = False):
    entry = PROJECT_ROOT / entry_name
    if not entry.exists():
        print(f"Error: entry point not found: {entry}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", executable_name,
        "--distpath", str(DIST_DIR),
        "--workpath", str(PROJECT_ROOT / "build-engine"),
        "--specpath", str(PROJECT_ROOT / "build-engine"),
        "--hidden-import", "librosa",
        "--hidden-import", "soundfile",
        "--hidden-import", "sklearn",
        "--hidden-import", "sklearn.utils._cython_blas",
        "--hidden-import", "scipy.signal",
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "_socket",
        "--collect-submodules", "video_extraction",
        "--add-data",
        str(PROJECT_ROOT / "video_extraction" / "vision" / "models" / "yolox_nano.onnx")
        + os.pathsep
        + "video_extraction/vision/models",
        "--collect-all", "librosa",
        "--collect-all", "soundfile",
        str(entry),
    ]
    if collect_foundry:
        cmd[3:3] = [
            "--collect-submodules", "foundry_local_sdk",
            "--collect-all", "foundry_local_core",
        ]

    print(f"Running PyInstaller...")
    print(f"  Entry: {entry}")
    print(f"  Output: {DIST_DIR}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("PyInstaller build failed!")
        sys.exit(1)

    print(f"\nExecutable built successfully at: {DIST_DIR / executable_name}")


def main():
    build("TennisCoachAnalysis.py", "TennisCoachAnalysis")
    build("TennisCoachLocalAnalysis.py", "TennisCoachLocalAnalysis", collect_foundry=True)
    print("Next: place ffmpeg.exe in dist-engine/ffmpeg/ then run electron-builder")


if __name__ == "__main__":
    main()
