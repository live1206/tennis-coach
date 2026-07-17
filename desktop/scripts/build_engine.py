"""
Build the Python analysis engine into a standalone executable using PyInstaller.
Run from the project root: python desktop/scripts/build_engine.py
"""
import importlib.util
import shutil
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIST_DIR = PROJECT_ROOT / "dist-engine"
MODELS_DIR = PROJECT_ROOT / "video_extraction" / "vision" / "models"


def require_module(candidates: tuple[str, ...], dependency: str) -> str:
    for candidate in candidates:
        if importlib.util.find_spec(candidate) is not None:
            return candidate
    names = ", ".join(candidates)
    raise SystemExit(
        f"Missing {dependency} packaging dependency ({names}). "
        "Install the project optional dependencies before building."
    )


def collect_all_args(module: str) -> list[str]:
    return ["--collect-all", module]


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    native_bin = Path(sys.prefix) / "Library" / "bin"
    if native_bin.is_dir():
        env["PATH"] = str(native_bin) + os.pathsep + env.get("PATH", "")
    return env


def native_runtime_args() -> list[str]:
    native_bin = Path(sys.prefix) / "Library" / "bin"
    if not native_bin.is_dir():
        return []
    return ["--add-binary", str(native_bin / "*.dll") + os.pathsep + "."]


def stage_models() -> None:
    target = DIST_DIR / "video_extraction" / "vision" / "models"
    shutil.copytree(MODELS_DIR, target, dirs_exist_ok=True)


def build(
    entry_name: str,
    executable_name: str,
    collect_foundry: bool = False,
    collect_pose: bool = False,
):
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
        "--collect-all", "librosa",
        "--collect-all", "soundfile",
        str(entry),
    ]
    optional_args = native_runtime_args()
    if collect_pose:
        optional_args.extend(
            collect_all_args(require_module(("mediapipe",), "pose"))
        )
    if collect_foundry:
        for candidates in (
            ("foundry_local_sdk",),
            ("foundry_local_core_winml", "foundry_local_core"),
            ("onnxruntime_core", "onnxruntime"),
            ("onnxruntime_genai_core", "onnxruntime_genai"),
        ):
            optional_args.extend(
                collect_all_args(require_module(candidates, "foundry-local"))
            )
    cmd[3:3] = optional_args

    print(f"Running PyInstaller...")
    print(f"  Entry: {entry}")
    print(f"  Output: {DIST_DIR}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=build_environment(),
    )

    if result.returncode != 0:
        print("PyInstaller build failed!")
        sys.exit(1)

    print(f"\nExecutable built successfully at: {DIST_DIR / executable_name}")


def main():
    build("TennisCoachAnalysis.py", "TennisCoachAnalysis", collect_pose=True)
    build("TennisCoachLocalAnalysis.py", "TennisCoachLocalAnalysis", collect_foundry=True)
    stage_models()
    print("Next: place ffmpeg.exe in dist-engine/ffmpeg/ then run electron-builder")


if __name__ == "__main__":
    main()
