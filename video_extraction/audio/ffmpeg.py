from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_ffmpeg_binary(command_name: str) -> str:
    if command_name != "ffmpeg":
        return command_name
    configured = os.environ.get("TENNIS_COACH_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if not configured:
        return command_name
    candidate = Path(configured).expanduser()
    return str(candidate) if candidate.exists() else configured


def _ffmpeg_install_hint() -> str:
    if sys.platform.startswith("linux"):
        return "Install ffmpeg (for Ubuntu/WSL: sudo apt update && sudo apt install -y ffmpeg)"
    if sys.platform == "darwin":
        return "Install ffmpeg (for macOS/Homebrew: brew install ffmpeg)"
    if sys.platform.startswith("win"):
        return "Install ffmpeg and add it to PATH, or set TENNIS_COACH_FFMPEG to ffmpeg.exe"
    return "Install ffmpeg and ensure it is on PATH"


def run_ffmpeg(command: list[str]) -> None:
    binary = _resolve_ffmpeg_binary(command[0])
    if not (shutil.which(binary) or Path(binary).exists()):
        raise RuntimeError(
            f"'{binary}' not found. {_ffmpeg_install_hint()}."
        )
    resolved_command = [binary, *command[1:]]
    try:
        subprocess.run(resolved_command, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode(errors="replace").strip()
        command_preview = " ".join(resolved_command[:6])
        if len(resolved_command) > 6:
            command_preview += "..."
        raise RuntimeError(
            f"ffmpeg failed (exit {error.returncode}):\n"
            f"  Command: {command_preview}\n"
            f"  {stderr[-500:]}"
        ) from None
