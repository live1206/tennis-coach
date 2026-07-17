from __future__ import annotations

import shutil
import subprocess


def run_ffmpeg(command: list[str]) -> None:
    if not shutil.which(command[0]):
        raise RuntimeError(
            f"'{command[0]}' not found. Install ffmpeg and ensure it is on PATH."
        )
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode(errors="replace").strip()
        command_preview = " ".join(command[:6])
        if len(command) > 6:
            command_preview += "..."
        raise RuntimeError(
            f"ffmpeg failed (exit {error.returncode}):\n"
            f"  Command: {command_preview}\n"
            f"  {stderr[-500:]}"
        ) from None
