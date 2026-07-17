from __future__ import annotations

import tempfile
from pathlib import Path

import cv2

from video_extraction.audio.hits import detect_hits, extract_audio
from video_extraction.audio.segmentation import segment_rallies


def _video_duration(video_path: str | Path) -> float | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return float(frame_count / fps)


def extract_rally_segments(
    video_path: str | Path,
    silence_gap: float = 6.0,
    buffer: float = 1.5,
    minimum_duration: float = 4.0,
    maximum_duration: float = 25.0,
    minimum_hit_count: int = 4,
) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="tennis-coach-audio-") as directory:
        audio_path = extract_audio(
            video_path,
            Path(directory) / "audio.wav",
        )
        hit_times, hit_energies, sample_rate = detect_hits(audio_path)

    rallies = segment_rallies(
        hit_times,
        hit_energies,
        silence_gap=silence_gap,
        buffer=buffer,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        minimum_hit_count=minimum_hit_count,
        total_duration=_video_duration(video_path),
    )
    return [
        {
            "index": index,
            "start": rally["start"],
            "end": rally["end"],
            "audio": {
                "sample_rate": sample_rate,
                "hit_times": rally["hit_times"],
                "hit_energies": rally["hit_energies"],
                "hit_count": len(rally["hit_times"]),
            },
        }
        for index, rally in enumerate(rallies, start=1)
    ]
