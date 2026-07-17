# Copyright (C) 2026 Zhang Xinyi <xinyi.zhang@outlook.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from video_extraction.audio.ffmpeg import run_ffmpeg


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 22050,
) -> Path:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    output = Path(output_path)
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output),
    ])
    return output


def bandpass_filter(
    samples: np.ndarray,
    sample_rate: int,
    low: int = 200,
    high: int = 4000,
) -> np.ndarray:
    sos = butter(5, [low, high], btype="band", fs=sample_rate, output="sos")
    return sosfilt(sos, samples)


def detect_hits(
    audio_path: str | Path,
    sample_rate: int = 22050,
    bandpass_low: int = 200,
    bandpass_high: int = 4000,
    hop_length: int = 512,
    onset_threshold: float = 0.2,
    minimum_gap: float = 0.3,
    window_seconds: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    samples, sample_rate = librosa.load(audio_path, sr=sample_rate)
    filtered = bandpass_filter(
        samples,
        sample_rate,
        bandpass_low,
        bandpass_high,
    )

    window_samples = int(window_seconds * sample_rate)
    all_times = []
    all_energies = []
    for start in range(0, len(filtered), window_samples):
        chunk = filtered[start : start + window_samples]
        if len(chunk) < sample_rate:
            continue
        offset = start / sample_rate
        onset_envelope = librosa.onset.onset_strength(
            y=chunk,
            sr=sample_rate,
            hop_length=hop_length,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length,
            backtrack=False,
            delta=onset_threshold,
            units="frames",
        )
        if len(onset_frames) == 0:
            continue
        all_times.append(
            librosa.frames_to_time(
                onset_frames,
                sr=sample_rate,
                hop_length=hop_length,
            )
            + offset
        )
        all_energies.append(onset_envelope[onset_frames])

    if not all_times:
        return np.array([]), np.array([]), sample_rate

    onset_times = np.concatenate(all_times)
    onset_energies = np.concatenate(all_energies)
    order = np.argsort(onset_times)
    onset_times = onset_times[order]
    onset_energies = onset_energies[order]

    if len(onset_times) > 1:
        keep = [0]
        for index in range(1, len(onset_times)):
            if onset_times[index] - onset_times[keep[-1]] >= minimum_gap:
                keep.append(index)
            elif onset_energies[index] > onset_energies[keep[-1]]:
                keep[-1] = index
        onset_times = onset_times[keep]
        onset_energies = onset_energies[keep]

    return onset_times, onset_energies, sample_rate
