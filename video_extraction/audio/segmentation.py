# Copyright (C) 2026 Zhang Xinyi <xinyi.zhang@outlook.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import numpy as np


def _trim_sparse_head(hit_times: np.ndarray, gap_multiplier: float = 3.0) -> int:
    if len(hit_times) < 4:
        return 0
    gaps = np.diff(hit_times)
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return 0
    trim_index = 0
    for index, gap in enumerate(gaps):
        if gap > median_gap * gap_multiplier:
            trim_index = index + 1
        else:
            break
    return trim_index


def _split_segment(
    hit_times: np.ndarray,
    start_index: int,
    end_index: int,
    maximum_duration: float,
    buffer: float,
    output: list[tuple[int, int]],
) -> None:
    duration = hit_times[end_index - 1] - hit_times[start_index] + 2 * buffer
    if duration <= maximum_duration or end_index - start_index < 2:
        output.append((start_index, end_index))
        return
    gaps = np.diff(hit_times[start_index:end_index])
    split_index = start_index + int(np.argmax(gaps)) + 1
    _split_segment(
        hit_times,
        start_index,
        split_index,
        maximum_duration,
        buffer,
        output,
    )
    _split_segment(
        hit_times,
        split_index,
        end_index,
        maximum_duration,
        buffer,
        output,
    )


def segment_rallies(
    hit_times: np.ndarray,
    hit_energies: np.ndarray,
    silence_gap: float = 6.0,
    buffer: float = 1.5,
    minimum_duration: float = 4.0,
    maximum_duration: float = 25.0,
    minimum_hit_count: int = 4,
    total_duration: float | None = None,
) -> list[dict]:
    if len(hit_times) == 0:
        return []

    coarse_segments = []
    start_index = 0
    for index in range(1, len(hit_times)):
        if hit_times[index] - hit_times[index - 1] >= silence_gap:
            coarse_segments.append((start_index, index))
            start_index = index
    coarse_segments.append((start_index, len(hit_times)))

    refined = []
    for start_index, end_index in coarse_segments:
        _split_segment(
            hit_times,
            start_index,
            end_index,
            maximum_duration,
            buffer,
            refined,
        )

    rallies = []
    for start_index, end_index in refined:
        start_index += _trim_sparse_head(hit_times[start_index:end_index])
        if end_index - start_index < minimum_hit_count:
            continue
        start = max(0.0, float(hit_times[start_index] - buffer))
        end = float(hit_times[end_index - 1] + buffer)
        if total_duration is not None:
            end = min(end, total_duration)
        if end - start < minimum_duration:
            continue
        rallies.append({
            "start": round(start, 6),
            "end": round(end, 6),
            "hit_times": hit_times[start_index:end_index].tolist(),
            "hit_energies": hit_energies[start_index:end_index].tolist(),
        })
    return rallies
