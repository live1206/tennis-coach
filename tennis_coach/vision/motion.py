# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Zhang Xinyi <xinyi.zhang@outlook.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import cv2
import numpy as np

from tennis_coach.vision.court import make_polygon_mask


def _scale_roi(roi: list[list[int]], scale: float) -> list[list[int]]:
    return [[int(x * scale), int(y * scale)] for x, y in roi]


def _roi_foreground_ratio(fg_mask: np.ndarray, roi_mask: np.ndarray, roi_area: int) -> float:
    if roi_area == 0:
        return 0.0
    masked = cv2.bitwise_and(fg_mask, roi_mask)
    return float(cv2.countNonZero(masked)) / roi_area


def analyze_motion(
    video_path: str,
    segments: list[dict],
    rois: dict,
    target_height: int = 540,
    sample_interval: int = 4,
) -> list[dict]:
    if not segments:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for motion analysis: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale = target_height / original_height if original_height > target_height else 1.0
        target_width = int(original_width * scale) if scale < 1.0 else original_width
        frame_shape = (target_height, target_width) if scale < 1.0 else (original_height, original_width)
        near_roi = _scale_roi(rois["near"], scale) if scale < 1.0 else rois["near"]
        far_roi = _scale_roi(rois["far"], scale) if scale < 1.0 else rois["far"]
        near_mask = make_polygon_mask(frame_shape, near_roi)
        far_mask = make_polygon_mask(frame_shape, far_roi)
        near_area = int(cv2.countNonZero(near_mask))
        far_area = int(cv2.countNonZero(far_mask))
        kernel_size = 3 if scale < 0.75 else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        return [
            _analyze_segment(cap, fps, segment, scale, target_width, target_height, kernel, sample_interval, near_mask, near_area, far_mask, far_area)
            for segment in segments
        ]
    finally:
        cap.release()


def _analyze_segment(
    cap,
    fps: float,
    segment: dict,
    scale: float,
    target_width: int,
    target_height: int,
    kernel: np.ndarray,
    sample_interval: int,
    near_mask: np.ndarray,
    near_area: int,
    far_mask: np.ndarray,
    far_area: int,
) -> dict:
    warmup_start = max(0.0, float(segment["start"]) - 0.5)
    warmup_frames = int((float(segment["start"]) - warmup_start) * fps)
    start_frame = int(warmup_start * fps)
    end_frame = int(float(segment["end"]) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    mog = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=50, detectShadows=False)

    motion_values: list[float] = []
    near_values: list[float] = []
    far_values: list[float] = []
    frame_count = 0
    total_frames = end_frame - start_frame

    while frame_count < total_frames:
        read, frame = cap.read()
        if not read:
            break
        frame_count += 1

        if scale < 1.0:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        if frame_count <= warmup_frames or frame_count % sample_interval != 0:
            mog.apply(frame)
            continue

        mask = mog.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        near_ratio = _roi_foreground_ratio(mask, near_mask, near_area)
        far_ratio = _roi_foreground_ratio(mask, far_mask, far_area)
        near_values.append(near_ratio)
        far_values.append(far_ratio)
        motion_values.append(max(near_ratio, far_ratio))

    if not motion_values:
        return {
            "player_motion_max": 0.0,
            "player_motion_var": 0.0,
            "near_motion_mean": 0.0,
            "far_motion_mean": 0.0,
            "motion_sample_count": 0,
        }

    diffs = np.diff(motion_values)
    return {
        "player_motion_max": round(float(np.max(motion_values)), 6),
        "player_motion_var": round(float(np.var(diffs)), 8) if len(diffs) > 0 else 0.0,
        "near_motion_mean": round(float(np.mean(near_values)), 6),
        "far_motion_mean": round(float(np.mean(far_values)), 6),
        "motion_sample_count": len(motion_values),
    }
