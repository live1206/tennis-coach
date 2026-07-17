# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Zhang Xinyi <xinyi.zhang@outlook.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


CACHE_PATH = Path(tempfile.gettempdir()) / "tennis_coach_rois_cache.json"


def detect_court_corners(frame: np.ndarray) -> dict | None:
    """Auto-detect tennis court corners using white lines on a blue court surface."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    court_mask = cv2.inRange(hsv, (90, 40, 50), (130, 255, 255))
    court_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, court_kernel, iterations=3)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, court_kernel, iterations=2)

    dilated_court = cv2.dilate(court_mask, court_kernel, iterations=3)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    white_mask = cv2.bitwise_and(white_mask, dilated_court)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=30)
    if lines is None or len(lines) < 4:
        return None

    baselines = []
    sidelines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        mid_y = (y1 + y2) / 2
        mid_x = (x1 + x2) / 2
        if angle < 10 or angle > 170:
            baselines.append((x1, y1, x2, y2, length, angle, mid_y, mid_x))
        elif 20 <= angle <= 80 or 100 <= angle <= 160:
            sidelines.append((x1, y1, x2, y2, length, angle, mid_y, mid_x))

    if len(baselines) < 2 or len(sidelines) < 2:
        return None

    baseline_ys = sorted(set(int(line[6]) for line in baselines))
    if len(baseline_ys) < 2:
        return None
    y_gap = max(baseline_ys) - min(baseline_ys)
    if y_gap < 200:
        return None
    y_split = (min(baseline_ys) + max(baseline_ys)) / 2
    top_baselines = [line for line in baselines if line[6] < y_split]
    bottom_baselines = [line for line in baselines if line[6] >= y_split]
    if not top_baselines or not bottom_baselines:
        return None

    def cluster_baselines(candidates):
        sorted_by_y = sorted(candidates, key=lambda line: line[6])
        clusters = []
        for line in sorted_by_y:
            if clusters and abs(line[6] - clusters[-1][-1][6]) < 30:
                clusters[-1].append(line)
            else:
                clusters.append([line])
        return clusters

    bottom_clusters = cluster_baselines(bottom_baselines)
    bottom_clusters.reverse()
    best_total_bottom = max(sum(line[4] for line in cluster) for cluster in bottom_clusters)
    min_bottom_length = best_total_bottom * 0.15
    bottom_cluster = next(
        cluster for cluster in bottom_clusters if sum(line[4] for line in cluster) >= min_bottom_length
    )
    bottom_line = max(bottom_cluster, key=lambda line: line[4])

    top_clusters = cluster_baselines(top_baselines)
    dominant_top = max(top_clusters, key=lambda cluster: sum(line[4] for line in cluster))
    dominant_min_y = int(min(line[6] for line in dominant_top))
    search_top = max(0, dominant_min_y - 150)
    search_bottom = dominant_min_y - 10
    if search_bottom > search_top:
        projection = np.sum(white_mask > 0, axis=1).astype(float)
        projection_smooth = gaussian_filter1d(projection, sigma=3)
        sub_projection = projection_smooth[search_top:search_bottom]
        peaks, _ = find_peaks(sub_projection, height=80, prominence=15)
        if len(peaks) > 0:
            best_peak = peaks[np.argmin(np.abs(search_top + peaks - dominant_min_y))]
            far_baseline_y = search_top + best_peak
            top_line = (
                int(bottom_line[0]),
                far_baseline_y,
                int(bottom_line[2]),
                far_baseline_y,
                int(bottom_line[4]),
                0.0,
                float(far_baseline_y),
                float((bottom_line[0] + bottom_line[2]) / 2),
            )
        else:
            top_line = max(dominant_top, key=lambda line: line[4])
    else:
        top_line = max(dominant_top, key=lambda line: line[4])

    def line_params(line):
        x1, y1, x2, y2 = line[:4]
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        return a, b, c

    def intersect(line_1, line_2):
        a1, b1, c1 = line_params(line_1)
        a2, b2, c2 = line_params(line_2)
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return [int(round(x)), int(round(y))]

    def score_sideline(sideline):
        top_point = intersect(top_line, sideline)
        bottom_point = intersect(bottom_line, sideline)
        if top_point is None or bottom_point is None:
            return -1
        if abs(top_point[1] - top_line[6]) > 100 or abs(bottom_point[1] - bottom_line[6]) > 100:
            return -1
        if top_point[0] < -50 or top_point[0] > width + 50:
            return -1
        if bottom_point[0] < -50 or bottom_point[0] > width + 50:
            return -1
        return sideline[4]

    valid = [(score, sideline) for score, sideline in ((score_sideline(line), line) for line in sidelines) if score > 0]
    if len(valid) < 2:
        return None

    valid_with_bottom_x = []
    for score, sideline in valid:
        bottom_point = intersect(bottom_line, sideline)
        valid_with_bottom_x.append((score, sideline, bottom_point[0]))

    bottom_mid_x = (bottom_line[0] + bottom_line[2]) / 2
    left_valid = [(score, sideline) for score, sideline, bottom_x in valid_with_bottom_x if bottom_x < bottom_mid_x]
    right_valid = [(score, sideline) for score, sideline, bottom_x in valid_with_bottom_x if bottom_x >= bottom_mid_x]
    if not left_valid or not right_valid:
        return None

    left_line = min(left_valid, key=lambda item: intersect(bottom_line, item[1])[0])[1]
    right_line = max(right_valid, key=lambda item: intersect(bottom_line, item[1])[0])[1]

    top_left = intersect(top_line, left_line)
    top_right = intersect(top_line, right_line)
    bottom_left = intersect(bottom_line, left_line)
    bottom_right = intersect(bottom_line, right_line)
    if any(point is None for point in [top_left, top_right, bottom_left, bottom_right]):
        return None

    margin = -100
    for point in [top_left, top_right, bottom_left, bottom_right]:
        if point[0] < margin or point[0] > width - margin or point[1] < margin or point[1] > height - margin:
            return None

    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)
    area = 0.5 * abs(
        (points[0][0] * points[1][1] - points[1][0] * points[0][1])
        + (points[1][0] * points[2][1] - points[2][0] * points[1][1])
        + (points[2][0] * points[3][1] - points[3][0] * points[2][1])
        + (points[3][0] * points[0][1] - points[0][0] * points[3][1])
    )
    frame_area = height * width
    if area < 0.05 * frame_area or area > 0.60 * frame_area:
        return None

    return {"tl": top_left, "tr": top_right, "bl": bottom_left, "br": bottom_right}


def corners_to_rois(corners: dict) -> dict:
    top_left, top_right, bottom_left, bottom_right = (
        corners["tl"],
        corners["tr"],
        corners["bl"],
        corners["br"],
    )
    mid_left = [int((top_left[0] + bottom_left[0]) / 2), int((top_left[1] + bottom_left[1]) / 2)]
    mid_right = [int((top_right[0] + bottom_right[0]) / 2), int((top_right[1] + bottom_right[1]) / 2)]
    return {
        "near": [mid_left, mid_right, bottom_right, bottom_left],
        "far": [top_left, top_right, mid_right, mid_left],
        "format": "polygon",
    }


def normalize_rois(roi_data: dict) -> dict:
    if roi_data.get("format") == "polygon":
        return roi_data
    near = roi_data["near"]
    far = roi_data["far"]
    near_x, near_y, near_w, near_h = near
    far_x, far_y, far_w, far_h = far
    return {
        "near": [[near_x, near_y], [near_x + near_w, near_y], [near_x + near_w, near_y + near_h], [near_x, near_y + near_h]],
        "far": [[far_x, far_y], [far_x + far_w, far_y], [far_x + far_w, far_y + far_h], [far_x, far_y + far_h]],
        "format": "polygon",
    }


def select_rois(video_path: str, cache_path: Path = CACHE_PATH) -> dict | None:
    video_name = Path(video_path).name
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if video_name in cache:
            return normalize_rois(cache[video_name])

    cap = cv2.VideoCapture(video_path)
    read, frame = cap.read()
    cap.release()
    if not read:
        raise RuntimeError(f"Cannot read first frame from {video_path}")

    corners = detect_court_corners(frame)
    if corners is None:
        return None

    rois = corners_to_rois(corners)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache[video_name] = rois
    cache_path.write_text(json.dumps(cache, indent=2))
    return rois


def make_polygon_mask(shape: tuple[int, int] | tuple[int, int, int], polygon: list[list[int]]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask
