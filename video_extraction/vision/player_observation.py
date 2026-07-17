# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Zhang Xinyi <xinyi.zhang@outlook.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from video_extraction.vision.onnx_inference import OnnxInference


MODEL_NAME = "yolox_nano.onnx"
MODEL_INPUT_SIZE = 416
PLAYER_IDS = ("player_1", "player_2")
COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32


def default_model_path() -> Path:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    bundled = bundle_root / "video_extraction" / "vision" / "models" / MODEL_NAME
    if bundled.exists():
        return bundled
    return Path(__file__).resolve().parent / "models" / MODEL_NAME


class YoloXDetector:
    """Run one COCO class from an Apache-2.0 YOLOX-Nano ONNX model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_id: int = COCO_PERSON_CLASS_ID,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        inference_backend: str = "opencv",
    ):
        self.model_path = Path(model_path) if model_path else default_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLOX person detector not found: {self.model_path}. "
                "Pass --model-path or add the model under video_extraction/vision/models/."
            )
        self.inference = OnnxInference(self.model_path, inference_backend)
        self.class_id = class_id
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._grid, self._expanded_strides = self._build_decode_grid()

    @staticmethod
    def _build_decode_grid():
        grids = []
        strides = []
        for stride in (8, 16, 32):
            height = MODEL_INPUT_SIZE // stride
            width = MODEL_INPUT_SIZE // stride
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            grids.append(np.stack((grid_x, grid_y), axis=2).reshape(-1, 2))
            strides.append(np.full((height * width, 1), stride))
        return np.concatenate(grids), np.concatenate(strides)

    @staticmethod
    def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        scale = min(MODEL_INPUT_SIZE / width, MODEL_INPUT_SIZE / height)
        resized_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(frame, resized_size, interpolation=cv2.INTER_LINEAR)
        padded = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[: resized_size[1], : resized_size[0]] = resized

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            swapRB=True,
        )
        return blob, scale

    def detect(self, frame: np.ndarray) -> list[dict]:
        height, width = frame.shape[:2]
        blob, scale = self._preprocess(frame)
        output = self.inference.forward(blob).squeeze(0)
        if output.ndim != 2 or output.shape[0] != len(self._grid):
            raise RuntimeError(f"Unexpected YOLOX output shape: {output.shape}")
        decoded = output.copy()
        decoded[:, :2] = (decoded[:, :2] + self._grid) * self._expanded_strides
        decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * self._expanded_strides
        scores = self._class_scores(decoded)
        candidate_indices = np.flatnonzero(scores >= self.confidence_threshold)

        boxes = []
        confidences = []
        for index in candidate_indices:
            center_x, center_y, box_width, box_height = decoded[index, :4] / scale
            x1 = max(0, int(center_x - box_width / 2))
            y1 = max(0, int(center_y - box_height / 2))
            x2 = min(width, int(center_x + box_width / 2))
            y2 = min(height, int(center_y + box_height / 2))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(scores[index]))

        if not boxes:
            return []

        kept = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        return [
            {
                "bbox": [
                    boxes[int(index)][0],
                    boxes[int(index)][1],
                    boxes[int(index)][0] + boxes[int(index)][2],
                    boxes[int(index)][1] + boxes[int(index)][3],
                ],
                "confidence": round(confidences[int(index)], 6),
            }
            for index in np.asarray(kept).reshape(-1)
        ]

    def _class_scores(self, decoded: np.ndarray) -> np.ndarray:
        class_score_index = 5 + self.class_id
        if decoded.shape[1] <= class_score_index:
            raise RuntimeError(
                f"YOLOX output does not contain COCO class {self.class_id}: {decoded.shape}"
            )
        return decoded[:, 4] * decoded[:, class_score_index]


class YoloXPersonDetector(YoloXDetector):
    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        inference_backend: str = "opencv",
    ):
        super().__init__(
            model_path,
            class_id=COCO_PERSON_CLASS_ID,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            inference_backend=inference_backend,
        )


def _normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    return cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1).flatten()


def _appearance_descriptor(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    upper_body = frame[
        y1 + int(box_height * 0.08) : y1 + int(box_height * 0.65),
        x1 + int(box_width * 0.15) : x2 - int(box_width * 0.15),
    ]
    if upper_body.size == 0:
        return None
    hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return _normalize_histogram(histogram)


def appearance_descriptor(
    frame: np.ndarray,
    bbox: list[int] | None = None,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    return _appearance_descriptor(
        frame,
        bbox if bbox is not None else [0, 0, width, height],
    )


def court_side(bbox: list[int], rois: dict, frame_height: int) -> str | None:
    x1, _y1, x2, y2 = bbox
    foot = ((x1 + x2) / 2.0, float(y2))
    margin = frame_height * 0.08
    distances = {
        side: cv2.pointPolygonTest(np.asarray(rois[side], dtype=np.float32), foot, True)
        for side in ("near", "far")
    }
    side = max(distances, key=distances.get)
    return side if distances[side] >= -margin else None


def _summarize_observations(observations: list[dict], frame_shape: tuple[int, int]) -> dict | None:
    if not observations:
        return None
    descriptors = [item["descriptor"] for item in observations if item["descriptor"] is not None]
    descriptor = np.mean(descriptors, axis=0) if descriptors else None
    if descriptor is not None:
        descriptor = _normalize_histogram(descriptor)

    positions = np.asarray([item["position"] for item in observations], dtype=float)
    frame_height, frame_width = frame_shape
    diagonal = max(float(np.hypot(frame_width, frame_height)), 1.0)
    movement = (
        float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() / diagonal)
        if len(positions) > 1
        else 0.0
    )
    mean_position = positions.mean(axis=0)
    return {
        "side": observations[0]["side"],
        "descriptor": descriptor,
        "detection_confidence": float(np.mean([item["confidence"] for item in observations])),
        "movement_distance": movement,
        "sample_count": len(observations),
        "mean_position": [
            float(mean_position[0] / max(frame_width, 1)),
            float(mean_position[1] / max(frame_height, 1)),
        ],
    }


def _appearance_distance(descriptor: np.ndarray | None, prototype: np.ndarray | None) -> float:
    if descriptor is None or prototype is None:
        return 1.0
    return float(
        cv2.compareHist(
            descriptor.astype(np.float32),
            prototype.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


def appearance_distance(
    descriptor: np.ndarray | None,
    prototype: np.ndarray | None,
) -> float:
    return _appearance_distance(descriptor, prototype)


def _assign_identities(observations: dict[str, dict | None], prototypes: dict[str, np.ndarray]) -> dict:
    available = [(side, value) for side, value in observations.items() if value is not None]
    if not available:
        return {}

    if prototypes and len(available) == 2:
        direct_cost = sum(
            _appearance_distance(available[i][1]["descriptor"], prototypes.get(PLAYER_IDS[i]))
            for i in range(2)
        )
        swapped_cost = sum(
            _appearance_distance(available[i][1]["descriptor"], prototypes.get(PLAYER_IDS[1 - i]))
            for i in range(2)
        )
        ids = PLAYER_IDS if direct_cost <= swapped_cost else tuple(reversed(PLAYER_IDS))
    elif prototypes:
        remaining = set(PLAYER_IDS)
        ids = []
        for _side, observation in available:
            player_id = min(
                remaining,
                key=lambda candidate: _appearance_distance(observation["descriptor"], prototypes.get(candidate)),
            )
            ids.append(player_id)
            remaining.remove(player_id)
    else:
        ids = tuple("player_1" if side == "near" else "player_2" for side, _ in available)

    assigned = {}
    for player_id, (_side, observation) in zip(ids, available):
        distance = _appearance_distance(observation["descriptor"], prototypes.get(player_id))
        assigned[player_id] = {
            **observation,
            "identity_confidence": 0.5 if player_id not in prototypes else max(0.0, 1.0 - distance),
        }
        descriptor = observation["descriptor"]
        if descriptor is not None:
            if player_id in prototypes:
                updated = prototypes[player_id] * 0.8 + descriptor * 0.2
                prototypes[player_id] = _normalize_histogram(updated)
            else:
                prototypes[player_id] = descriptor.copy()
    return assigned


def _public_player_data(observation: dict | None) -> dict:
    if observation is None:
        return {"detected": False}
    return {
        "detected": True,
        "side": observation["side"],
        "detection_confidence": round(observation["detection_confidence"], 4),
        "identity_confidence": round(observation["identity_confidence"], 4),
        "movement_distance": round(observation["movement_distance"], 6),
        "sample_count": observation["sample_count"],
        "mean_position": [round(value, 6) for value in observation["mean_position"]],
    }


def _attach_player_ids(
    sampled_frames: list[dict],
    assigned: dict,
    frame_shape: tuple[int, int],
) -> dict[str, list[dict]]:
    frame_height, frame_width = frame_shape
    assignments_by_side = {
        observation["side"]: {
            "player_id": player_id,
            "identity_confidence": round(observation["identity_confidence"], 4),
        }
        for player_id, observation in assigned.items()
    }
    trajectories = {player_id: [] for player_id in PLAYER_IDS}

    for sampled_frame in sampled_frames:
        for detection in sampled_frame["detections"]:
            assignment = (
                assignments_by_side.get(detection["side"])
                if detection["is_primary_player_detection"]
                else None
            )
            detection["player_id"] = assignment["player_id"] if assignment else None
            detection["identity_confidence"] = assignment["identity_confidence"] if assignment else None

            if assignment is None:
                continue

            x1, _y1, x2, y2 = detection["bbox"]
            trajectories[assignment["player_id"]].append({
                "time": sampled_frame["time"],
                "frame_index": sampled_frame["frame_index"],
                "side": detection["side"],
                "bbox": detection["bbox"],
                "detection_confidence": detection["confidence"],
                "identity_confidence": assignment["identity_confidence"],
                "position": [
                    round(((x1 + x2) / 2.0) / max(frame_width, 1), 6),
                    round(y2 / max(frame_height, 1), 6),
                ],
            })

    return trajectories


def analyze_player_observations(
    video_path: str,
    segments: list[dict],
    rois: dict,
    model_path: str | Path | None = None,
    sample_seconds: float = 0.5,
    detector=None,
    include_sampled_detections: bool = True,
    inference_backend: str = "opencv",
) -> list[dict]:
    if not segments:
        return []
    detector = detector or YoloXPersonDetector(
        model_path,
        inference_backend=inference_backend,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for player observation analysis: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_interval = max(1, int(fps * sample_seconds))
    prototypes: dict[str, np.ndarray] = {}
    results = []

    try:
        for segment in segments:
            start_frame = max(0, int(float(segment["start"]) * fps))
            end_frame = max(start_frame + 1, int(float(segment["end"]) * fps))
            side_observations = {"near": [], "far": []}
            sampled_frames = []

            for frame_index in range(start_frame, end_frame, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                read, frame = cap.read()
                if not read:
                    break

                detections = []
                best_index_by_side = {}
                for detection in detector.detect(frame):
                    side = court_side(detection["bbox"], rois, frame_height)
                    enriched_detection = {
                        "bbox": detection["bbox"],
                        "confidence": detection["confidence"],
                        "side": side,
                        "is_primary_player_detection": False,
                    }
                    detections.append(enriched_detection)
                    if side is None:
                        continue
                    best_index = best_index_by_side.get(side)
                    if best_index is None or detection["confidence"] > detections[best_index]["confidence"]:
                        best_index_by_side[side] = len(detections) - 1

                for detection_index in best_index_by_side.values():
                    detections[detection_index]["is_primary_player_detection"] = True

                sampled_frames.append({
                    "time": round(frame_index / fps, 3),
                    "frame_index": frame_index,
                    "detections": detections,
                })

                for side, detection_index in best_index_by_side.items():
                    detection = detections[detection_index]
                    x1, _y1, x2, y2 = detection["bbox"]
                    side_observations[side].append({
                        "side": side,
                        "confidence": detection["confidence"],
                        "position": ((x1 + x2) / 2.0, float(y2)),
                        "descriptor": _appearance_descriptor(frame, detection["bbox"]),
                    })

            summarized = {
                side: _summarize_observations(items, (frame_height, frame_width))
                for side, items in side_observations.items()
            }
            assigned = _assign_identities(summarized, prototypes)
            player_trajectories = _attach_player_ids(
                sampled_frames,
                assigned,
                (frame_height, frame_width),
            )
            result = {
                "players": {
                    player_id: _public_player_data(assigned.get(player_id))
                    for player_id in PLAYER_IDS
                },
                "player_trajectories": player_trajectories,
                "player_observation_status": "complete",
            }
            if include_sampled_detections:
                result["sampled_frames"] = sampled_frames
            results.append(result)
    finally:
        cap.release()

    return results
