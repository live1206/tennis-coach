from __future__ import annotations

from pathlib import Path

import numpy as np

from video_extraction.vision.player_observation import (
    COCO_SPORTS_BALL_CLASS_ID,
    YoloXDetector,
)


class YoloXBallDetector:
    """Use YOLOX-Nano's licensed COCO sports-ball class as a ball baseline."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.05,
        nms_threshold: float = 0.45,
        inference_backend: str = "opencv",
    ):
        self.detector = YoloXDetector(
            model_path,
            class_id=COCO_SPORTS_BALL_CLASS_ID,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            inference_backend=inference_backend,
        )

    def detect(self, frames: list[np.ndarray]) -> dict:
        if len(frames) != 3:
            raise ValueError("Ball tracking requires exactly three frame inputs")
        height, width = frames[-1].shape[:2]
        detections = self.detector.detect(frames[-1])
        if not detections:
            return {"visible": False, "confidence": 0.0}
        detection = max(detections, key=lambda item: item["confidence"])
        x1, y1, x2, y2 = detection["bbox"]
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        return {
            "visible": True,
            "x": round(x, 3),
            "y": round(y, 3),
            "x_normalized": round(x / max(width, 1), 6),
            "y_normalized": round(y / max(height, 1), 6),
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
        }
