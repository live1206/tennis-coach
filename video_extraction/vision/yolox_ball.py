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
        tile_grid: int = 1,
        tile_overlap: float = 0.15,
    ):
        if tile_grid < 1:
            raise ValueError("tile_grid must be at least 1")
        if not 0.0 <= tile_overlap < 0.5:
            raise ValueError("tile_overlap must be between 0 and 0.5")
        self.detector = YoloXDetector(
            model_path,
            class_id=COCO_SPORTS_BALL_CLASS_ID,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            inference_backend=inference_backend,
        )
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap

    def _detections(self, frame: np.ndarray) -> list[dict]:
        if self.tile_grid == 1:
            return self.detector.detect(frame)

        height, width = frame.shape[:2]
        detections = []
        for row in range(self.tile_grid):
            for column in range(self.tile_grid):
                base_x1 = round(column * width / self.tile_grid)
                base_x2 = round((column + 1) * width / self.tile_grid)
                base_y1 = round(row * height / self.tile_grid)
                base_y2 = round((row + 1) * height / self.tile_grid)
                overlap_x = round((base_x2 - base_x1) * self.tile_overlap)
                overlap_y = round((base_y2 - base_y1) * self.tile_overlap)
                x1 = max(0, base_x1 - overlap_x)
                x2 = min(width, base_x2 + overlap_x)
                y1 = max(0, base_y1 - overlap_y)
                y2 = min(height, base_y2 + overlap_y)
                crop = frame[y1:y2, x1:x2]
                for detection in self.detector.detect(crop):
                    crop_box = detection["bbox"]
                    detections.append({
                        **detection,
                        "bbox": [
                            crop_box[0] + x1,
                            crop_box[1] + y1,
                            crop_box[2] + x1,
                            crop_box[3] + y1,
                        ],
                    })
        return detections

    def detect(self, frames: list[np.ndarray]) -> dict:
        if len(frames) != 3:
            raise ValueError("Ball tracking requires exactly three frame inputs")
        height, width = frames[-1].shape[:2]
        detections = self._detections(frames[-1])
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
