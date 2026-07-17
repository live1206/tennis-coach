from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


class MediaPipePoseDetector:
    """Run an externally supplied MediaPipe Pose Landmarker model."""

    def __init__(self, model_path: str | Path):
        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(f"MediaPipe pose model not found: {model}")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Pose extraction requires the optional 'mediapipe' dependency."
            ) from error

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, crop: np.ndarray) -> dict[str, dict] | None:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        return {
            name: {
                "x": float(landmarks[index].x),
                "y": float(landmarks[index].y),
                "visibility": float(landmarks[index].visibility),
                "presence": float(landmarks[index].presence),
            }
            for name, index in POSE_LANDMARKS.items()
        }


def expanded_crop(
    frame: np.ndarray,
    bbox: list[int],
    expansion: float = 0.2,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    crop_box = (
        max(0, int(x1 - width * expansion)),
        max(0, int(y1 - height * expansion)),
        min(frame_width, int(x2 + width * expansion)),
        min(frame_height, int(y2 + height * expansion)),
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    return frame[crop_y1:crop_y2, crop_x1:crop_x2], crop_box


def map_landmarks_to_frame(
    landmarks: dict[str, dict],
    crop_box: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> dict[str, dict]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    frame_height, frame_width = frame_shape
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    return {
        name: {
            "x": round((crop_x1 + point["x"] * crop_width) / frame_width, 6),
            "y": round((crop_y1 + point["y"] * crop_height) / frame_height, 6),
            "visibility": round(point["visibility"], 6),
            "presence": round(point["presence"], 6),
        }
        for name, point in landmarks.items()
    }
