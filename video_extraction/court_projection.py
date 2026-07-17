from __future__ import annotations

import cv2
import numpy as np


class CourtProjector:
    """Project image points onto a normalized top-down court plane."""

    def __init__(self, rois: dict, frame_width: int, frame_height: int):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        far = rois["far"]
        near = rois["near"]
        source = np.asarray(
            [far[0], far[1], near[3], near[2]],
            dtype=np.float32,
        )
        target = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        self.homography = cv2.getPerspectiveTransform(source, target)
        self.frame_width = frame_width
        self.frame_height = frame_height

    def project_pixel(self, x: float, y: float) -> dict:
        point = np.asarray([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.homography)[0, 0]
        court_x = float(projected[0])
        court_y = float(projected[1])
        return {
            "x": round(court_x, 6),
            "y": round(court_y, 6),
            "inside_court": 0.0 <= court_x <= 1.0 and 0.0 <= court_y <= 1.0,
        }

    def project_normalized_image(self, position: list[float]) -> dict:
        return self.project_pixel(
            position[0] * self.frame_width,
            position[1] * self.frame_height,
        )


def add_court_projections(
    player_trajectories: dict[str, list[dict]],
    ball_trajectory: list[dict] | None,
    projector: CourtProjector,
) -> None:
    for points in player_trajectories.values():
        for point in points:
            point["court_position"] = projector.project_normalized_image(point["position"])

    if ball_trajectory is None:
        return
    for observation in ball_trajectory:
        if observation.get("visible"):
            observation["court_projection"] = projector.project_pixel(
                observation["x"],
                observation["y"],
            )
