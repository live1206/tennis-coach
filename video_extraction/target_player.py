from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_extraction.vision.player_observation import (
    appearance_descriptor,
    appearance_distance,
)


def select_target_match(
    target_descriptor: np.ndarray,
    player_descriptors: dict[str, np.ndarray],
    minimum_confidence: float = 0.45,
    minimum_margin: float = 0.08,
) -> dict:
    if not player_descriptors:
        return {
            "player_id": None,
            "confidence": 0.0,
            "reason": "no_player_descriptors",
        }
    ranked = sorted(
        (
            1.0 - appearance_distance(target_descriptor, descriptor),
            player_id,
        )
        for player_id, descriptor in player_descriptors.items()
    )
    best_confidence, best_player_id = ranked[-1]
    margin = (
        best_confidence - ranked[-2][0]
        if len(ranked) > 1
        else best_confidence
    )
    if best_confidence < minimum_confidence:
        return {
            "player_id": None,
            "confidence": round(best_confidence, 6),
            "reason": "low_match_confidence",
        }
    if margin < minimum_margin:
        return {
            "player_id": None,
            "confidence": round(best_confidence, 6),
            "reason": "ambiguous_match",
        }
    return {
        "player_id": best_player_id,
        "confidence": round(best_confidence, 6),
        "margin": round(margin, 6),
        "reason": None,
    }


def match_target_player(
    video_path: str | Path,
    segments: list[dict],
    target_image_path: str | Path,
    maximum_samples_per_player: int = 30,
) -> tuple[list[dict], dict]:
    target_image = cv2.imread(str(target_image_path))
    if target_image is None:
        raise RuntimeError(f"Cannot read target player image: {target_image_path}")
    target_descriptor = appearance_descriptor(target_image)
    if target_descriptor is None:
        raise RuntimeError("Target player image does not contain a usable appearance crop")

    samples_by_player: dict[str, list[dict]] = {}
    for segment in segments:
        for player_id, trajectory in segment.get("player_trajectories", {}).items():
            samples_by_player.setdefault(player_id, []).extend(trajectory)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for target-player matching: {video_path}")
    descriptors: dict[str, np.ndarray] = {}
    try:
        for player_id, samples in samples_by_player.items():
            if not samples:
                continue
            step = max(1, len(samples) // maximum_samples_per_player)
            player_descriptors = []
            for sample in samples[::step][:maximum_samples_per_player]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_index"]))
                read, frame = cap.read()
                if not read:
                    continue
                descriptor = appearance_descriptor(frame, sample["bbox"])
                if descriptor is not None:
                    player_descriptors.append(descriptor)
            if player_descriptors:
                mean_descriptor = np.mean(player_descriptors, axis=0)
                descriptors[player_id] = cv2.normalize(
                    mean_descriptor,
                    mean_descriptor,
                    alpha=1.0,
                    norm_type=cv2.NORM_L1,
                ).flatten()
    finally:
        cap.release()

    match = select_target_match(target_descriptor, descriptors)
    return [
        {**segment, "target_player": match}
        for segment in segments
    ], match
