from __future__ import annotations

import math
from pathlib import Path

import cv2

from video_extraction.vision.pose import (
    MediaPipePoseDetector,
    expanded_crop,
    map_landmarks_to_frame,
)


def _nearest(items: list[dict], time: float, maximum_delta: float) -> dict | None:
    candidates = [
        item for item in items
        if abs(float(item["time"]) - time) <= maximum_delta
    ]
    return min(
        candidates,
        key=lambda item: abs(float(item["time"]) - time),
        default=None,
    )


def _point_box_distance(point: tuple[float, float], bbox: list[int]) -> float:
    x, y = point
    x1, y1, x2, y2 = bbox
    delta_x = max(x1 - x, 0.0, x - x2)
    delta_y = max(y1 - y, 0.0, y - y2)
    diagonal = max(math.hypot(x2 - x1, y2 - y1), 1.0)
    return math.hypot(delta_x, delta_y) / diagonal


def associate_hitter(
    hit_time: float,
    ball_trajectory: list[dict],
    player_trajectories: dict[str, list[dict]],
    ball_time_tolerance: float = 0.15,
    player_time_tolerance: float = 0.35,
    maximum_box_distance: float = 0.75,
    minimum_detection_confidence: float = 0.25,
    minimum_identity_confidence: float = 0.4,
) -> dict:
    ball = _nearest(
        [
            observation
            for observation in ball_trajectory
            if observation.get("visible")
        ],
        hit_time,
        ball_time_tolerance,
    )
    if ball is None:
        return {"player_id": None, "ball": None, "reason": "no_nearby_ball"}

    candidates = []
    low_quality_player_found = False
    for player_id, trajectory in player_trajectories.items():
        player = _nearest(trajectory, hit_time, player_time_tolerance)
        if player is None or not player.get("bbox"):
            continue
        detection_confidence = float(player.get("detection_confidence") or 0.0)
        identity_confidence = float(player.get("identity_confidence") or 0.0)
        if (
            detection_confidence < minimum_detection_confidence
            or identity_confidence < minimum_identity_confidence
        ):
            low_quality_player_found = True
            continue
        distance = _point_box_distance((ball["x"], ball["y"]), player["bbox"])
        candidates.append((distance, player_id, player))
    if not candidates:
        return {
            "player_id": None,
            "ball": ball,
            "reason": (
                "low_player_confidence"
                if low_quality_player_found
                else "no_nearby_player"
            ),
        }

    candidates.sort(key=lambda item: item[0])
    distance, player_id, player = candidates[0]
    if distance > maximum_box_distance:
        return {"player_id": None, "ball": ball, "reason": "ball_too_far_from_players"}
    if len(candidates) > 1 and candidates[1][0] - distance < 0.1:
        return {"player_id": None, "ball": ball, "reason": "ambiguous_player"}
    player_confidence = min(
        float(player["detection_confidence"]),
        float(player["identity_confidence"]),
    )
    proximity_confidence = max(0.0, 1.0 - distance / maximum_box_distance)
    contact_confidence = (
        float(ball["confidence"]) * player_confidence * proximity_confidence
    ) ** (1.0 / 3.0)
    return {
        "player_id": player_id,
        "player": player,
        "ball": ball,
        "box_distance": round(distance, 6),
        "player_confidence": player_confidence,
        "contact_confidence": round(contact_confidence, 6),
        "reason": None,
    }


def classify_stroke_side(
    landmarks: dict[str, dict],
    contact_point: tuple[float, float],
    handedness: str | None,
    ball_confidence: float,
    player_confidence: float = 1.0,
    minimum_confidence: float = 0.35,
) -> dict:
    if handedness not in {"left", "right"}:
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "reason": "handedness_required",
        }
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    pose_confidence = min(
        left_shoulder["visibility"],
        right_shoulder["visibility"],
        left_shoulder["presence"],
        right_shoulder["presence"],
    )
    shoulder_x = right_shoulder["x"] - left_shoulder["x"]
    shoulder_y = right_shoulder["y"] - left_shoulder["y"]
    shoulder_width = math.hypot(shoulder_x, shoulder_y)
    if shoulder_width < 0.01 or pose_confidence < 0.4:
        return {
            "classification": "unknown",
            "confidence": round(pose_confidence, 6),
            "reason": "low_pose_quality",
        }

    midpoint_x = (left_shoulder["x"] + right_shoulder["x"]) / 2.0
    midpoint_y = (left_shoulder["y"] + right_shoulder["y"]) / 2.0
    contact_x, contact_y = contact_point
    side_score = (
        (contact_x - midpoint_x) * shoulder_x
        + (contact_y - midpoint_y) * shoulder_y
    ) / (shoulder_width * shoulder_width)
    separation = min(abs(side_score), 1.0)
    confidence = (
        pose_confidence * ball_confidence * player_confidence * separation
    ) ** 0.25
    if abs(side_score) < 0.2 or confidence < minimum_confidence:
        return {
            "classification": "unknown",
            "confidence": round(confidence, 6),
            "side_score": round(side_score, 6),
            "reason": "ambiguous_contact_side",
        }

    anatomical_side = "right" if side_score > 0 else "left"
    classification = (
        "forehand" if anatomical_side == handedness else "backhand"
    )
    return {
        "classification": classification,
        "confidence": round(confidence, 6),
        "side_score": round(side_score, 6),
        "reason": None,
    }


def serve_candidate_confidence(
    landmarks: dict[str, dict],
    contact_point: tuple[float, float],
    handedness: str | None,
) -> float:
    if handedness not in {"left", "right"}:
        return 0.0
    shoulder = landmarks[f"{handedness}_shoulder"]
    wrist = landmarks[f"{handedness}_wrist"]
    opposite_shoulder = landmarks[
        "left_shoulder" if handedness == "right" else "right_shoulder"
    ]
    shoulder_width = math.hypot(
        shoulder["x"] - opposite_shoulder["x"],
        shoulder["y"] - opposite_shoulder["y"],
    )
    if shoulder_width < 0.01:
        return 0.0
    shoulder_mid_y = (shoulder["y"] + opposite_shoulder["y"]) / 2.0
    contact_height = (shoulder_mid_y - contact_point[1]) / shoulder_width
    wrist_height = (shoulder_mid_y - wrist["y"]) / shoulder_width
    landmark_confidence = min(
        shoulder["visibility"],
        shoulder["presence"],
        wrist["visibility"],
        wrist["presence"],
    )
    overhead_score = min(max(min(contact_height, wrist_height), 0.0), 1.0)
    return round(landmark_confidence * overhead_score, 6)


def assign_shot_roles(shots: list[dict]) -> list[dict]:
    resolved_indices = [
        index
        for index, shot in enumerate(shots)
        if shot.get("player_id") is not None
    ]
    if not resolved_indices:
        return [
            {
                **shot,
                "shot_role": "unknown",
                "role_confidence": 0.0,
                "role_reason": "hitter_unresolved",
            }
            for shot in shots
        ]

    first_index = resolved_indices[0]
    first = shots[first_index]
    serve_confidence = (
        float(first.get("serve_candidate_confidence") or 0.0)
        * float(first.get("contact_confidence") or 0.0)
    ) ** 0.5
    serve_detected = serve_confidence >= 0.45
    output = []
    return_assigned = False
    for index, shot in enumerate(shots):
        if shot.get("player_id") is None:
            role = {
                "shot_role": "unknown",
                "role_confidence": 0.0,
                "role_reason": "hitter_unresolved",
            }
        elif index == first_index:
            role = {
                "shot_role": "serve" if serve_detected else "unknown",
                "role_confidence": serve_confidence,
                "role_reason": None if serve_detected else "serve_evidence_insufficient",
            }
        elif (
            serve_detected
            and not return_assigned
            and shot["player_id"] != first["player_id"]
        ):
            return_assigned = True
            role = {
                "shot_role": "return",
                "role_confidence": round(
                    min(
                        serve_confidence,
                        float(shot.get("contact_confidence") or 0.0),
                    ),
                    6,
                ),
                "role_reason": None,
            }
        else:
            role = {
                "shot_role": "rally_shot",
                "role_confidence": float(shot.get("contact_confidence") or 0.0),
                "role_reason": None,
            }
        output.append({**shot, **role})
    return output


def analyze_segment_shots(
    video_path: str | Path,
    segments: list[dict],
    pose_model_path: str | Path,
    player_handedness: dict[str, str],
    pose_detector=None,
) -> list[dict]:
    detector = pose_detector or MediaPipePoseDetector(pose_model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for shot analysis: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    results = []
    try:
        for segment in segments:
            shots = []
            hit_times = segment.get("audio", {}).get("hit_times", [])
            for hit_time in hit_times:
                association = associate_hitter(
                    float(hit_time),
                    segment.get("ball_trajectory", []),
                    segment.get("player_trajectories", {}),
                )
                base = {
                    "time": float(hit_time),
                    "player_id": association["player_id"],
                    "classification": "unknown",
                    "confidence": 0.0,
                }
                if association["player_id"] is None:
                    shots.append({**base, "reason": association["reason"]})
                    continue
                base["contact_confidence"] = association["contact_confidence"]

                frame_index = max(0, round(float(hit_time) * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                read, frame = cap.read()
                if not read:
                    shots.append({**base, "reason": "frame_unavailable"})
                    continue
                crop, crop_box = expanded_crop(frame, association["player"]["bbox"])
                if crop.size == 0:
                    shots.append({**base, "reason": "empty_player_crop"})
                    continue
                crop_landmarks = detector.detect(crop)
                if crop_landmarks is None:
                    shots.append({**base, "reason": "pose_unavailable"})
                    continue
                landmarks = map_landmarks_to_frame(
                    crop_landmarks,
                    crop_box,
                    (frame_height, frame_width),
                )
                ball = association["ball"]
                classification = classify_stroke_side(
                    landmarks,
                    (
                        ball["x"] / max(frame_width, 1),
                        ball["y"] / max(frame_height, 1),
                    ),
                    player_handedness.get(association["player_id"]),
                    float(ball["confidence"]),
                    association["player_confidence"],
                )
                contact_point = (
                    ball["x"] / max(frame_width, 1),
                    ball["y"] / max(frame_height, 1),
                )
                shots.append({
                    **base,
                    **classification,
                    "frame_index": frame_index,
                    "ball_time": ball["time"],
                    "ball_confidence": ball["confidence"],
                    "player_box_distance": association["box_distance"],
                    "contact_method": "audio_ball_player_pose",
                    "contact_point": {
                        "x": ball["x"],
                        "y": ball["y"],
                        "x_normalized": round(contact_point[0], 6),
                        "y_normalized": round(contact_point[1], 6),
                    },
                    "racket_hand_position": landmarks.get(
                        f"{player_handedness.get(association['player_id'])}_wrist"
                    ),
                    "pose_landmarks": landmarks,
                    "serve_candidate_confidence": serve_candidate_confidence(
                        landmarks,
                        contact_point,
                        player_handedness.get(association["player_id"]),
                    ),
                })
            results.append({**segment, "shots": assign_shot_roles(shots)})
    finally:
        cap.release()
    return results
