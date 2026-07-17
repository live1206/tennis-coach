import numpy as np

from video_extraction.outcomes import infer_segment_outcome
from video_extraction.shot_analysis import assign_shot_roles
from video_extraction.statistics import build_llm_statistics
from video_extraction.target_player import select_target_match


def test_selects_target_player_with_clear_appearance_margin():
    target = np.array([1.0, 0.0], dtype=np.float32)

    match = select_target_match(
        target,
        {
            "player_1": np.array([1.0, 0.0], dtype=np.float32),
            "player_2": np.array([0.0, 1.0], dtype=np.float32),
        },
    )

    assert match["player_id"] == "player_1"
    assert match["confidence"] == 1.0


def test_assigns_serve_and_return_only_after_overhead_evidence():
    shots = assign_shot_roles([
        {
            "time": 1.0,
            "player_id": "player_1",
            "confidence": 0.8,
            "contact_confidence": 0.8,
            "serve_candidate_confidence": 0.7,
        },
        {
            "time": 2.0,
            "player_id": "player_2",
            "confidence": 0.75,
            "contact_confidence": 0.75,
            "serve_candidate_confidence": 0.0,
        },
    ])

    assert shots[0]["shot_role"] == "serve"
    assert shots[1]["shot_role"] == "return"


def test_outcome_marks_continuation_from_opponent_contact():
    segment = infer_segment_outcome({
        "players": {"player_1": {}, "player_2": {}},
        "shots": [
            {"time": 1.0, "player_id": "player_1", "contact_confidence": 0.8},
            {"time": 2.0, "player_id": "player_2", "contact_confidence": 0.7},
        ],
        "ball_trajectory": [],
    })

    assert segment["shots"][0]["outcome"] == "continued"
    assert segment["shots"][1]["outcome"] == "terminal_unknown"
    assert segment["outcome"]["classification"] == "unknown"


def test_outcome_uses_validated_terminal_error_event():
    segment = infer_segment_outcome({
        "players": {"player_1": {}, "player_2": {}},
        "shots": [
            {"time": 1.0, "player_id": "player_1", "contact_confidence": 0.8},
        ],
        "ball_trajectory": [
            {"time": 1.2, "event": "out", "confidence": 0.9},
        ],
    })

    assert segment["shots"][0]["outcome"] == "error"
    assert segment["outcome"]["winner_player_id"] == "player_2"
    assert segment["outcome"]["terminal_event"] == "out"


def test_outcome_rejects_zero_confidence_terminal_event():
    segment = infer_segment_outcome({
        "players": {"player_1": {}, "player_2": {}},
        "shots": [{
            "time": 1.0,
            "player_id": "player_1",
            "contact_confidence": 0.8,
        }],
        "ball_trajectory": [
            {"time": 1.2, "event": "out", "confidence": 0.0},
        ],
    })

    assert segment["outcome"]["classification"] == "unknown"
    assert segment["outcome"]["reason"] == "low_terminal_event_confidence"


def test_outcome_rejects_unresolved_contact_before_terminal_event():
    segment = infer_segment_outcome({
        "players": {"player_1": {}, "player_2": {}},
        "shots": [
            {
                "time": 1.0,
                "player_id": "player_1",
                "contact_confidence": 0.8,
            },
            {
                "time": 1.5,
                "player_id": None,
                "contact_confidence": 0.0,
            },
        ],
        "ball_trajectory": [
            {"time": 1.7, "event": "net", "confidence": 0.9},
        ],
    })

    assert segment["outcome"]["classification"] == "unknown"
    assert segment["outcome"]["reason"] == "terminal_hitter_unresolved"


def test_statistics_expose_target_roles_and_outcomes():
    report = [{
        "index": 1,
        "start": 1.0,
        "end": 3.0,
        "players": {
            "player_1": {"detected": False},
            "player_2": {"detected": False},
        },
        "target_player": {
            "player_id": "player_1",
            "confidence": 0.9,
            "margin": 0.4,
            "reason": None,
        },
        "shots": [{
            "time": 1.0,
            "player_id": "player_1",
            "classification": "forehand",
            "confidence": 0.8,
            "reason": None,
            "shot_role": "serve",
            "role_confidence": 0.8,
            "outcome": "error",
            "outcome_confidence": 0.8,
            "contact_method": "audio_ball_player_pose",
            "contact_point": {
                "x": 100.0,
                "y": 50.0,
                "x_normalized": 0.5,
                "y_normalized": 0.25,
            },
            "racket_hand_position": {
                "x": 0.48,
                "y": 0.3,
                "visibility": 0.9,
                "presence": 0.9,
            },
        }],
        "outcome": {
            "classification": "point_ended",
            "winner_player_id": "player_2",
            "confidence": 0.8,
        },
    }]

    stats = build_llm_statistics(report)

    assert stats["target_player"]["player_id"] == "player_1"
    assert stats["players"]["player_1"]["is_target"] is True
    assert stats["players"]["player_1"]["shot_continuation_rate"] == 0.0
    assert stats["segments"][0]["shots"][0]["contact_point"]["x"] == 100.0
    assert "serve and return classification" not in stats["analysis_capabilities"]["unsupported"]
    assert "winner/error attribution" not in stats["analysis_capabilities"]["unsupported"]
