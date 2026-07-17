from video_extraction.cli import parse_player_handedness
from video_extraction.shot_analysis import associate_hitter, classify_stroke_side
from video_extraction.statistics import build_llm_statistics


def _pose():
    return {
        "left_shoulder": {
            "x": 0.4,
            "y": 0.4,
            "visibility": 0.9,
            "presence": 0.9,
        },
        "right_shoulder": {
            "x": 0.6,
            "y": 0.4,
            "visibility": 0.9,
            "presence": 0.9,
        },
    }


def test_associates_ball_with_nearest_player_box():
    association = associate_hitter(
        1.0,
        [{"time": 1.02, "visible": True, "x": 110, "y": 100, "confidence": 0.8}],
        {
            "player_1": [{
                "time": 1.0,
                "bbox": [80, 60, 140, 180],
                "detection_confidence": 0.8,
                "identity_confidence": 0.7,
            }],
            "player_2": [{
                "time": 1.0,
                "bbox": [400, 60, 460, 180],
                "detection_confidence": 0.8,
                "identity_confidence": 0.7,
            }],
        },
    )

    assert association["player_id"] == "player_1"
    assert association["reason"] is None


def test_hitter_association_abstains_without_ball():
    association = associate_hitter(
        1.0,
        [],
        {"player_1": [{"time": 1.0, "bbox": [80, 60, 140, 180]}]},
    )

    assert association == {
        "player_id": None,
        "ball": None,
        "reason": "no_nearby_ball",
    }


def test_hitter_association_abstains_on_low_identity_confidence():
    association = associate_hitter(
        1.0,
        [{"time": 1.0, "visible": True, "x": 110, "y": 100, "confidence": 0.8}],
        {
            "player_1": [{
                "time": 1.0,
                "bbox": [80, 60, 140, 180],
                "detection_confidence": 0.8,
                "identity_confidence": 0.2,
            }]
        },
    )

    assert association["player_id"] is None
    assert association["reason"] == "low_player_confidence"


def test_classifies_right_handed_contact_side():
    forehand = classify_stroke_side(_pose(), (0.8, 0.4), "right", 0.8)
    backhand = classify_stroke_side(_pose(), (0.2, 0.4), "right", 0.8)

    assert forehand["classification"] == "forehand"
    assert backhand["classification"] == "backhand"


def test_stroke_classification_requires_handedness():
    result = classify_stroke_side(_pose(), (0.8, 0.4), None, 0.8)

    assert result["classification"] == "unknown"
    assert result["reason"] == "handedness_required"


def test_parses_player_handedness():
    assert parse_player_handedness([
        "player_1=right",
        "player_2=left",
    ]) == {
        "player_1": "right",
        "player_2": "left",
    }


def test_statistics_expose_classified_shots():
    report = [{
        "index": 1,
        "start": 1.0,
        "end": 2.0,
        "players": {"player_1": {"detected": False}},
        "shots": [{
            "time": 1.5,
            "player_id": "player_1",
            "classification": "forehand",
            "confidence": 0.8,
            "reason": None,
        }],
    }]

    stats = build_llm_statistics(report)

    assert stats["players"]["player_1"]["shot_counts"]["forehand"] == 1
    assert stats["data_quality"]["shots"]["classified_count"] == 1
    assert (
        "confidence-gated forehand/backhand classification"
        in stats["analysis_capabilities"]["supported"]
    )
    assert (
        "forehand/backhand classification"
        not in stats["analysis_capabilities"]["unsupported"]
    )
