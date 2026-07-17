import numpy as np

from tennis_coach import video_extraction
from tennis_coach.video_extraction import enrich_report, load_report
from tennis_coach.vision.player_observation import _assign_identities, _public_player_data


def test_load_report_requires_segment_timestamps(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('[{"start": 1.0, "end": 2.0}]')

    assert load_report(report) == [{"start": 1.0, "end": 2.0}]


def test_enrich_report_marks_court_detection_skip(monkeypatch):
    monkeypatch.setattr(video_extraction, "select_rois", lambda _video_path: None)

    enriched = enrich_report("missing.mp4", [{"start": 1.0, "end": 2.0}], rois=None)

    assert enriched[0]["video_extraction"]["status"] == "skipped_court_detection"


def test_public_player_data_keeps_json_safe_fields():
    public = _public_player_data({
        "side": "near",
        "detection_confidence": 0.98765,
        "identity_confidence": 0.87654,
        "movement_distance": 0.1234567,
        "sample_count": 3,
        "mean_position": [0.3333333, 0.6666666],
        "descriptor": np.asarray([1.0, 0.0], dtype=np.float32),
    })

    assert public == {
        "detected": True,
        "side": "near",
        "detection_confidence": 0.9877,
        "identity_confidence": 0.8765,
        "movement_distance": 0.123457,
        "sample_count": 3,
        "mean_position": [0.333333, 0.666667],
    }


def test_identity_assignment_follows_appearance_after_side_switch():
    prototypes = {}
    first = _assign_identities({
        "near": {
            "side": "near",
            "descriptor": np.asarray([1.0, 0.0], dtype=np.float32),
            "detection_confidence": 0.9,
            "movement_distance": 0.1,
            "sample_count": 1,
            "mean_position": [0.5, 0.8],
        },
        "far": {
            "side": "far",
            "descriptor": np.asarray([0.0, 1.0], dtype=np.float32),
            "detection_confidence": 0.9,
            "movement_distance": 0.1,
            "sample_count": 1,
            "mean_position": [0.5, 0.2],
        },
    }, prototypes)
    second = _assign_identities({
        "near": {**first["player_2"], "descriptor": np.asarray([0.0, 1.0], dtype=np.float32), "side": "near"},
        "far": {**first["player_1"], "descriptor": np.asarray([1.0, 0.0], dtype=np.float32), "side": "far"},
    }, prototypes)

    assert second["player_1"]["side"] == "far"
    assert second["player_2"]["side"] == "near"
