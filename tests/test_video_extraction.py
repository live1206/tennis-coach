import numpy as np

from video_extraction import cli
from video_extraction.cli import enrich_report, load_report
from video_extraction.vision.player_observation import (
    YoloXPersonDetector,
    _attach_player_ids,
    _assign_identities,
    _public_player_data,
)


def test_load_report_requires_segment_timestamps(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('[{"start": 1.0, "end": 2.0}]')

    assert load_report(report) == [{"start": 1.0, "end": 2.0}]


def test_enrich_report_marks_court_detection_skip(monkeypatch):
    monkeypatch.setattr(cli, "select_rois", lambda _video_path: None)

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


def test_yolox_preprocessing_preserves_raw_rgb_values():
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]

    blob, scale = YoloXPersonDetector._preprocess(frame)

    assert scale == 1.0
    np.testing.assert_array_equal(blob[0, :, 0, 0], [30.0, 20.0, 10.0])


def test_attaches_player_ids_and_builds_grouped_trajectories():
    sampled_frames = [{
        "time": 1.5,
        "frame_index": 90,
        "detections": [
            {
                "bbox": [100, 200, 200, 400],
                "confidence": 0.9,
                "side": "near",
                "is_primary_player_detection": True,
            },
            {
                "bbox": [300, 100, 350, 200],
                "confidence": 0.8,
                "side": "near",
                "is_primary_player_detection": False,
            },
        ],
    }]
    assigned = {
        "player_2": {
            "side": "near",
            "identity_confidence": 0.81234,
        },
    }

    trajectories = _attach_player_ids(sampled_frames, assigned, (500, 1000))

    primary, unrelated = sampled_frames[0]["detections"]
    assert primary["player_id"] == "player_2"
    assert primary["identity_confidence"] == 0.8123
    assert unrelated["player_id"] is None
    assert unrelated["identity_confidence"] is None
    assert trajectories["player_1"] == []
    assert trajectories["player_2"] == [{
        "time": 1.5,
        "frame_index": 90,
        "side": "near",
        "bbox": [100, 200, 200, 400],
        "detection_confidence": 0.9,
        "identity_confidence": 0.8123,
        "position": [0.15, 0.8],
    }]
