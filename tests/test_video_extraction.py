import numpy as np

from video_extraction import cli
from video_extraction.ball_annotations import evaluate_predictions, validate_annotation_manifest
from video_extraction.cli import enrich_report, load_report
from video_extraction.vision.ball_tracking import TrackNetOnnxDetector, observations_for_segment
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


def test_empty_report_does_not_initialize_models():
    assert enrich_report(
        "missing.mp4",
        [],
        ball_model_path="missing.onnx",
    ) == []


def test_ball_tracking_survives_court_detection_failure(monkeypatch):
    monkeypatch.setattr(cli, "TrackNetOnnxDetector", lambda _path: object())
    monkeypatch.setattr(
        cli,
        "track_ball_video",
        lambda *_args, **_kwargs: [{"time": 1.5, "visible": True}],
    )
    monkeypatch.setattr(cli, "select_rois", lambda _video_path: None)

    enriched = enrich_report(
        "missing.mp4",
        [{"start": 1.0, "end": 2.0}],
        ball_model_path="ball.onnx",
    )

    assert enriched[0]["video_extraction"]["status"] == "skipped_court_detection"
    assert enriched[0]["video_extraction"]["ball_tracking_status"] == "complete"
    assert enriched[0]["ball_trajectory"] == [{"time": 1.5, "visible": True}]


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


def test_validates_complete_ball_annotation_manifest():
    manifest = {
        "width": 1920,
        "height": 1080,
        "frames": [{
            "frame_index": 42,
            "visibility": "visible",
            "x": 1000,
            "y": 500,
            "event": "hit",
        }],
    }

    assert validate_annotation_manifest(manifest, require_complete=True) == []


def test_rejects_visible_ball_outside_frame():
    manifest = {
        "width": 1920,
        "height": 1080,
        "frames": [{
            "frame_index": 42,
            "visibility": "visible",
            "x": 2000,
            "y": 500,
            "event": None,
        }],
    }

    assert validate_annotation_manifest(manifest) == [
        "frames[0].x must be within the frame for a visible ball"
    ]


def test_reports_invalid_dimensions_without_crashing():
    manifest = {
        "width": None,
        "height": None,
        "frames": [{
            "frame_index": 42,
            "visibility": "visible",
            "x": 100,
            "y": 100,
            "event": None,
        }],
    }

    assert validate_annotation_manifest(manifest) == [
        "width must be a positive integer",
        "height must be a positive integer",
    ]


def test_tracknet_heatmap_decoder_uses_largest_component():
    detector = TrackNetOnnxDetector.__new__(TrackNetOnnxDetector)
    detector.class_threshold = 128
    detector.minimum_component_area = 3
    class_map = np.zeros((20, 20), dtype=np.uint8)
    class_map[2:4, 2:4] = 200
    class_map[10:13, 12:15] = 240

    x, y, confidence = detector._locate_ball(class_map)

    assert (x, y) == (13.0, 11.0)
    assert confidence == 240 / 255


def test_tracknet_decoder_accepts_pixel_major_flattened_output():
    detector = TrackNetOnnxDetector.__new__(TrackNetOnnxDetector)
    detector.input_height = 2
    detector.input_width = 2
    output = np.zeros((1, 4, 256), dtype=np.float32)
    output[0, :, 200] = 1.0

    np.testing.assert_array_equal(
        detector._class_map(output),
        np.full((2, 2), 200, dtype=np.uint8),
    )


def test_filters_ball_observations_by_segment_time():
    observations = [
        {"time": 0.5, "visible": True},
        {"time": 1.0, "visible": False},
        {"time": 2.0, "visible": True},
    ]

    assert observations_for_segment(observations, 1.0, 2.0) == [
        {"time": 1.0, "visible": False}
    ]


def test_evaluates_ball_predictions_with_localization_tolerance():
    manifest = {
        "frames": [
            {"frame_index": 1, "visibility": "visible", "x": 100, "y": 100},
            {"frame_index": 2, "visibility": "visible", "x": 200, "y": 200},
            {"frame_index": 3, "visibility": "absent", "x": None, "y": None},
            {"frame_index": 4, "visibility": "occluded", "x": None, "y": None},
        ],
    }
    observations = [
        {"frame_index": 1, "visible": True, "x": 103, "y": 104},
        {"frame_index": 2, "visible": False},
        {"frame_index": 3, "visible": True, "x": 300, "y": 300},
        {"frame_index": 4, "visible": False},
    ]

    metrics = evaluate_predictions(manifest, observations, tolerance_pixels=5)

    assert metrics == {
        "evaluated_frames": 4,
        "tolerance_pixels": 5,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "mean_localization_error": 5.0,
        "median_localization_error": 5.0,
    }
