import numpy as np

from video_extraction import cli
from video_extraction.ball_annotations import evaluate_predictions, validate_annotation_manifest
from video_extraction.cli import enrich_report, load_report
from video_extraction.court_projection import CourtProjector, add_court_projections
from video_extraction.vision import ball_tracking
from video_extraction.vision.ball_tracking import (
    TrackNetOnnxDetector,
    interpolate_short_gaps,
    observations_for_segment,
    track_ball_intervals,
)
from video_extraction.statistics import (
    aggregate_ball_summaries,
    build_llm_statistics,
    summarize_ball_trajectory,
)
from video_extraction.vision.player_observation import (
    COCO_SPORTS_BALL_CLASS_ID,
    YoloXDetector,
    YoloXPersonDetector,
    _attach_player_ids,
    _assign_identities,
    _public_player_data,
)
from video_extraction.vision.yolox_ball import YoloXBallDetector


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


def test_cli_writes_analysis_and_separates_internal_outputs(monkeypatch, tmp_path):
    segments = [{
        "index": 1,
        "start": 1.0,
        "end": 2.0,
        "audio": {
            "sample_rate": 22050,
            "hit_times": [1.5],
            "hit_energies": [1.0],
        },
    }]
    monkeypatch.setattr(cli, "extract_rally_segments", lambda _video: segments)
    monkeypatch.setattr(
        cli,
        "enrich_report",
        lambda _video, report, **_options: report,
    )
    analysis_path = tmp_path / "analysis.json"
    internal_dir = tmp_path / "internal"

    result = cli.main([
        "video.mp4",
        "--output",
        str(analysis_path),
        "--internal-output-dir",
        str(internal_dir),
    ])

    assert result == 0
    assert analysis_path.exists()
    assert (internal_dir / "segments.json").exists()
    assert (internal_dir / "report.json").exists()
    assert not (tmp_path / "reports.json").exists()


def test_ball_tracking_survives_court_detection_failure(monkeypatch):
    monkeypatch.setattr(
        cli,
        "TrackNetOnnxDetector",
        lambda _path, **_options: object(),
    )
    monkeypatch.setattr(
        cli,
        "track_ball_intervals",
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


def test_yolox_sports_ball_uses_coco_class_32_score():
    detector = YoloXDetector.__new__(YoloXDetector)
    detector.class_id = COCO_SPORTS_BALL_CLASS_ID
    decoded = np.zeros((2, 85), dtype=np.float32)
    decoded[:, 4] = [0.8, 0.5]
    decoded[:, 5] = [0.99, 0.99]
    decoded[:, 5 + COCO_SPORTS_BALL_CLASS_ID] = [0.25, 0.9]

    np.testing.assert_allclose(detector._class_scores(decoded), [0.2, 0.45])


def test_yolox_ball_detector_returns_highest_confidence_center():
    detector = YoloXBallDetector.__new__(YoloXBallDetector)
    detector.tile_grid = 1
    detector.tile_overlap = 0.15
    detector.detector = type(
        "FakeDetector",
        (),
        {
            "detect": lambda _self, _frame: [
                {"bbox": [10, 10, 20, 20], "confidence": 0.4},
                {"bbox": [40, 20, 60, 40], "confidence": 0.8},
            ]
        },
    )()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    observation = detector.detect([frame, frame, frame])

    assert observation["x"] == 50.0
    assert observation["y"] == 30.0
    assert observation["x_normalized"] == 0.25
    assert observation["confidence"] == 0.8


def test_yolox_ball_tiling_maps_crop_box_to_full_frame():
    detector = YoloXBallDetector.__new__(YoloXBallDetector)
    detector.tile_grid = 2
    detector.tile_overlap = 0.0
    call_count = 0

    class FakeDetector:
        def detect(self, _frame):
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                return [{"bbox": [10, 20, 30, 40], "confidence": 0.8}]
            return []

    detector.detector = FakeDetector()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    observation = detector.detect([frame, frame, frame])

    assert observation["bbox"] == [110, 70, 130, 90]
    assert observation["x"] == 120.0
    assert observation["y"] == 80.0


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


def test_interpolates_short_speed_consistent_ball_gap():
    observations = [
        {
            "frame_index": 0,
            "time": 0.0,
            "visible": True,
            "x": 0.0,
            "y": 0.0,
            "x_normalized": 0.0,
            "y_normalized": 0.0,
            "confidence": 0.8,
            "interpolated": False,
        },
        {
            "frame_index": 1,
            "time": 0.1,
            "visible": False,
            "confidence": 0.0,
            "interpolated": False,
        },
        {
            "frame_index": 2,
            "time": 0.2,
            "visible": True,
            "x": 20.0,
            "y": 10.0,
            "x_normalized": 0.2,
            "y_normalized": 0.1,
            "confidence": 0.6,
            "interpolated": False,
        },
    ]

    result = interpolate_short_gaps(observations, maximum_missing_observations=1)

    assert result[1]["visible"] is True
    assert result[1]["interpolated"] is True
    assert result[1]["x"] == 10.0
    assert result[1]["y_normalized"] == 0.05
    assert result[1]["confidence"] == 0.3


def test_does_not_interpolate_implausibly_fast_ball_gap():
    observations = [
        {
            "time": 0.0,
            "visible": True,
            "x": 0.0,
            "y": 0.0,
            "x_normalized": 0.0,
            "y_normalized": 0.0,
            "confidence": 0.8,
        },
        {"time": 0.1, "visible": False},
        {
            "time": 0.2,
            "visible": True,
            "x": 100.0,
            "y": 100.0,
            "x_normalized": 1.0,
            "y_normalized": 1.0,
            "confidence": 0.8,
        },
    ]

    result = interpolate_short_gaps(
        observations,
        maximum_missing_observations=1,
        maximum_normalized_speed=3.5,
    )

    assert result[1]["visible"] is False


def test_ball_tracking_only_processes_requested_intervals(monkeypatch):
    calls = []

    def fake_track(_video, _detector, **options):
        calls.append((options["start"], options["end"]))
        return [{"time": options["start"]}]

    monkeypatch.setattr(ball_tracking, "track_ball_video", fake_track)

    observations = track_ball_intervals(
        "video.mp4",
        object(),
        [(1.0, 2.0), (10.0, 12.0)],
        frame_step=2,
        temporal_stride=2,
    )

    assert calls == [(1.0, 2.0), (10.0, 12.0)]
    assert observations == [{"time": 1.0}, {"time": 10.0}]


def test_ball_tracking_merges_overlapping_intervals(monkeypatch):
    calls = []

    def fake_track(_video, _detector, **options):
        calls.append((options["start"], options["end"]))
        return []

    monkeypatch.setattr(ball_tracking, "track_ball_video", fake_track)

    track_ball_intervals(
        "video.mp4",
        object(),
        [(10.0, 12.0), (1.0, 3.0), (2.5, 4.0)],
    )

    assert calls == [(1.0, 4.0), (10.0, 12.0)]


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


def test_projects_image_positions_to_normalized_court():
    rois = {
        "far": [[20, 20], [80, 20], [90, 50], [10, 50]],
        "near": [[10, 50], [90, 50], [100, 100], [0, 100]],
    }
    projector = CourtProjector(rois, frame_width=100, frame_height=100)

    assert projector.project_pixel(20, 20) == {
        "x": 0.0,
        "y": 0.0,
        "inside_court": True,
    }
    assert projector.project_pixel(100, 100) == {
        "x": 1.0,
        "y": 1.0,
        "inside_court": True,
    }


def test_adds_court_projections_to_players_and_visible_ball():
    rois = {
        "far": [[0, 0], [100, 0], [100, 50], [0, 50]],
        "near": [[0, 50], [100, 50], [100, 100], [0, 100]],
    }
    projector = CourtProjector(rois, frame_width=100, frame_height=100)
    players = {"player_1": [{"position": [0.5, 0.75]}]}
    ball = [
        {"visible": True, "x": 25, "y": 50},
        {"visible": False},
    ]

    add_court_projections(players, ball, projector)

    assert players["player_1"][0]["court_position"] == {
        "x": 0.5,
        "y": 0.75,
        "inside_court": True,
    }
    assert ball[0]["court_projection"] == {
        "x": 0.25,
        "y": 0.5,
        "inside_court": True,
    }
    assert "court_projection" not in ball[1]


def test_summarizes_ball_trajectory_without_claiming_shot_events():
    observations = [
        {
            "time": 0.0,
            "visible": True,
            "x_normalized": 0.1,
            "y_normalized": 0.2,
            "confidence": 0.9,
            "court_projection": {"x": 0.1, "y": 0.4, "inside_court": True},
        },
        {
            "time": 0.1,
            "visible": True,
            "x_normalized": 0.2,
            "y_normalized": 0.3,
            "confidence": 0.8,
            "court_projection": {"x": 0.2, "y": 0.6, "inside_court": True},
        },
        {"time": 0.2, "visible": False, "confidence": 0.0},
    ]

    summary = summarize_ball_trajectory(observations)

    assert summary["visible_ratio"] == 0.666667
    assert summary["longest_missing_run_observations"] == 1
    assert summary["court_half_crossings"] == 1
    assert summary["direction_change_candidates"] == 0
    assert "hits" not in summary
    assert "bounces" not in summary


def test_builds_compact_llm_statistics_with_capability_limits():
    report = [{
        "index": 1,
        "start": 0.0,
        "end": 1.0,
        "features": {
            "player_motion_max": 0.1,
            "player_motion_var": 0.01,
            "near_motion_mean": 0.02,
            "far_motion_mean": 0.01,
            "motion_sample_count": 15,
        },
        "players": {
            "player_1": {
                "detected": True,
                "side": "near",
                "detection_confidence": 0.9,
                "identity_confidence": 0.8,
                "movement_distance": 0.2,
                "mean_position": [0.5, 0.8],
            },
        },
        "player_trajectories": {
            "player_1": [{
                "side": "near",
                "court_position": {"x": 0.5, "y": 0.8, "inside_court": True},
            }],
        },
    }]

    stats = build_llm_statistics(report)

    assert stats["stats_version"] == 1
    assert stats["schema"]["name"] == "tennis-coach-analysis"
    assert "unsupported" in stats["schema"]["sections"]["analysis_capabilities"]
    assert stats["players"]["player_1"]["segment_detection_rate"] == 1.0
    assert stats["players"]["player_1"]["mean_court_position"] == [0.5, 0.8]
    assert "shot success ratios" in stats["analysis_capabilities"]["unsupported"]
    assert stats["segments"][0]["ball"]["available"] is False


def test_global_ball_summary_does_not_bridge_segments():
    first = summarize_ball_trajectory([{
        "time": 0.0,
        "visible": True,
        "x_normalized": 0.1,
        "y_normalized": 0.1,
        "confidence": 0.9,
    }])
    second = summarize_ball_trajectory([{
        "time": 10.0,
        "visible": True,
        "x_normalized": 0.9,
        "y_normalized": 0.9,
        "confidence": 0.8,
    }])

    summary = aggregate_ball_summaries([first, second])

    assert summary["normalized_image_travel"] == 0.0
    assert summary["speed_sample_count"] == 0
    assert summary["court_half_crossings"] == 0


def test_empty_statistics_do_not_claim_analysis_capabilities():
    stats = build_llm_statistics([])

    assert stats["analysis_capabilities"]["supported"] == []


def test_out_of_court_projection_does_not_count_as_crossing():
    observations = [
        {
            "time": 0.0,
            "visible": True,
            "x_normalized": 0.1,
            "y_normalized": 0.2,
            "confidence": 0.9,
            "court_projection": {"x": -0.2, "y": 0.4, "inside_court": False},
        },
        {
            "time": 0.1,
            "visible": True,
            "x_normalized": 0.2,
            "y_normalized": 0.3,
            "confidence": 0.9,
            "court_projection": {"x": -0.1, "y": 0.6, "inside_court": False},
        },
    ]

    assert summarize_ball_trajectory(observations)["court_half_crossings"] == 0


def test_player_confidence_is_weighted_by_detection_samples():
    report = [
        {
            "start": 0.0,
            "end": 1.0,
            "players": {
                "player_1": {
                    "detected": True,
                    "detection_confidence": 0.2,
                    "identity_confidence": 0.5,
                    "movement_distance": 0.1,
                    "sample_count": 1,
                },
            },
            "player_trajectories": {"player_1": []},
        },
        {
            "start": 1.0,
            "end": 2.0,
            "players": {
                "player_1": {
                    "detected": True,
                    "detection_confidence": 0.8,
                    "identity_confidence": 0.9,
                    "movement_distance": 0.1,
                    "sample_count": 9,
                },
            },
            "player_trajectories": {"player_1": []},
        },
    ]

    player = build_llm_statistics(report)["players"]["player_1"]

    assert player["mean_detection_confidence"] == 0.74
    assert player["mean_identity_confidence"] == 0.86
    assert player["mean_identity_confidence_after_initialization"] == 0.9


def test_single_player_and_segment_do_not_enable_comparisons():
    report = [{
        "start": 0.0,
        "end": 1.0,
        "features": {"motion_sample_count": 10},
        "players": {
            "player_1": {
                "detected": True,
                "detection_confidence": 0.9,
                "identity_confidence": 0.8,
                "movement_distance": 0.2,
                "sample_count": 2,
            },
        },
        "player_trajectories": {
            "player_1": [
                {"court_position": {"x": 0.5, "y": 1.1, "inside_court": False}},
                {"court_position": {"x": 0.6, "y": 1.2, "inside_court": False}},
            ],
        },
    }]

    stats = build_llm_statistics(report)

    assert "player movement comparison" not in stats["analysis_capabilities"]["supported"]
    assert "segment activity comparison" not in stats["analysis_capabilities"]["supported"]
    assert stats["players"]["player_1"]["mean_court_position"] == [0.55, 1.15]
    assert stats["players"]["player_1"]["mean_in_court_position"] is None
    assert stats["players"]["player_1"]["total_court_movement_normalized"] == 0.141421


def test_disconnected_ball_sightings_only_enable_visibility_analysis():
    report = [
        {
            "start": 0.0,
            "end": 1.0,
            "ball_trajectory": [{
                "time": 0.0,
                "visible": True,
                "x_normalized": 0.1,
                "y_normalized": 0.1,
                "confidence": 0.9,
            }],
        },
        {
            "start": 10.0,
            "end": 11.0,
            "ball_trajectory": [{
                "time": 10.0,
                "visible": True,
                "x_normalized": 0.9,
                "y_normalized": 0.9,
                "confidence": 0.8,
            }],
        },
    ]

    supported = build_llm_statistics(report)["analysis_capabilities"]["supported"]

    assert "ball visibility analysis" in supported
    assert "ball image-space trajectory analysis" not in supported


def test_stationary_players_still_enable_movement_comparison():
    player = {
        "detected": True,
        "detection_confidence": 0.9,
        "identity_confidence": 0.8,
        "movement_distance": 0.0,
        "sample_count": 2,
    }
    stationary_points = [
        {"court_position": {"x": 0.5, "y": 0.8, "inside_court": True}},
        {"court_position": {"x": 0.5, "y": 0.8, "inside_court": True}},
    ]
    report = [{
        "start": 0.0,
        "end": 1.0,
        "players": {"player_1": player, "player_2": player},
        "player_trajectories": {
            "player_1": stationary_points,
            "player_2": stationary_points,
        },
    }]

    stats = build_llm_statistics(report)

    assert "player movement comparison" in stats["analysis_capabilities"]["supported"]
    assert stats["players"]["player_1"]["total_court_movement_normalized"] == 0.0
    assert stats["players"]["player_1"]["court_transition_samples"] == 1
