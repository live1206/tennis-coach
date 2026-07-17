from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from video_extraction.audio import extract_rally_segments
from video_extraction.court_projection import CourtProjector, add_court_projections
from video_extraction.outcomes import infer_outcomes
from video_extraction.shot_analysis import analyze_segment_shots
from video_extraction.statistics import build_llm_statistics
from video_extraction.target_player import match_target_player
from video_extraction.vision.ball_tracking import (
    TrackNetOnnxDetector,
    observations_for_segment,
    track_ball_intervals,
)
from video_extraction.vision.court import select_rois
from video_extraction.vision.motion import analyze_motion
from video_extraction.vision.player_observation import analyze_player_observations
from video_extraction.vision.yolox_ball import YoloXBallDetector


EXTRACTION_VERSION = 1


def parse_player_handedness(values: list[str]) -> dict[str, str]:
    handedness = {}
    for value in values:
        try:
            player_id, hand = value.split("=", 1)
        except ValueError as error:
            raise ValueError(
                f"Invalid handedness '{value}'; expected player_id=left|right"
            ) from error
        if not player_id or hand not in {"left", "right"}:
            raise ValueError(
                f"Invalid handedness '{value}'; expected player_id=left|right"
            )
        handedness[player_id] = hand
    return handedness


def load_report(path: str | Path) -> list[dict]:
    report_path = Path(path)
    data = json.loads(report_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected report JSON list in {report_path}")
    for index, segment in enumerate(data):
        if not isinstance(segment, dict) or "start" not in segment or "end" not in segment:
            raise ValueError(f"Report segment at index {index} must contain start and end fields")
    return data


def enrich_report(
    video_path: str | Path,
    report: list[dict],
    model_path: str | Path | None = None,
    include_sampled_detections: bool = True,
    sample_seconds: float = 0.5,
    rois: dict | None = None,
    ball_model_path: str | Path | None = None,
    ball_frame_step: int = 1,
    ball_temporal_stride: int = 1,
    ball_detector_type: str = "tracknet",
    inference_backend: str = "opencv",
    ball_tile_grid: int = 1,
) -> list[dict]:
    if not report:
        return []

    ball_observations = None
    if ball_model_path is not None:
        if ball_detector_type == "tracknet":
            ball_detector = TrackNetOnnxDetector(
                ball_model_path,
                inference_backend=inference_backend,
            )
        elif ball_detector_type == "yolox":
            ball_detector = YoloXBallDetector(
                ball_model_path,
                inference_backend=inference_backend,
                tile_grid=ball_tile_grid,
            )
        else:
            raise ValueError(f"Unsupported ball detector: {ball_detector_type}")
        ball_observations = track_ball_intervals(
            video_path,
            ball_detector,
            [
                (float(segment["start"]), float(segment["end"]))
                for segment in report
            ],
            frame_step=ball_frame_step,
            temporal_stride=ball_temporal_stride,
        )

    selected_rois = rois if rois is not None else select_rois(str(video_path))
    if selected_rois is None:
        enriched = []
        for segment in report:
            enriched_segment = {
                **segment,
                "video_extraction": {
                    "version": EXTRACTION_VERSION,
                    "status": "skipped_court_detection",
                    "ball_tracking_status": "complete" if ball_observations is not None else "disabled",
                    "ball_detector": ball_detector_type if ball_observations is not None else None,
                    "inference_backend": inference_backend,
                    "ball_tile_grid": ball_tile_grid if ball_detector_type == "yolox" else None,
                    "ball_frame_step": ball_frame_step if ball_observations is not None else None,
                    "ball_temporal_stride": ball_temporal_stride if ball_observations is not None else None,
                },
            }
            if ball_observations is not None:
                enriched_segment["ball_trajectory"] = observations_for_segment(
                    ball_observations,
                    float(segment["start"]),
                    float(segment["end"]),
                )
            enriched.append(enriched_segment)
        return enriched

    motion_data = analyze_motion(str(video_path), report, selected_rois)
    observation_data = analyze_player_observations(
        str(video_path),
        report,
        selected_rois,
        model_path=model_path,
        sample_seconds=sample_seconds,
        include_sampled_detections=include_sampled_detections,
        inference_backend=inference_backend,
    )
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    projector = CourtProjector(selected_rois, frame_width, frame_height)
    for observation in observation_data:
        add_court_projections(
            observation["player_trajectories"],
            None,
            projector,
        )
    if ball_observations is not None:
        add_court_projections({}, ball_observations, projector)
    enriched = []
    for segment, motion, observation in zip(report, motion_data, observation_data):
        features = {**segment.get("features", {}), **motion}
        enriched_segment = {
            **segment,
            "features": features,
            "players": observation["players"],
            "video_extraction": {
                "version": EXTRACTION_VERSION,
                "status": "complete",
                "court_rois": selected_rois,
                "sample_seconds": sample_seconds,
                "ball_tracking_status": "complete" if ball_observations is not None else "disabled",
                "ball_detector": ball_detector_type if ball_observations is not None else None,
                "inference_backend": inference_backend,
                "ball_tile_grid": ball_tile_grid if ball_detector_type == "yolox" else None,
                "ball_frame_step": ball_frame_step if ball_observations is not None else None,
                "ball_temporal_stride": ball_temporal_stride if ball_observations is not None else None,
            },
            "player_trajectories": observation["player_trajectories"],
            "sampled_frames": observation.get("sampled_frames", []),
        }
        if ball_observations is not None:
            enriched_segment["ball_trajectory"] = observations_for_segment(
                ball_observations,
                float(segment["start"]),
                float(segment["end"]),
            )
        enriched.append(enriched_segment)
    return enriched


def enrich_report_file(
    video_path: str | Path,
    report_path: str | Path,
    output_path: str | Path,
    model_path: str | Path | None = None,
    include_sampled_detections: bool = True,
    sample_seconds: float = 0.5,
    ball_model_path: str | Path | None = None,
    ball_frame_step: int = 1,
    ball_temporal_stride: int = 1,
    ball_detector_type: str = "tracknet",
    inference_backend: str = "opencv",
    ball_tile_grid: int = 1,
) -> list[dict]:
    report = load_report(report_path)
    enriched = enrich_report(
        video_path,
        report,
        model_path=model_path,
        include_sampled_detections=include_sampled_detections,
        sample_seconds=sample_seconds,
        ball_model_path=ball_model_path,
        ball_frame_step=ball_frame_step,
        ball_temporal_stride=ball_temporal_stride,
        ball_detector_type=ball_detector_type,
        inference_backend=inference_backend,
        ball_tile_grid=ball_tile_grid,
    )
    Path(output_path).write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract local tennis video signals into LLM-ready analysis."
    )
    parser.add_argument("video", help="Path to the original tennis video")
    parser.add_argument(
        "report",
        nargs="?",
        help="Optional report JSON; audio-derived rally candidates are generated when omitted",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="analysis.json",
        help="Output path for consolidated LLM-ready analysis JSON",
    )
    parser.add_argument(
        "--internal-output-dir",
        default=None,
        help="Optionally save internal segments.json and report.json for debugging",
    )
    parser.add_argument("--model-path", default=None, help="Path to yolox_nano.onnx")
    parser.add_argument(
        "--ball-model-path",
        default=None,
        help="Path to the selected ball detector's ONNX model",
    )
    parser.add_argument(
        "--ball-detector",
        choices=("tracknet", "yolox"),
        default="tracknet",
        help="Ball detector architecture for --ball-model-path",
    )
    parser.add_argument(
        "--inference-backend",
        choices=("opencv", "cuda"),
        default="opencv",
        help="ONNX inference backend; cuda requires the gpu optional dependency",
    )
    parser.add_argument(
        "--ball-tile-grid",
        type=int,
        default=1,
        help="Split frames into an NxN overlapping grid for higher-resolution YOLOX ball detection",
    )
    parser.add_argument(
        "--pose-model-path",
        default=None,
        help="Optional MediaPipe Pose Landmarker model for stroke-side analysis",
    )
    parser.add_argument(
        "--target-player-image",
        default=None,
        help="Optional cropped key frame used to bind target_player to an anonymous ID",
    )
    parser.add_argument(
        "--player-handedness",
        action="append",
        default=[],
        metavar="PLAYER_ID=HAND",
        help="Player handedness used for forehand/backhand classification",
    )
    parser.add_argument("--ball-frame-step", type=int, default=1, help="Run ball tracking every Nth video frame")
    parser.add_argument(
        "--ball-temporal-stride",
        type=int,
        default=1,
        help="Frame spacing between TrackNet inputs; use 2 for 60 fps footage trained at 30 fps",
    )
    parser.add_argument("--sample-seconds", type=float, default=0.5, help="Seconds between sampled YOLO frames")
    parser.add_argument(
        "--no-sampled-detections",
        action="store_true",
        help="Do not include per-sampled-frame YOLO boxes in output JSON",
    )
    args = parser.parse_args(argv)

    common_options = {
        "model_path": args.model_path,
        "include_sampled_detections": not args.no_sampled_detections,
        "sample_seconds": args.sample_seconds,
        "ball_model_path": args.ball_model_path,
        "ball_frame_step": args.ball_frame_step,
        "ball_temporal_stride": args.ball_temporal_stride,
        "ball_detector_type": args.ball_detector,
        "inference_backend": args.inference_backend,
        "ball_tile_grid": args.ball_tile_grid,
    }
    if args.report:
        report = load_report(args.report)
    else:
        report = extract_rally_segments(args.video)
        if not report:
            parser.error("Audio analysis found no rally candidates")
    enriched = enrich_report(args.video, report, **common_options)
    if args.target_player_image:
        enriched, _target_match = match_target_player(
            args.video,
            enriched,
            args.target_player_image,
        )
    if args.pose_model_path:
        try:
            player_handedness = parse_player_handedness(args.player_handedness)
        except ValueError as error:
            parser.error(str(error))
        enriched = analyze_segment_shots(
            args.video,
            enriched,
            args.pose_model_path,
            player_handedness,
        )
    enriched = infer_outcomes(enriched)
    analysis = build_llm_statistics(enriched)
    Path(args.output).write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False)
    )
    if args.internal_output_dir:
        internal_dir = Path(args.internal_output_dir)
        internal_dir.mkdir(parents=True, exist_ok=True)
        if not args.report:
            (internal_dir / "segments.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False)
            )
        (internal_dir / "report.json").write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False)
        )
    print(f"Wrote analysis for {len(enriched)} segments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
