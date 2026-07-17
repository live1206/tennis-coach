from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from video_extraction.court_projection import CourtProjector, add_court_projections
from video_extraction.vision.ball_tracking import (
    TrackNetOnnxDetector,
    observations_for_segment,
    track_ball_video,
)
from video_extraction.vision.court import select_rois
from video_extraction.vision.motion import analyze_motion
from video_extraction.vision.player_observation import analyze_player_observations


EXTRACTION_VERSION = 1


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
) -> list[dict]:
    if not report:
        return []

    ball_observations = None
    if ball_model_path is not None:
        ball_detector = TrackNetOnnxDetector(ball_model_path)
        ball_observations = track_ball_video(
            video_path,
            ball_detector,
            frame_step=ball_frame_step,
            temporal_stride=ball_temporal_stride,
            start=min(float(segment["start"]) for segment in report),
            end=max(float(segment["end"]) for segment in report),
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
    )
    Path(output_path).write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich tennis report JSON with local video-derived signals.")
    parser.add_argument("video", help="Path to the original tennis video")
    parser.add_argument("report", help="Path to existing report JSON with start/end segment fields")
    parser.add_argument("-o", "--output", default="reports.json", help="Output path for enriched reports JSON")
    parser.add_argument("--model-path", default=None, help="Path to yolox_nano.onnx")
    parser.add_argument("--ball-model-path", default=None, help="Path to a TrackNet-compatible ONNX model")
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

    enriched = enrich_report_file(
        args.video,
        args.report,
        args.output,
        model_path=args.model_path,
        include_sampled_detections=not args.no_sampled_detections,
        sample_seconds=args.sample_seconds,
        ball_model_path=args.ball_model_path,
        ball_frame_step=args.ball_frame_step,
        ball_temporal_stride=args.ball_temporal_stride,
    )
    print(f"Wrote {len(enriched)} enriched segments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
