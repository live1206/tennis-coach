from __future__ import annotations

import argparse
import json
from pathlib import Path

from tennis_coach.vision.court import select_rois
from tennis_coach.vision.motion import analyze_motion
from tennis_coach.vision.player_observation import analyze_player_observations


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
) -> list[dict]:
    selected_rois = rois if rois is not None else select_rois(str(video_path))
    if selected_rois is None:
        return [
            {
                **segment,
                "video_extraction": {
                    "version": EXTRACTION_VERSION,
                    "status": "skipped_court_detection",
                },
            }
            for segment in report
        ]

    motion_data = analyze_motion(str(video_path), report, selected_rois)
    observation_data = analyze_player_observations(
        str(video_path),
        report,
        selected_rois,
        model_path=model_path,
        sample_seconds=sample_seconds,
        include_sampled_detections=include_sampled_detections,
    )

    enriched = []
    for segment, motion, observation in zip(report, motion_data, observation_data):
        features = {**segment.get("features", {}), **motion}
        enriched.append({
            **segment,
            "features": features,
            "players": observation["players"],
            "video_extraction": {
                "version": EXTRACTION_VERSION,
                "status": "complete",
                "court_rois": selected_rois,
                "sample_seconds": sample_seconds,
            },
            "sampled_frames": observation.get("sampled_frames", []),
        })
    return enriched


def enrich_report_file(
    video_path: str | Path,
    report_path: str | Path,
    output_path: str | Path,
    model_path: str | Path | None = None,
    include_sampled_detections: bool = True,
    sample_seconds: float = 0.5,
) -> list[dict]:
    report = load_report(report_path)
    enriched = enrich_report(
        video_path,
        report,
        model_path=model_path,
        include_sampled_detections=include_sampled_detections,
        sample_seconds=sample_seconds,
    )
    Path(output_path).write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich tennis report JSON with local video-derived signals.")
    parser.add_argument("video", help="Path to the original tennis video")
    parser.add_argument("report", help="Path to existing report JSON with start/end segment fields")
    parser.add_argument("-o", "--output", default="reports.json", help="Output path for enriched reports JSON")
    parser.add_argument("--model-path", default=None, help="Path to yolox_nano.onnx")
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
    )
    print(f"Wrote {len(enriched)} enriched segments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
