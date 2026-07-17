from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import cv2


VISIBILITY_VALUES = {"unlabeled", "visible", "occluded", "absent"}
EVENT_VALUES = {None, "hit", "bounce", "net"}
ANNOTATION_VERSION = 1


def create_annotation_manifest(
    video_path: str | Path,
    output_dir: str | Path,
    start: float = 0.0,
    end: float | None = None,
    stride: int = 1,
    image_extension: str = ".jpg",
) -> dict:
    if stride < 1:
        raise ValueError("stride must be at least 1")

    video = Path(video_path)
    destination = Path(output_dir)
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for annotation extraction: {video}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = max(0, int(start * fps))
        end_frame = frame_count if end is None else min(frame_count, int(end * fps))
        if end_frame <= start_frame:
            raise ValueError("end must be greater than start and overlap the video")

        annotations = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        while frame_index < end_frame:
            read, frame = cap.read()
            if not read:
                break
            if (frame_index - start_frame) % stride == 0:
                image_name = f"{frame_index:08d}{image_extension}"
                image_path = frames_dir / image_name
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Failed to write annotation frame: {image_path}")
                annotations.append({
                    "frame_index": frame_index,
                    "time": round(frame_index / fps, 6),
                    "image": str(Path("frames") / image_name),
                    "visibility": "unlabeled",
                    "x": None,
                    "y": None,
                    "x_normalized": None,
                    "y_normalized": None,
                    "event": None,
                })
            frame_index += 1
    finally:
        cap.release()

    manifest = {
        "annotation_version": ANNOTATION_VERSION,
        "source_video": video.name,
        "source_video_path": str(video.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "stride": stride,
        "frames": annotations,
    }
    manifest_path = destination / "annotations.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def validate_annotation_manifest(manifest: dict, require_complete: bool = False) -> list[str]:
    errors = []
    width = manifest.get("width")
    height = manifest.get("height")
    valid_width = isinstance(width, int) and width > 0
    valid_height = isinstance(height, int) and height > 0
    if not valid_width:
        errors.append("width must be a positive integer")
    if not valid_height:
        errors.append("height must be a positive integer")

    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return errors + ["frames must be a list"]

    seen_indices = set()
    for offset, frame in enumerate(frames):
        prefix = f"frames[{offset}]"
        frame_index = frame.get("frame_index")
        if not isinstance(frame_index, int) or frame_index < 0:
            errors.append(f"{prefix}.frame_index must be a non-negative integer")
        elif frame_index in seen_indices:
            errors.append(f"{prefix}.frame_index is duplicated")
        else:
            seen_indices.add(frame_index)

        visibility = frame.get("visibility")
        if visibility not in VISIBILITY_VALUES:
            errors.append(f"{prefix}.visibility must be one of {sorted(VISIBILITY_VALUES)}")
            continue
        if require_complete and visibility == "unlabeled":
            errors.append(f"{prefix}.visibility is still unlabeled")

        event = frame.get("event")
        if event not in EVENT_VALUES:
            errors.append(f"{prefix}.event must be hit, bounce, net, or null")

        x = frame.get("x")
        y = frame.get("y")
        if visibility == "visible":
            if not isinstance(x, (int, float)) or (valid_width and not 0 <= x < width):
                errors.append(f"{prefix}.x must be within the frame for a visible ball")
            if not isinstance(y, (int, float)) or (valid_height and not 0 <= y < height):
                errors.append(f"{prefix}.y must be within the frame for a visible ball")
        elif x is not None or y is not None:
            errors.append(f"{prefix}.x and y must be null unless visibility is visible")

    return errors


def validate_annotation_file(path: str | Path, require_complete: bool = False) -> list[str]:
    manifest = json.loads(Path(path).read_text())
    return validate_annotation_manifest(manifest, require_complete=require_complete)


def evaluate_predictions(
    manifest: dict,
    observations: list[dict],
    tolerance_pixels: float = 10.0,
) -> dict:
    predictions = {item["frame_index"]: item for item in observations}
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    localization_errors = []
    evaluated = 0

    for frame in manifest["frames"]:
        if frame["visibility"] == "unlabeled":
            continue
        evaluated += 1
        prediction = predictions.get(frame["frame_index"], {"visible": False})
        predicted_visible = bool(prediction.get("visible"))
        expected_visible = frame["visibility"] == "visible"

        if expected_visible and predicted_visible:
            error = math.hypot(prediction["x"] - frame["x"], prediction["y"] - frame["y"])
            localization_errors.append(error)
            if error <= tolerance_pixels:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1
        elif expected_visible:
            false_negative += 1
        elif predicted_visible:
            false_positive += 1
        else:
            true_negative += 1

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluated_frames": evaluated,
        "tolerance_pixels": tolerance_pixels,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_localization_error": (
            round(statistics.fmean(localization_errors), 6) if localization_errors else None
        ),
        "median_localization_error": (
            round(statistics.median(localization_errors), 6) if localization_errors else None
        ),
    }


def annotate_manifest(path: str | Path) -> None:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["frames"]
    if not frames:
        raise ValueError("Annotation manifest contains no frames")

    index = 0
    window_name = "Tennis ball annotation"
    display_scale = min(1.0, 1600 / manifest["width"], 900 / manifest["height"])
    display_size = (
        round(manifest["width"] * display_scale),
        round(manifest["height"] * display_scale),
    )
    state = {"advance": False, "redraw": True}

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    def mark_visible(event, x, y, _flags, _parameter) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        source_x = min(manifest["width"] - 1, max(0, round(x / display_scale)))
        source_y = min(manifest["height"] - 1, max(0, round(y / display_scale)))
        frame = frames[index]
        frame["visibility"] = "visible"
        frame["x"] = source_x
        frame["y"] = source_y
        frame["x_normalized"] = round(source_x / manifest["width"], 6)
        frame["y_normalized"] = round(source_y / manifest["height"], 6)
        save()
        state["advance"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mark_visible)
    try:
        while True:
            if state["advance"]:
                index = min(len(frames) - 1, index + 1)
                state["advance"] = False
                state["redraw"] = True

            frame = frames[index]
            if state["redraw"]:
                image_path = manifest_path.parent / frame["image"]
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Cannot read annotation image: {image_path}")
                image = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
                if frame["visibility"] == "visible":
                    display_point = (
                        round(frame["x"] * display_scale),
                        round(frame["y"] * display_scale),
                    )
                    cv2.circle(image, display_point, 12, (0, 0, 255), 2)
                status = (
                    f"{index + 1}/{len(frames)} frame={frame['frame_index']} "
                    f"visibility={frame['visibility']} event={frame['event'] or '-'}"
                )
                cv2.putText(image, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow(window_name, image)
                state["redraw"] = False

            key = cv2.waitKey(50)
            if key == -1:
                continue
            key &= 0xFF

            if key in (ord("q"), 27):
                save()
                break
            if key in (ord("d"), 83):
                index = min(len(frames) - 1, index + 1)
                state["redraw"] = True
                continue
            if key in (ord("a"), 81):
                index = max(0, index - 1)
                state["redraw"] = True
                continue

            if key == ord("o"):
                frame.update({
                    "visibility": "occluded",
                    "x": None,
                    "y": None,
                    "x_normalized": None,
                    "y_normalized": None,
                })
            elif key == ord("x"):
                frame.update({
                    "visibility": "absent",
                    "x": None,
                    "y": None,
                    "x_normalized": None,
                    "y_normalized": None,
                })
            elif key == ord("u"):
                frame.update({
                    "visibility": "unlabeled",
                    "x": None,
                    "y": None,
                    "x_normalized": None,
                    "y_normalized": None,
                    "event": None,
                })
            elif key == ord("h"):
                frame["event"] = "hit"
            elif key == ord("b"):
                frame["event"] = "bounce"
            elif key == ord("n"):
                frame["event"] = "net"
            elif key == ord("e"):
                frame["event"] = None
            else:
                continue
            save()
            state["redraw"] = True
            if key in (ord("o"), ord("x")):
                state["advance"] = True
    finally:
        cv2.destroyWindow(window_name)


def extract_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frames and create a tennis-ball annotation manifest.")
    parser.add_argument("video", help="Source video path")
    parser.add_argument("output_dir", help="Directory for frames and annotations.json")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--stride", type=int, default=1, help="Extract every Nth frame")
    args = parser.parse_args(argv)

    manifest = create_annotation_manifest(
        args.video,
        args.output_dir,
        start=args.start,
        end=args.end,
        stride=args.stride,
    )
    print(f"Wrote {len(manifest['frames'])} frames to {args.output_dir}")
    return 0


def annotate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactively label tennis-ball annotation frames.")
    parser.add_argument("manifest", help="Path to annotations.json")
    args = parser.parse_args(argv)
    print("Left click: visible ball | o: occluded | x: absent | u: unlabeled")
    print("h: hit | b: bounce | n: net | e: clear event | a/d: previous/next | q: save and quit")
    annotate_manifest(args.manifest)
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a tennis-ball annotation manifest.")
    parser.add_argument("manifest", help="Path to annotations.json")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any frame is unlabeled")
    args = parser.parse_args(argv)

    errors = validate_annotation_file(args.manifest, require_complete=args.require_complete)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Annotation manifest is valid")
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ball trajectory predictions against annotations.")
    parser.add_argument("manifest", help="Path to completed annotations.json")
    parser.add_argument("predictions", help="Path to ball trajectory JSON")
    parser.add_argument("--tolerance-pixels", type=float, default=10.0)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    errors = validate_annotation_manifest(manifest, require_complete=True)
    if errors:
        for error in errors:
            print(error)
        return 1
    prediction_data = json.loads(Path(args.predictions).read_text())
    observations = prediction_data.get("observations", prediction_data)
    metrics = evaluate_predictions(manifest, observations, tolerance_pixels=args.tolerance_pixels)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(extract_main())
