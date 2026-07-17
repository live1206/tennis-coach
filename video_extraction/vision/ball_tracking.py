from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


TRACKNET_INPUT_WIDTH = 640
TRACKNET_INPUT_HEIGHT = 360


class TrackNetOnnxDetector:
    """Run an externally supplied TrackNet-compatible ONNX model with OpenCV DNN."""

    def __init__(
        self,
        model_path: str | Path,
        input_width: int = TRACKNET_INPUT_WIDTH,
        input_height: int = TRACKNET_INPUT_HEIGHT,
        class_threshold: int = 128,
        minimum_component_area: int = 3,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TrackNet ONNX model not found: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.input_width = input_width
        self.input_height = input_height
        self.class_threshold = class_threshold
        self.minimum_component_area = minimum_component_area

    def detect(self, frames: list[np.ndarray]) -> dict:
        if len(frames) != 3:
            raise ValueError("TrackNet requires exactly three consecutive frames")
        original_height, original_width = frames[-1].shape[:2]
        resized = [
            cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
        stacked = np.concatenate((resized[2], resized[1], resized[0]), axis=2)
        tensor = np.transpose(stacked.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        self.net.setInput(tensor)
        output = np.asarray(self.net.forward())
        class_map = self._class_map(output)
        detected = self._locate_ball(class_map)
        if detected is None:
            return {"visible": False, "confidence": 0.0}

        x_model, y_model, confidence = detected
        x = x_model * original_width / self.input_width
        y = y_model * original_height / self.input_height
        return {
            "visible": True,
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "x_normalized": round(float(x / max(original_width, 1)), 6),
            "y_normalized": round(float(y / max(original_height, 1)), 6),
            "confidence": round(confidence, 6),
        }

    def _class_map(self, output: np.ndarray) -> np.ndarray:
        squeezed = np.squeeze(output, axis=0)
        if squeezed.ndim == 3:
            if squeezed.shape[0] == 256:
                return np.argmax(squeezed, axis=0).astype(np.uint8)
            if squeezed.shape[-1] == 256:
                return np.argmax(squeezed, axis=-1).astype(np.uint8)
        if squeezed.ndim == 2 and squeezed.shape[0] == 256:
            return np.argmax(squeezed, axis=0).reshape(self.input_height, self.input_width).astype(np.uint8)
        if squeezed.ndim == 2 and squeezed.shape[1] == 256:
            return np.argmax(squeezed, axis=1).reshape(self.input_height, self.input_width).astype(np.uint8)
        raise RuntimeError(f"Unexpected TrackNet output shape: {output.shape}")

    def _locate_ball(self, class_map: np.ndarray) -> tuple[float, float, float] | None:
        mask = np.where(class_map >= self.class_threshold, 255, 0).astype(np.uint8)
        component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < self.minimum_component_area:
                continue
            x, y = centroids[component]
            confidence = float(class_map[_labels == component].max() / 255.0)
            candidates.append((area, float(x), float(y), confidence))
        if not candidates:
            return None
        _area, x, y, confidence = max(candidates, key=lambda candidate: candidate[0])
        return x, y, confidence


def track_ball_video(
    video_path: str | Path,
    detector,
    frame_step: int = 1,
    temporal_stride: int = 1,
    start: float = 0.0,
    end: float | None = None,
) -> list[dict]:
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be at least 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for ball tracking: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    start_frame = max(0, int(start * fps))
    end_frame = None if end is None else int(end * fps)
    read_start_frame = max(0, start_frame - 2 * temporal_stride)
    observations = []
    frame_window: deque[np.ndarray] = deque(maxlen=2 * temporal_stride + 1)
    frame_index = read_start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, read_start_frame)

    try:
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            read, frame = cap.read()
            if not read:
                break
            frame_window.append(frame)
            if (
                len(frame_window) < frame_window.maxlen
                or frame_index < start_frame
                or frame_index % frame_step != 0
            ):
                frame_index += 1
                continue

            window = list(frame_window)
            detection = detector.detect([
                window[0],
                window[temporal_stride],
                window[2 * temporal_stride],
            ])
            observations.append({
                "frame_index": frame_index,
                "time": round(frame_index / fps, 6),
                **detection,
                "interpolated": False,
            })
            frame_index += 1
    finally:
        cap.release()

    return observations


def observations_for_segment(observations: list[dict], start: float, end: float) -> list[dict]:
    return [observation for observation in observations if start <= observation["time"] < end]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track a tennis ball with a TrackNet-compatible ONNX model.")
    parser.add_argument("video", help="Source video path")
    parser.add_argument("model", help="TrackNet-compatible ONNX model path")
    parser.add_argument("-o", "--output", default="ball_trajectory.json", help="Output trajectory JSON path")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--frame-step", type=int, default=1, help="Run inference every Nth frame")
    parser.add_argument("--temporal-stride", type=int, default=1, help="Frame spacing between model inputs")
    args = parser.parse_args(argv)

    detector = TrackNetOnnxDetector(args.model)
    observations = track_ball_video(
        args.video,
        detector,
        frame_step=args.frame_step,
        temporal_stride=args.temporal_stride,
        start=args.start,
        end=args.end,
    )
    output = {
        "source_video": str(Path(args.video).resolve()),
        "model": str(Path(args.model).resolve()),
        "start": args.start,
        "end": args.end,
        "frame_step": args.frame_step,
        "temporal_stride": args.temporal_stride,
        "observations": observations,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {len(observations)} ball observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
