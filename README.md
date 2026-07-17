# Tennis Coach

Tennis Coach is a local-first coaching workspace for turning tennis video into structured match intelligence, training feedback, and highlight decisions.

The project builds on the ideas in the sibling [Breakpoint](../Breakpoint) project: detect tennis events from video/audio, segment rallies, rank interesting moments, and export useful clips. Tennis Coach extends that direction from highlight extraction into coaching analysis by routing extracted signals through small/local models or LLM workflows.

![Breakpoint Local Coach user journey and data routing](docs/images/tennis-coach.png)

## Product direction

Tennis Coach starts with raw match or practice footage and extracts structured information that can be analyzed without requiring the model to watch the full video directly.

The intended workflow is:

1. Import a raw match video and an optional player key frame.
2. Run local processing for audio hit detection, rally segmentation, person detection, player re-identification, motion ranking, pose or shot classification, and video clipping.
3. Produce one self-describing `analysis.json` containing timestamps, density, intensity, player IDs, and supported analysis signals.
4. Let the user ask for coaching insights, such as technical stats, training tips, tactical patterns, or highlight reel requests.
5. Route simple requests to local rules or a small local model.
6. Delegate complex requests to a cloud LLM/agent using structured JSON and optional sampled frames only.
7. Present results in the app as technical stats, training tips, and selected rally IDs for export.

## Planned extraction foundation

The first implementation phase reuses the focused Breakpoint YOLO + OpenCV approach for local video understanding, then emits a JSON analysis file for later coaching workflows. The initial target is structured data extraction rather than direct model-driven video interpretation.

Planned JSON signals include:

- rally timestamps and duration
- hit density and intensity metrics
- anonymous player IDs and court-side tracking
- player motion and positioning features
- shot or pose labels when confidence is sufficient
- selected rally IDs for downstream clip export

## Video extraction

Install the Python package and ensure `ffmpeg` is available on `PATH`:

```bash
python -m pip install -e .
```

By default, Tennis Coach extracts mono audio with ffmpeg, detects hit
candidates, groups them into candidate rallies, and enriches those intervals
with video-derived data:

```bash
tennis-coach-extract match.mp4 \
  --model-path /path/to/yolox_nano.onnx \
  --output analysis.json
```

`analysis.json` is the only output intended for an LLM. Generated rally
segments and the detailed extraction report remain in memory unless internal
debugging output is requested:

```bash
tennis-coach-extract match.mp4 \
  --output analysis.json \
  --internal-output-dir artifacts/internal
```

An existing JSON list containing `start` and `end` fields can still be supplied
as the second positional argument to bypass automatic segmentation.

The optional `artifacts/internal/report.json` includes:

- `audio` with absolute hit times, onset energies, sample rate, and hit count
- `features.player_motion_max`, `features.player_motion_var`
- `features.near_motion_mean`, `features.far_motion_mean`, `features.motion_sample_count`
- `players.player_1` and `players.player_2` with anonymous side, confidence, movement, and normalized mean position
- `player_trajectories.player_1` and `player_trajectories.player_2` grouped by stable anonymous identity
- `video_extraction.status`, `video_extraction.version`, `video_extraction.court_rois`, and `video_extraction.sample_seconds`
- `sampled_frames` with person boxes, confidence, court side, primary-player status, `player_id`, and identity confidence

If court detection fails, each segment is preserved and marked with `video_extraction.status: "skipped_court_detection"`.

The report schema is still pre-release. Its version remains `1` while the
fields are being designed; version increments begin after the first published
schema release.

### LLM-ready statistics

The extraction command generates compact deterministic statistics directly.
For an existing internal report, the standalone conversion command remains
available:

```bash
tennis-coach-stats artifacts/internal/report.json --output analysis.json
```

`analysis.json` contains per-player movement, identity quality, side usage, and
mean court position; compact per-segment audio, motion, and ball summaries;
global data quality warnings; and explicit supported/unsupported analysis capabilities.
It intentionally does not claim forehand/backhand, shot success, winners, or
errors until those signals are implemented and validated.

The top-level `schema` section explains field meanings, coordinate systems,
units, confidence ranges, candidate-event semantics, and mandatory
limitations so an LLM can interpret the evidence without external
documentation. `examples/analysis.json` is the canonical LLM input example;
detailed and legacy artifacts are isolated under `examples/internal/`.

The output shape is:

```json
{
  "schema": {
    "name": "tennis-coach-analysis",
    "version": 1,
    "purpose": "Deterministic tennis-video evidence for LLM coaching analysis.",
    "conventions": {
      "time": "Seconds from the start of the source video.",
      "confidence": "Values range from 0 to 1; higher means more reliable.",
      "candidate": "A heuristic observation that is not a validated semantic event."
    },
    "sections": {
      "players": {
        "mean_detection_confidence": "Sample-weighted YOLOX person confidence.",
        "mean_court_position": "Mean projected court [x, y]."
      },
      "segments": {
        "audio": {
          "hit_times": "Absolute audio onset times, not validated racket contacts."
        },
        "ball": {
          "visible_ratio": "Visible detections divided by ball observations."
        }
      }
    }
  },
  "source": {"segment_count": 7, "start": 1.31, "end": 156.67},
  "data_quality": {"warnings": ["Ball tracking is unavailable in this report."]},
  "analysis_capabilities": {
    "supported": ["player movement comparison"],
    "unsupported": ["forehand/backhand classification"]
  },
  "players": {
    "player_1": {
      "segments_detected": 7,
      "mean_detection_confidence": 0.816,
      "mean_court_position": [0.487, 0.987],
      "shot_counts": {"forehand": 0, "backhand": 0, "unknown": 0}
    }
  },
  "segments": [
    {
      "index": 1,
      "shots": [
        {
          "time": 2.81,
          "player_id": "player_1",
          "classification": "forehand",
          "confidence": 0.78,
          "reason": null
        }
      ]
    }
  ]
}
```

The embedded `schema` in the actual file is more complete than this abbreviated
README sample. The LLM must honor `analysis_capabilities.unsupported` and
`data_quality.warnings`; absent fields must not be inferred.

### Forehand/backhand classification

Forehand/backhand analysis is confidence-gated and depends on all of:

- a ball observation near an audio hit candidate;
- an unambiguous nearby player box;
- usable shoulder pose from an externally supplied MediaPipe Pose Landmarker;
- declared player handedness.

Install the optional pose dependency and run the complete publishable path:

```bash
python -m pip install -e '.[gpu,pose]'

tennis-coach-extract match.mp4 \
  --model-path /path/to/yolox_nano.onnx \
  --ball-detector yolox \
  --ball-model-path /path/to/yolox_nano.onnx \
  --inference-backend cuda \
  --pose-model-path /path/to/pose_landmarker_heavy.task \
  --player-handedness player_1=right \
  --player-handedness player_2=right \
  --output analysis.json \
  --internal-output-dir artifacts/internal
```

The classifier associates each audio onset with a nearby ball and anonymous
player, projects the contact onto the anatomical shoulder axis, and maps the
contact side using declared handedness. Ambiguous contacts, missing balls,
weak player identity, weak poses, and unknown handedness produce
`classification: "unknown"` with a reason rather than a forced label. Pose
model assets are not bundled; users must obtain and review their upstream
terms separately.

### Ball annotation and tracking

Create a local frame-level annotation set from footage you have the right to use:

```bash
tennis-coach-ball-frames match.mp4 annotations.local/session-01 \
  --start 30 --end 40 --stride 1
```

This writes source frames and `annotations.json`. Label each frame as
`visible`, `occluded`, or `absent`; visible balls require pixel `x`/`y`
coordinates, and optional events are `hit`, `bounce`, or `net`.

```bash
tennis-coach-ball-annotate annotations.local/session-01/annotations.json
```

The annotation window uses left click for a visible ball; `o` for occluded,
`x` for absent, `h`/`b`/`n` for hit/bounce/net, `a`/`d` to navigate, and `q`
to save and exit. Visibility actions save and advance automatically.

```bash
tennis-coach-ball-validate annotations.local/session-01/annotations.json \
  --require-complete
```

The publishable ball baseline uses the Apache-2.0 YOLOX-Nano COCO `sports
ball` class. A CUDA runner can install the optional GPU dependency and run:

```bash
python -m pip install -e '.[gpu]'

tennis-coach-extract match.mp4 \
  --model-path /path/to/yolox_nano.onnx \
  --ball-detector yolox \
  --ball-model-path /path/to/yolox_nano.onnx \
  --inference-backend cuda \
  --output analysis.json \
  --internal-output-dir artifacts/internal
```

YOLOX is expected to be less accurate than a temporal model for tiny, blurred,
or occluded balls, so its output remains confidence-gated and must be measured
on independent labels. The available TrackNet V1 checkpoint has no verified
redistribution license and is not part of the publishable workflow.

Standalone tracking and evaluation are also available:

```bash
tennis-coach-ball-track match.mp4 /path/to/tracknet.onnx \
  --start 30 --end 40 --temporal-stride 2 \
  --output ball_trajectory.json

tennis-coach-ball-evaluate annotations.local/session-01/annotations.json \
  ball_trajectory.json --tolerance-pixels 10
```

The initial independent smoke set under `validation/baseline-30/` contains 30
manually labeled frames sampled from a two-second 1080p60 clip. The private
TrackNet V1 baseline detected 28/30 balls within 10 pixels: recall `0.9333`,
F1 `0.9655`, and mean localization error `4.33 px`. All frames contain a
visible ball, so this small set does not measure false-positive behavior and
is not a release-quality benchmark.

### Sample output

`examples/internal/legacy_report.json` was generated from Breakpoint's
`video/DJI_20260503154223_0534_D_highlight.MP4` using its bundled
`yolox_nano.onnx` model. `examples/internal/legacy_segments.json` divides the full
156.7-second video into five fixed windows solely to exercise report
enrichment; those windows are not detected rally boundaries and are not
required by the normal automatic extraction flow.

The sample contains court polygons, motion summaries, anonymous player
summaries, and sampled person detections from the complete video.

## Privacy and data routing

The default principle is local-first processing:

- Raw video stays on the user's device.
- Deterministic computer vision and small-model processing runs locally where possible.
- Structured JSON may be used for AI analysis.
- Complex cloud analysis should receive only structured signals and optional sampled frames, never the raw video.
- Final video clips are cut locally from the original source video.

## Relationship to Breakpoint

Breakpoint provides a strong foundation for:

- audio-based hit detection
- rally segmentation
- vision-based player motion ranking
- player identity tracking
- ffmpeg-based clip export
- Electron desktop app packaging

Tennis Coach should reuse those concepts while evolving the product goal from automatic highlight extraction to actionable coaching intelligence.

## Initial repository scope

This repository currently documents the project concept and data-routing model. Implementation code, model choices, and app architecture will be added after the initial product direction is agreed.

## Open Source License and Commercial Licensing (License)

This project follows Breakpoint's licensing model and is released under the **GNU Affero General Public License v3 (AGPL-3.0)**.

- **Personal / coach / research use**: free of charge. You may freely deploy, modify, and use this project for personal match review or teaching.
- **Cloud service and commercial use**: if you integrate this project's core algorithms, including tennis target detection, rally segmentation, video-derived JSON extraction, coaching analysis, or automatic highlight editing logic, into a commercial SaaS, mini-program, commercial app, or paid website backend service, AGPL-3.0 requires you to open-source the complete source code of that system under compatible terms.
- **Commercial License**: if you do not want to open-source your system code but would like to use Tennis Coach technology in commercial products, contact the author for a commercial license.

Future bundled YOLOX-Nano model files should preserve their upstream Apache License 2.0 attribution and model source documentation, following Breakpoint's `MODEL_INFO.txt` pattern.
