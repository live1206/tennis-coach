# Tennis Coach

Tennis Coach is a local-first coaching workspace for turning tennis video into structured match intelligence, training feedback, and highlight decisions.

The project builds on the ideas in the sibling [Breakpoint](../Breakpoint) project: detect tennis events from video/audio, segment rallies, rank interesting moments, and export useful clips. Tennis Coach extends that direction from highlight extraction into coaching analysis by routing extracted signals through small/local models or LLM workflows.

![Breakpoint Local Coach user journey and data routing](docs/images/tennis-coach.png)

## Product direction

Tennis Coach starts with raw match or practice footage and extracts structured information that can be analyzed without requiring the model to watch the full video directly.

The intended workflow is:

1. Import a raw match video and an optional player key frame.
2. Run local processing for audio hit detection, rally segmentation, person detection, player re-identification, motion ranking, pose or shot classification, and video clipping.
3. Produce structured outputs such as `report.json` and `stats.json` containing timestamps, density, intensity, player IDs, and shot labels.
4. Let the user ask for coaching insights, such as technical stats, training tips, tactical patterns, or highlight reel requests.
5. Route simple requests to local rules or a small local model.
6. Delegate complex requests to a cloud LLM/agent using structured JSON and optional sampled frames only.
7. Present results in the app as technical stats, training tips, and selected rally IDs for export.

## Planned extraction foundation

The first implementation phase is expected to reuse Breakpoint's YOLO + OpenCV approach for local video understanding, then emit a JSON analysis file for later coaching workflows. The initial target is structured data extraction rather than direct model-driven video interpretation.

Planned JSON signals include:

- rally timestamps and duration
- hit density and intensity metrics
- anonymous player IDs and court-side tracking
- player motion and positioning features
- shot or pose labels when confidence is sufficient
- selected rally IDs for downstream clip export

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
