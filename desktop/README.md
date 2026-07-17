# Tennis Coach desktop

The desktop application combines Breakpoint's rally-review workflow with
Tennis Coach's structured video evidence and optional Foundry Local coaching
analysis.

## Development

```bash
npm install
npm run dev
```

In development, the app automatically uses `../.venv` when present. Set
`TENNIS_COACH_PYTHON` when you need a different Python executable. The selected
environment must have this repository and dependencies installed (including
OpenCV/cv2 and `foundry-local` for AI analysis).

```bash
python -m pip install -e ".[foundry-local]"
```

Audio extraction also requires `ffmpeg` to be available. In WSL/Ubuntu:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

```bash
TENNIS_COACH_PYTHON=/path/to/python npm run dev
```

After a video finishes extraction, **AI Analysis** becomes available from the
review workspace. The renderer displays deterministic quality/capability
metadata, while model execution goes through Electron IPC to
`video_extraction.local_analysis`. Users never need to locate the intermediate
JSON, and raw video is not sent to the model.

AI Analysis defaults to **Cloud** mode. Configure cloud execution with:

```bash
TENNIS_COACH_CLOUD_API_BASE=https://api.openai.com/v1
TENNIS_COACH_CLOUD_API_KEY=<your-api-key>
# Optional, defaults to /chat/completions
TENNIS_COACH_CLOUD_API_PATH=/chat/completions
```

Local mode remains available as a fallback.

If extraction fails, Tennis Coach writes a copyable failure log to the video
output folder as `analysis-error.log` (next to `analysis.json`).

## Attribution

The Electron/React rally review application was migrated from
[Breakpoint](https://github.com/xinyiz1226/Breakpoint), created by Xinyi
Zhang and licensed under AGPL-3.0. Tennis Coach remains AGPL-3.0-or-later.
The Apache-2.0 YOLOX-Nano model used for person and sports-ball detection is
bundled with its source, checksum, and license under
`video_extraction/vision/models/` and `third_party/YOLOX/`.
