# Tennis Coach desktop

The desktop application combines Breakpoint's rally-review workflow with
Tennis Coach's canonical `analysis.json` and optional Foundry Local coaching
analysis.

## Development

```bash
npm install
npm run dev
```

Set `TENNIS_COACH_PYTHON` when the desired Python executable is not available
as `python`/`python3`. The selected environment must have this repository and
the `foundry-local` optional dependency installed.

```bash
TENNIS_COACH_PYTHON=/path/to/python npm run dev
```

The **AI Analysis** entry opens a canonical `tennis-coach-analysis` version 1
JSON document. The renderer displays deterministic quality/capability
metadata, while model execution goes through Electron IPC to
`video_extraction.local_analysis`. Raw video is not sent to the model.

## Attribution

The Electron/React rally review application was migrated from
[Breakpoint](https://github.com/xinyiz1226/Breakpoint), created by Xinyi
Zhang and licensed under AGPL-3.0. Tennis Coach remains AGPL-3.0-or-later.
The Apache-2.0 YOLOX-Nano model used for person and sports-ball detection is
bundled with its source, checksum, and license under
`video_extraction/vision/models/` and `third_party/YOLOX/`.
