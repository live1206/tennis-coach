from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SHOT_TRANSPORT_FIELDS = (
    "time",
    "player_id",
    "classification",
    "confidence",
    "reason",
    "contact_confidence",
    "shot_role",
    "role_confidence",
    "role_reason",
    "outcome",
    "outcome_confidence",
    "outcome_reason",
)


def _is_resolved_shot(shot: dict) -> bool:
    return (
        shot.get("player_id") is not None
        or shot.get("classification") in {"forehand", "backhand"}
        or shot.get("outcome") in {"continued", "error"}
    )


def _compact_shot(shot: dict) -> dict:
    return {
        field: shot[field]
        for field in SHOT_TRANSPORT_FIELDS
        if field in shot
    }


def _compact_segment(segment: dict) -> dict:
    shots = [
        shot
        for shot in segment.get("shots", [])
        if isinstance(shot, dict)
    ]
    reasons = Counter(
        str(shot.get("reason") or "classified")
        for shot in shots
    )
    return {
        key: segment.get(key)
        for key in (
            "index",
            "start",
            "end",
            "duration",
            "motion",
            "audio",
            "ball",
            "outcome",
        )
    } | {
        "shots": [
            _compact_shot(shot)
            for shot in shots
            if _is_resolved_shot(shot)
        ],
        "shot_candidate_summary": {
            "total": len(shots),
            "by_reason": dict(reasons),
        },
    }


def _compact_report_payload(parsed: dict, max_segments: int = 12) -> dict:
    segments = parsed.get("segments") if isinstance(parsed.get("segments"), list) else []
    compact_segments = [
        _compact_segment(segment)
        for segment in segments[:max_segments]
        if isinstance(segment, dict)
    ]
    schema = parsed.get("schema")
    compact_schema = (
        {
            "name": schema.get("name"),
            "version": schema.get("version"),
        }
        if isinstance(schema, dict)
        else schema
    )
    compact: dict = {
        "schema": compact_schema,
        "source": parsed.get("source"),
        "data_quality": parsed.get("data_quality"),
        "analysis_capabilities": parsed.get("analysis_capabilities"),
        "target_player": parsed.get("target_player"),
        "players": parsed.get("players"),
        "segments": compact_segments,
        "truncation": {
            "segments_total": len(segments),
            "segments_included": len(compact_segments),
            "note": (
                "Verbose schema documentation, repeated player samples, pose landmarks, "
                "contact coordinates, and unresolved shot rows were omitted for transport. "
                "Unresolved shot totals and reasons are retained in shot_candidate_summary."
            ),
        },
    }
    return compact


def _run_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _resolve_azd_command() -> str:
    executable = shutil.which("azd")
    if executable:
        return executable
    user_install = Path.home() / ".local" / "bin" / "azd"
    if user_install.is_file():
        return str(user_install)
    raise RuntimeError(
        "Azure Developer CLI (azd) is not installed. "
        "Install it from https://aka.ms/install-azd.sh and run `azd auth login`."
    )


def _build_extract_command(args: argparse.Namespace, report_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "video_extraction.cli",
        str(args.video),
        "-o",
        str(report_path),
    ]
    if args.input_report:
        command.insert(4, str(args.input_report))
    if args.segments_output:
        command.extend(["--segments-output", str(args.segments_output)])
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if args.ball_model_path:
        command.extend(["--ball-model-path", str(args.ball_model_path)])
    if args.ball_frame_step != 1:
        command.extend(["--ball-frame-step", str(args.ball_frame_step)])
    if args.ball_temporal_stride != 1:
        command.extend(["--ball-temporal-stride", str(args.ball_temporal_stride)])
    if args.sample_seconds != 0.5:
        command.extend(["--sample-seconds", str(args.sample_seconds)])
    if args.no_sampled_detections:
        command.append("--no-sampled-detections")
    return command


def _load_report_json_for_invoke(report_path: Path, inline_json_max_chars: int) -> str:
    raw_text = report_path.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    compact_payload = _compact_report_payload(parsed)
    compact_json = json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))
    if len(compact_json) > inline_json_max_chars:
        raise ValueError(
            "Report JSON is too large to inline for hosted invoke "
            f"({len(compact_json)} chars > {inline_json_max_chars}). "
            "Increase --inline-json-max-chars or reduce report size."
        )
    return compact_json


def _build_invoke_message(report_json_text: str, analysis_focus: str) -> str:
    focus = analysis_focus.strip()
    if not focus:
        focus = (
            "Summarize the strongest evidence-backed coaching observations and "
            "prioritize the next three areas to practice."
        )

    # Send JSON inline because hosted agents cannot read the caller's local file path.
    return (
        "Call analyze_tennis_technique_from_json_text first with these inputs:\n"
        f"1) analysis_focus: {focus}\n"
        "2) analysis_json_text: the JSON block below exactly as provided\n"
        "After that, follow the instructions returned by the tool.\n\n"
        "```json\n"
        f"{report_json_text}\n"
        "```"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-click pipeline: video extraction -> hosted llm_process agent invoke."
        )
    )
    parser.add_argument("video", type=Path, help="Path to tennis video file")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("analysis.json"),
        help="Output path for canonical analysis JSON",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("analysis.txt"),
        help="Path to save LLM analysis response",
    )
    parser.add_argument(
        "--analysis-focus",
        default="",
        help="Optional technique focus for LLM analysis",
    )
    parser.add_argument(
        "--inline-json-max-chars",
        type=int,
        default=24000,
        help="Maximum number of characters allowed when inlining report JSON into hosted invoke message.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip video extraction and analyze an existing report JSON",
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        default=None,
        help="Optional existing report JSON passed into extraction as the input segments/report",
    )
    parser.add_argument(
        "--segments-output",
        type=Path,
        default=None,
        help="Optional output path for audio-generated rally candidates",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Path to yolox_nano.onnx")
    parser.add_argument(
        "--ball-model-path",
        type=Path,
        default=None,
        help="Path to TrackNet-compatible ONNX model",
    )
    parser.add_argument("--ball-frame-step", type=int, default=1)
    parser.add_argument("--ball-temporal-stride", type=int, default=1)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--no-sampled-detections", action="store_true")
    parser.add_argument(
        "--agent-ready-timeout",
        type=float,
        default=90.0,
        help="Reserved for backward compatibility; not used in hosted invoke mode",
    )
    parser.add_argument(
        "--agent-port",
        type=int,
        default=8088,
        help="Reserved for backward compatibility; not used in hosted invoke mode",
    )
    parser.add_argument(
        "--print-agent-logs",
        action="store_true",
        help="Reserved for backward compatibility; not used in hosted invoke mode",
    )
    parser.add_argument(
        "--model-deployment-name",
        default="",
        help="Optional override for AZURE_AI_MODEL_DEPLOYMENT_NAME",
    )
    parser.add_argument(
        "--foundry-project-endpoint",
        default="",
        help="Reserved for backward compatibility; not used in hosted invoke mode",
    )
    parser.add_argument(
        "--agent-endpoint",
        default=(
            "https://foundry-tennis-001.services.ai.azure.com/api/projects/proj-default/agents/"
            "tennis-coach/endpoint/protocols/openai/responses?api-version=v1"
        ),
        help="Hosted agent endpoint URL used by azd ai agent invoke",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    llm_process_root = repo_root / "llm_process"

    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()

    analysis_output = args.analysis_output
    if not analysis_output.is_absolute():
        analysis_output = (repo_root / analysis_output).resolve()

    video_path = args.video
    if not video_path.is_absolute():
        video_path = (repo_root / video_path).resolve()
    if not video_path.is_file() and not args.skip_extraction:
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not args.skip_extraction:
        extract_args = argparse.Namespace(**vars(args))
        extract_args.video = video_path
        if extract_args.input_report and not extract_args.input_report.is_absolute():
            extract_args.input_report = (repo_root / extract_args.input_report).resolve()
        if extract_args.segments_output and not extract_args.segments_output.is_absolute():
            extract_args.segments_output = (repo_root / extract_args.segments_output).resolve()
        if extract_args.model_path and not extract_args.model_path.is_absolute():
            extract_args.model_path = (repo_root / extract_args.model_path).resolve()
        if extract_args.ball_model_path and not extract_args.ball_model_path.is_absolute():
            extract_args.ball_model_path = (repo_root / extract_args.ball_model_path).resolve()

        report_path.parent.mkdir(parents=True, exist_ok=True)
        extract_command = _build_extract_command(extract_args, report_path)
        print("Step 1/3: running video extraction...")
        _run_command(extract_command, cwd=repo_root)
    elif not report_path.is_file():
        raise FileNotFoundError(
            f"Report file not found while using --skip-extraction: {report_path}"
        )

    if not report_path.is_file():
        raise FileNotFoundError(f"Report file was not generated: {report_path}")

    print("Step 2/3: invoking hosted llm_process agent...")
    report_json_text = _load_report_json_for_invoke(report_path, args.inline_json_max_chars)
    invoke_message = _build_invoke_message(report_json_text, args.analysis_focus)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as temp_file:
        temp_file.write(invoke_message)
        invoke_input_file = temp_file.name

    invoke_command = [
        _resolve_azd_command(),
        "ai",
        "agent",
        "invoke",
        "--new-session",
        "--new-conversation",
        "--agent-endpoint",
        args.agent_endpoint,
        "--input-file",
        invoke_input_file,
    ]
    try:
        completed = subprocess.run(
            invoke_command,
            cwd=str(llm_process_root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        try:
            os.remove(invoke_input_file)
        except OSError:
            pass
    if completed.returncode != 0:
        raise RuntimeError(
            "Invoke failed with exit code "
            f"{completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )

    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis_text = completed.stdout or ""
    analysis_output.write_text(analysis_text, encoding="utf-8")
    print(f"Step 3/3: hosted analysis complete. Analysis saved to: {analysis_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
