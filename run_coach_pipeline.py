from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def _drain_output(stream, sink: list[str], print_lines: bool) -> None:
    for line in iter(stream.readline, ""):
        text = line.rstrip("\n")
        sink.append(text)
        if print_lines:
            print(f"[agent] {text}")


def _wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _run_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_value = value.strip().strip('"').strip("'")
        if clean_key:
            values[clean_key] = clean_value
    return values


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


def _build_invoke_message(report_path: Path, analysis_focus: str) -> str:
    focus = analysis_focus.strip()
    if focus:
        return (
            "Use analyze_tennis_technique_from_json with "
            f"file_name='{report_path.as_posix()}' and analysis_focus='{focus}', "
            "then follow the instructions returned by the tool."
        )
    return (
        "Use analyze_tennis_technique_from_json with "
        f"file_name='{report_path.as_posix()}', then follow the instructions returned by the tool."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-click pipeline: video extraction -> local llm_process agent invoke."
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
        help="Seconds to wait for local agent server readiness",
    )
    parser.add_argument(
        "--agent-port",
        type=int,
        default=8088,
        help="Local port used by azd ai agent run",
    )
    parser.add_argument(
        "--print-agent-logs",
        action="store_true",
        help="Print local agent startup logs while waiting",
    )
    parser.add_argument(
        "--model-deployment-name",
        default="",
        help="Optional override for AZURE_AI_MODEL_DEPLOYMENT_NAME",
    )
    parser.add_argument(
        "--foundry-project-endpoint",
        default="",
        help="Optional override for FOUNDRY_PROJECT_ENDPOINT",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    llm_process_root = repo_root / "llm_process"
    agent_env_file = llm_process_root / "src" / "tennis_analysis_agent" / ".env"

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

    agent_env = os.environ.copy()
    agent_env.update(_load_dotenv(agent_env_file))
    if args.model_deployment_name.strip():
        agent_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"] = args.model_deployment_name.strip()
    if args.foundry_project_endpoint.strip():
        agent_env["FOUNDRY_PROJECT_ENDPOINT"] = args.foundry_project_endpoint.strip()

    missing = [
        key
        for key in ["FOUNDRY_PROJECT_ENDPOINT", "AZURE_AI_MODEL_DEPLOYMENT_NAME"]
        if not str(agent_env.get(key, "")).strip()
    ]
    if missing:
        raise ValueError(
            "Missing agent configuration: "
            + ", ".join(missing)
            + ". Set them in llm_process/src/tennis_analysis_agent/.env or pass overrides via "
            "--foundry-project-endpoint / --model-deployment-name."
        )

    print("Step 2/3: starting local llm_process agent...")
    agent_command = ["azd", "ai", "agent", "run"]
    agent_process = subprocess.Popen(
        agent_command,
        cwd=str(llm_process_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=agent_env,
        encoding="utf-8",
        errors="replace",
    )

    output_lines: list[str] = []
    reader_thread = threading.Thread(
        target=_drain_output,
        args=(agent_process.stdout, output_lines, args.print_agent_logs),
        daemon=True,
    )
    reader_thread.start()

    try:
        deadline = time.time() + args.agent_ready_timeout
        ready = False
        while time.time() < deadline:
            if agent_process.poll() is not None:
                break
            if _wait_for_port("127.0.0.1", args.agent_port, 1.0):
                ready = True
                break
        if not ready:
            raise TimeoutError(
                "Timed out waiting for local agent server on "
                f"127.0.0.1:{args.agent_port}. Last logs:\n"
                + "\n".join(output_lines[-20:])
            )

        print("Step 3/3: invoking local agent with analysis.json...")
        invoke_message = _build_invoke_message(report_path, args.analysis_focus)
        invoke_command = ["azd", "ai", "agent", "invoke", "--local", invoke_message]
        completed = subprocess.run(
            invoke_command,
            cwd=str(llm_process_root),
            check=False,
            capture_output=True,
            text=True,
            env=agent_env,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Invoke failed with exit code "
                f"{completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
            )

        analysis_output.parent.mkdir(parents=True, exist_ok=True)
        analysis_text = completed.stdout or ""
        analysis_output.write_text(analysis_text, encoding="utf-8")
        print(f"Pipeline completed. Analysis saved to: {analysis_output}")
        return 0
    finally:
        if agent_process.poll() is None:
            agent_process.terminate()
            try:
                agent_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                agent_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
