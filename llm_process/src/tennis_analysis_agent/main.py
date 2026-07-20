# Copyright (c) Microsoft. All rights reserved.

import csv
import json
import os
from statistics import mean
from pathlib import Path
from typing import Any

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import Annotated

if __package__:
    from .analysis_prompt import build_analysis_prompt
else:
    from analysis_prompt import build_analysis_prompt

# Load environment variables from .env file
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
_MAIN_FILE_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = _MAIN_FILE_PATH.parents[3] if len(_MAIN_FILE_PATH.parents) > 3 else _MAIN_FILE_PATH.parent


def _resolve_dataset_path(file_name: str) -> str:
    # Keep file access scoped to the local data directory.
    safe_name = os.path.basename(file_name.strip())
    if not safe_name:
        raise ValueError("File name cannot be empty.")
    if not safe_name.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    file_path = os.path.join(DATA_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset '{safe_name}' was not found in the data directory.")
    return file_path


def _resolve_video_json_path(file_name: str) -> str:
    file_ref = file_name.strip()
    if not file_ref:
        raise ValueError("File name cannot be empty.")
    if not file_ref.lower().endswith(".json"):
        raise ValueError("Only JSON files are supported for video analysis input.")

    # First, allow an explicit existing path so reports.json can be consumed
    # directly from the video_extraction output location.
    explicit_path = Path(file_ref).expanduser()
    if explicit_path.is_file():
        return str(explicit_path.resolve())

    workspace_relative = (WORKSPACE_ROOT / explicit_path).resolve()
    if workspace_relative.is_file():
        return str(workspace_relative)

    safe_name = os.path.basename(file_ref)

    file_path = os.path.join(DATA_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Video analysis JSON '{file_ref}' was not found. "
            f"Provide an existing absolute/relative path or place '{safe_name}' in the data directory."
        )
    return file_path


def _parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_segments(parsed: Any) -> list[dict]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        segments = parsed.get("segments")
        if isinstance(segments, list):
            return [item for item in segments if isinstance(item, dict)]
    return []


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _build_player_breakdown(segments: list[dict]) -> dict[str, Any]:
    player_views: dict[str, dict[str, Any]] = {
        "player_1": {"segments": []},
        "player_2": {"segments": []},
    }

    for index, segment in enumerate(segments, start=1):
        players = segment.get("players") if isinstance(segment.get("players"), dict) else {}
        trajectories = (
            segment.get("player_trajectories")
            if isinstance(segment.get("player_trajectories"), dict)
            else {}
        )

        for player_key in ["player_1", "player_2"]:
            player_info = players.get(player_key) if isinstance(players, dict) else None
            trajectory = trajectories.get(player_key) if isinstance(trajectories, dict) else None

            if not isinstance(player_info, dict) and not isinstance(trajectory, list):
                continue

            movement_value = None
            confidence_value = None
            side_value = None
            mean_position = None
            sample_points = 0

            if isinstance(player_info, dict):
                movement = player_info.get("movement_score")
                confidence = player_info.get("confidence")
                movement_value = float(movement) if isinstance(movement, (int, float)) else None
                confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
                side_value = player_info.get("side") if isinstance(player_info.get("side"), str) else None
                mean_pos = player_info.get("mean_position")
                if isinstance(mean_pos, dict):
                    mean_position = {
                        "x": mean_pos.get("x") if isinstance(mean_pos.get("x"), (int, float)) else None,
                        "y": mean_pos.get("y") if isinstance(mean_pos.get("y"), (int, float)) else None,
                    }

            if isinstance(trajectory, list):
                sample_points = len([p for p in trajectory if isinstance(p, dict)])

            player_views[player_key]["segments"].append(
                {
                    "segment_index": index,
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "side": side_value,
                    "confidence": confidence_value,
                    "movement_score": movement_value,
                    "trajectory_points": sample_points,
                    "mean_position": mean_position,
                }
            )

    for player_key in ["player_1", "player_2"]:
        entries = player_views[player_key]["segments"]
        confidence_values = [e["confidence"] for e in entries if isinstance(e.get("confidence"), (int, float))]
        movement_values = [e["movement_score"] for e in entries if isinstance(e.get("movement_score"), (int, float))]
        trajectory_points = [e["trajectory_points"] for e in entries if isinstance(e.get("trajectory_points"), int)]

        side_counts: dict[str, int] = {}
        for e in entries:
            side = e.get("side")
            if isinstance(side, str) and side:
                side_counts[side] = side_counts.get(side, 0) + 1

        player_views[player_key]["summary"] = {
            "segment_count": len(entries),
            "avg_confidence": _avg([float(v) for v in confidence_values]),
            "avg_movement_score": _avg([float(v) for v in movement_values]),
            "avg_trajectory_points": _avg([float(v) for v in trajectory_points]),
            "side_counts": side_counts,
        }

    return {
        "player_1": player_views["player_1"],
        "player_2": player_views["player_2"],
    }


@tool(approval_mode="never_require")
def upload_tennis_dataset(
    file_name: Annotated[str, Field(description="Target CSV file name, e.g. matches.csv")],
    csv_content: Annotated[
        str,
        Field(description="CSV text content. Include header row and data rows."),
    ],
) -> str:
    """Upload tennis data by writing CSV content into the local data folder."""
    safe_name = os.path.basename(file_name.strip())
    if not safe_name:
        return "Upload failed: file name cannot be empty."
    if not safe_name.lower().endswith(".csv"):
        return "Upload failed: only CSV files are supported."

    lines = [line for line in csv_content.splitlines() if line.strip()]
    if len(lines) < 2:
        return "Upload failed: CSV must include a header and at least one data row."

    target_path = os.path.join(DATA_DIR, safe_name)
    with open(target_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(csv_content)

    return (
        f"Uploaded dataset to {safe_name}. "
        f"Detected approximately {max(len(lines) - 1, 0)} data rows."
    )


@tool(approval_mode="never_require")
def list_tennis_datasets() -> str:
    """List CSV datasets currently available in the local data folder."""
    files = sorted([name for name in os.listdir(DATA_DIR) if name.lower().endswith(".csv")])
    if not files:
        return "No CSV datasets found. Upload one with upload_tennis_dataset first."

    lines = ["Available tennis datasets:"]
    lines.extend([f"- {name}" for name in files])
    return "\n".join(lines)


@tool(approval_mode="never_require")
def preview_tennis_dataset(
    file_name: Annotated[str, Field(description="CSV file name in the local data folder")],
    rows: Annotated[int, Field(description="Number of preview rows", ge=1, le=30)] = 8,
) -> str:
    """Preview the first few rows of a tennis CSV dataset."""
    try:
        file_path = _resolve_dataset_path(file_name)
    except (ValueError, FileNotFoundError) as exc:
        return f"Preview failed: {exc}"

    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        sample_rows = []
        for idx, row in enumerate(reader):
            if idx >= rows:
                break
            sample_rows.append(row)

    if not headers:
        return "Preview failed: no header row detected in CSV."
    if not sample_rows:
        return f"Dataset '{os.path.basename(file_path)}' contains headers but no data rows."

    lines = [f"Preview for {os.path.basename(file_path)}", f"Columns: {', '.join(headers)}", "Rows:"]
    for idx, row in enumerate(sample_rows, start=1):
        compact = ", ".join([f"{col}={str(row.get(col, '')).strip()}" for col in headers[:10]])
        lines.append(f"{idx}. {compact}")
    return "\n".join(lines)


@tool(approval_mode="never_require")
def analyze_tennis_dataset(
    file_name: Annotated[str, Field(description="CSV file name in the local data folder")],
    focus: Annotated[
        str,
        Field(description="Optional analysis focus, e.g. 'Nadal on clay' or 'serve performance'"),
    ] = "",
) -> str:
    """Analyze a tennis dataset and return compact descriptive statistics and trends."""
    try:
        file_path = _resolve_dataset_path(file_name)
    except (ValueError, FileNotFoundError) as exc:
        return f"Analysis failed: {exc}"

    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not headers:
        return "Analysis failed: no header row detected in CSV."
    if not rows:
        return "Analysis failed: dataset has no records."

    numeric_columns: dict[str, list[float]] = {header: [] for header in headers}
    categorical_counts: dict[str, dict[str, int]] = {
        key: {} for key in ["winner_name", "loser_name", "surface", "tourney_name"] if key in headers
    }

    for row in rows:
        for header in headers:
            numeric = _parse_float(row.get(header, ""))
            if numeric is not None:
                numeric_columns[header].append(numeric)

        for key in categorical_counts:
            value = str(row.get(key, "")).strip()
            if value:
                bucket = categorical_counts[key]
                bucket[value] = bucket.get(value, 0) + 1

    numeric_summaries = []
    for header, values in numeric_columns.items():
        if len(values) >= max(10, int(len(rows) * 0.3)):
            numeric_summaries.append(
                f"- {header}: avg={mean(values):.2f}, min={min(values):.2f}, max={max(values):.2f}, n={len(values)}"
            )

    lines = [
        f"Tennis dataset analysis for {os.path.basename(file_path)}",
        f"Total rows: {len(rows)}",
        f"Total columns: {len(headers)}",
    ]

    if focus.strip():
        lines.append(f"Requested focus: {focus.strip()}")

    lines.append("Numeric trends:")
    if numeric_summaries:
        lines.extend(numeric_summaries[:12])
    else:
        lines.append("- No sufficiently populated numeric columns found.")

    if categorical_counts:
        lines.append("Frequent categorical values:")
        for key, counts in categorical_counts.items():
            top_values = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
            if not top_values:
                lines.append(f"- {key}: no values")
                continue
            joined = ", ".join([f"{value} ({count})" for value, count in top_values])
            lines.append(f"- {key}: {joined}")

    lines.append(
        "Suggested next prompt: ask for player-level comparison, surface-specific performance, or trend by season."
    )
    return "\n".join(lines)


@tool(approval_mode="never_require")
def analyze_tennis_technique_from_json(
    file_name: Annotated[
        str,
        Field(description="JSON file name or path, e.g. analysis.json or C:/.../analysis.json"),
    ],
    analysis_focus: Annotated[
        str,
        Field(description="Optional focus, e.g. 'forehand biomechanics' or 'serve toss consistency'"),
    ] = "",
) -> str:
    """Load a tennis video analysis JSON file and package it with an expert prompt for technique analysis."""
    try:
        file_path = _resolve_video_json_path(file_name)
    except (ValueError, FileNotFoundError) as exc:
        return f"JSON analysis failed: {exc}"

    with open(file_path, "r", encoding="utf-8") as handle:
        raw_text = handle.read()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return f"JSON analysis failed: invalid JSON format ({exc})."

    question = analysis_focus.strip() or (
        "Summarize the strongest evidence-backed coaching observations and "
        "prioritize the next three areas to practice."
    )
    try:
        return build_analysis_prompt(parsed, question)
    except ValueError as exc:
        return f"JSON analysis failed: {exc}"


@tool(approval_mode="never_require")
def analyze_tennis_technique_from_json_text(
    analysis_json_text: Annotated[
        str,
        Field(description="Tennis analysis JSON text content (stringified JSON object)."),
    ],
    analysis_focus: Annotated[
        str,
        Field(description="Optional focus, e.g. 'forehand biomechanics' or 'serve toss consistency'"),
    ] = "",
) -> str:
    """Analyze tennis technique from raw JSON text, useful when the agent cannot access local file paths."""
    raw_text = analysis_json_text.strip()
    if not raw_text:
        return "JSON analysis failed: analysis_json_text is empty."

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return f"JSON analysis failed: invalid JSON format ({exc})."

    question = analysis_focus.strip() or (
        "Summarize the strongest evidence-backed coaching observations and "
        "prioritize the next three areas to practice."
    )
    try:
        return build_analysis_prompt(parsed, question)
    except ValueError as exc:
        return f"JSON analysis failed: {exc}"


def main():
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )

    agent = Agent(
        client=client,
        instructions=(
            "You are a tennis analytics assistant. "
            "Use tools to inspect uploaded datasets before drawing conclusions. "
            "When the user provides a JSON file path for tennis video analysis, call analyze_tennis_technique_from_json first. "
            "When the user provides JSON content directly, call analyze_tennis_technique_from_json_text first. "
            "and follow the shared evidence and response instructions returned by that tool. "
            "For JSON-based technique reports, always return exactly four sections: "
            "Summary, Best Performance, Biggest Weakness, Improvement Advice. "
            "Each section must be bilingual (English + Chinese). "
            "Each section must include separate conclusions for Player 1 and Player 2. "
            "Then add a `Recommended Coaching Videos` section with 3-5 tailored video suggestions. "
            "Each recommended video must include a direct clickable URL starting with https://. "
            "The report must be athlete-facing coaching feedback, not a data-centric report. "
            "When data is missing, ask for a CSV/JSON upload or a clearer analysis objective."
        ),
        tools=[
            upload_tennis_dataset,
            list_tennis_datasets,
            preview_tennis_dataset,
            analyze_tennis_dataset,
            analyze_tennis_technique_from_json,
            analyze_tennis_technique_from_json_text,
        ],
        # History will be managed by the hosting infrastructure, thus there
        # is no need to store history by the service. Learn more at:
        # https://developers.openai.com/api/reference/resources/responses/methods/create
        default_options={"store": False},
    )

    server = ResponsesHostServer(agent)
    server.run()


if __name__ == "__main__":
    main()
