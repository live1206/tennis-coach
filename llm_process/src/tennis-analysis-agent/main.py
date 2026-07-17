# Copyright (c) Microsoft. All rights reserved.

import csv
import json
import os
from statistics import mean
from pathlib import Path

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import Annotated

# Load environment variables from .env file
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


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
        Field(description="JSON file name or path, e.g. reports.json or C:/.../reports.json"),
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

    normalized_json = json.dumps(parsed, ensure_ascii=False, indent=2)

    prompt_lines = [
        "You are an elite tennis technique analyst.",
        "Analyze the JSON as video-derived tennis motion data and provide:",
        "1) summary,",
        "2) best performance,",
        "3) biggest weakness,",
        "4) improvement advice.",
        "Return in bilingual format (English + Chinese) with exactly these Markdown headings:",
        "## Summary",
        "## Best Performance",
        "## Biggest Weakness",
        "## Improvement Advice",
        "Inside each section, write concise English first, then concise Chinese.",
        "After these sections, add:",
        "## Recommended Coaching Videos",
        "Provide 3-5 practical video recommendations tailored to the athlete's biggest weakness.",
        "For each recommendation, include title, one-line reason, and a direct clickable URL starting with https://.",
        "Write for the athlete in coaching language: clear, direct, and actionable.",
        "Ground conclusions in JSON evidence, but avoid raw data dump style.",
    ]
    if analysis_focus.strip():
        prompt_lines.append(f"Focus area: {analysis_focus.strip()}")

    return (
        "\n".join(prompt_lines)
        + "\n\nTennis video analysis JSON:\n```json\n"
        + normalized_json
        + "\n```"
    )


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
            "When the user provides a JSON file for tennis video analysis, call analyze_tennis_technique_from_json first "
            "and then produce a technique-focused report based on that JSON evidence. "
            "For JSON-based technique reports, always return exactly four sections: "
            "Summary, Best Performance, Biggest Weakness, Improvement Advice. "
            "Each section must be bilingual (English + Chinese). "
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
