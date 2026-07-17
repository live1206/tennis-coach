from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from llm_process.src.tennis_analysis_agent.analysis_prompt import (
    build_analysis_evidence_chunks,
    build_chunk_messages,
    build_synthesis_messages,
    validate_analysis,
)


DEFAULT_QUESTION = (
    "Summarize the strongest evidence-backed coaching observations, data-quality "
    "limitations, and the next three practice priorities."
)


def load_analysis(path: str | Path) -> dict:
    analysis_path = Path(path)
    return validate_analysis(json.loads(analysis_path.read_text(encoding="utf-8")))


def _complete_streaming_chat(
    client,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_format: dict[str, str] | None = None,
) -> str:
    client.settings.temperature = 0.0
    client.settings.max_tokens = max_tokens
    client.settings.response_format = response_format
    chunks = []
    for chunk in client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            chunks.append(content)
    return "".join(chunks)


def _parse_claim_response(response: str) -> list[dict]:
    decoder = json.JSONDecoder()
    parsed = None
    for match in re.finditer(r"\{", response):
        try:
            candidate, _ = decoder.raw_decode(response[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "claims" in candidate:
            parsed = candidate
            break
    if parsed is None or not isinstance(parsed["claims"], list):
        preview = response.strip().replace("\n", " ")[:200]
        raise ValueError(
            "Chunk response did not contain a valid claims JSON object"
            + (f": {preview}" if preview else "")
        )
    return parsed["claims"]


_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
_NUMBER_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _resolve_json_path(document: object, path: str) -> Any:
    if not isinstance(path, str) or not path.startswith("$."):
        raise ValueError(f"Unsupported JSON path: {path!r}")
    current = document
    position = 1
    for match in _PATH_TOKEN.finditer(path, position):
        if match.start() != position:
            raise ValueError(f"Unsupported JSON path: {path!r}")
        key, index = match.groups()
        if key is not None:
            if not isinstance(current, dict) or key not in current:
                raise ValueError(f"JSON path does not exist: {path}")
            current = current[key]
        else:
            numeric_index = int(index)
            if not isinstance(current, list) or numeric_index >= len(current):
                raise ValueError(f"JSON path does not exist: {path}")
            current = current[numeric_index]
        position = match.end()
    if position != len(path):
        raise ValueError(f"Unsupported JSON path: {path!r}")
    return current


def validate_claims(analysis: dict, claims: list[dict]) -> list[dict]:
    validated = []
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("claim"), str)
            or not claim["claim"].strip()
        ):
            continue
        citations = claim.get("citations")
        if not isinstance(citations, list) or not citations:
            continue
        validated_citations = []
        for citation in citations:
            if not isinstance(citation, dict) or "path" not in citation:
                break
            try:
                actual = _resolve_json_path(analysis, citation["path"])
            except ValueError:
                break
            cited_value = citation.get("value")
            if (
                isinstance(actual, (dict, list))
                or type(cited_value) is not type(actual)
                or cited_value != actual
            ):
                break
            validated_citations.append({"path": citation["path"], "value": actual})
        else:
            allowed_numbers = {
                token
                for citation in validated_citations
                for token in _NUMBER_TOKEN.findall(
                    json.dumps(citation["value"], ensure_ascii=False)
                )
            }
            if not set(_NUMBER_TOKEN.findall(claim["claim"])).issubset(
                allowed_numbers
            ):
                continue
            validated.append(
                {"claim": claim["claim"].strip(), "citations": validated_citations}
            )
    return validated


def validate_synthesis(
    synthesis: str, analysis: dict, validated_claims: list[dict]
) -> str:
    evidence_numbers = {
        token
        for claim in validated_claims
        for citation in claim["citations"]
        for token in _NUMBER_TOKEN.findall(
            json.dumps(citation["value"], ensure_ascii=False)
        )
    }
    warning_numbers = {
        token
        for warning in analysis["data_quality"]["warnings"]
        for token in _NUMBER_TOKEN.findall(warning)
    }
    prose = re.sub(r"(?m)^\s*\d+[.)]\s+", "", synthesis)
    unsupported = sorted(
        set(_NUMBER_TOKEN.findall(prose)) - evidence_numbers - warning_numbers
    )
    if unsupported:
        raise ValueError(
            "Final synthesis introduced numeric values without validated evidence: "
            + ", ".join(unsupported)
        )

    warnings = analysis["data_quality"]["warnings"]
    ledger_lines = [
        "> **Deterministic evidence ledger**",
        *[f"> Warning: {warning}" for warning in warnings],
    ]
    for claim in validated_claims:
        for citation in claim["citations"]:
            value = json.dumps(citation["value"], ensure_ascii=False)
            ledger_lines.append(f"> `{citation['path']}` = `{value}`")
    return synthesis.rstrip() + "\n\n" + "\n".join(ledger_lines)


def analyze_in_chunks(
    client,
    analysis: dict,
    question: str,
    segment_batch_size: int,
    max_map_tokens: int,
    max_final_tokens: int,
) -> str:
    validated_claims = []
    rejected_count = 0
    for chunk in build_analysis_evidence_chunks(analysis, segment_batch_size):
        response = _complete_streaming_chat(
            client,
            build_chunk_messages(chunk, question),
            max_map_tokens,
            response_format={"type": "json_object"},
        )
        claims = _parse_claim_response(response)
        validated = validate_claims(analysis, claims)
        validated_claims.extend(validated)
        rejected_count += len(claims) - len(validated)

    if not validated_claims:
        raise ValueError("The model produced no claims with valid evidence citations")

    unique_claims = []
    seen = set()
    for claim in validated_claims:
        key = json.dumps(claim, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique_claims.append(claim)

    synthesis = _complete_streaming_chat(
        client,
        build_synthesis_messages(analysis, question, unique_claims),
        max_final_tokens,
    )
    result = validate_synthesis(synthesis, analysis, unique_claims)
    if rejected_count:
        result += (
            f"\n\n> Evidence validation rejected {rejected_count} "
            "model-generated claim(s) with missing or mismatched citations."
        )
    return result


def analyze_with_foundry_local(
    analysis: dict,
    question: str,
    model_alias: str = "qwen2.5-0.5b",
    register_execution_providers: bool = True,
    segment_batch_size: int = 2,
    max_map_tokens: int = 512,
    max_final_tokens: int = 768,
) -> str:
    validate_analysis(analysis)
    if max_map_tokens < 1 or max_final_tokens < 1:
        raise ValueError("Token limits must be positive")
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ImportError as error:
        raise RuntimeError(
            "Foundry Local is not installed. Install the 'foundry-local' "
            "optional dependency for this project."
        ) from error

    configuration = Configuration(app_name="tennis_coach")
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(configuration)
    manager = FoundryLocalManager.instance
    if register_execution_providers:
        manager.download_and_register_eps()

    model = manager.catalog.get_model(model_alias)
    if model is None:
        raise ValueError(f"Unknown Foundry Local model alias: {model_alias}")
    model.download()
    model.load()
    try:
        client = model.get_chat_client()
        return analyze_in_chunks(
            client,
            analysis,
            question,
            segment_batch_size,
            max_map_tokens,
            max_final_tokens,
        )
    finally:
        model.unload()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze tennis analysis.json locally with Foundry Local."
    )
    parser.add_argument("analysis", help="Path to canonical analysis.json")
    parser.add_argument(
        "-q",
        "--question",
        default=DEFAULT_QUESTION,
        help="Coaching question grounded in the analysis evidence",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-0.5b",
        help="Foundry Local model alias",
    )
    parser.add_argument(
        "--skip-ep-download",
        action="store_true",
        help="Skip execution-provider discovery/download",
    )
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=2,
        help="Number of segments per evidence chunk",
    )
    parser.add_argument("--max-map-tokens", type=int, default=512)
    parser.add_argument("--max-final-tokens", type=int, default=768)
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path for the generated coaching analysis",
    )
    args = parser.parse_args(argv)

    result = analyze_with_foundry_local(
        load_analysis(args.analysis),
        args.question,
        model_alias=args.model,
        register_execution_providers=not args.skip_ep_download,
        segment_batch_size=args.segment_batch_size,
        max_map_tokens=args.max_map_tokens,
        max_final_tokens=args.max_final_tokens,
    )
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
