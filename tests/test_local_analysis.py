import json

import pytest

from llm_process.src.tennis_analysis_agent.analysis_prompt import (
    build_analysis_evidence_chunks,
    build_analysis_messages,
    build_analysis_prompt,
)
from video_extraction.local_analysis import (
    _parse_claim_response,
    load_analysis,
    validate_claims,
    validate_synthesis,
)


def _analysis():
    return {
        "schema": {"name": "tennis-coach-analysis", "version": 1},
        "data_quality": {"warnings": ["Ball confidence is low."]},
        "analysis_capabilities": {
            "supported": ["player movement comparison"],
            "unsupported": ["winner/error attribution"],
        },
        "players": {"player_1": {"trajectory_samples": 10}},
        "segments": [],
    }


def test_builds_grounded_foundry_messages():
    messages = build_analysis_messages(_analysis(), "Who moved more?")

    assert messages[0]["role"] == "system"
    assert "Use only claims listed" in messages[0]["content"]
    assert "Who moved more?" in messages[1]["content"]
    assert '"winner/error attribution"' in messages[1]["content"]
    assert "Ball confidence is low." in messages[1]["content"]


def test_local_and_cloud_prompts_share_grounding_rules():
    analysis = _analysis()
    messages = build_analysis_messages(analysis, "Who moved more?")
    cloud_prompt = build_analysis_prompt(analysis, "Who moved more?")

    assert cloud_prompt.startswith(messages[0]["content"])
    assert messages[1]["content"] in cloud_prompt
    assert "## Improvement Advice" in messages[0]["content"]


def test_load_analysis_requires_capability_metadata(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text('{"players": {}, "segments": []}')

    with pytest.raises(ValueError, match="analysis_capabilities"):
        load_analysis(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", None, "unsupported schema"),
        (
            "data_quality",
            {"warnings": "low confidence"},
            "data_quality.warnings",
        ),
        (
            "analysis_capabilities",
            {"supported": [], "unsupported": "winner attribution"},
            "analysis_capabilities.unsupported",
        ),
        ("players", [], "players must be an object"),
        ("segments", {}, "segments must be a list"),
    ],
)
def test_load_analysis_validates_grounding_metadata(tmp_path, field, value, message):
    analysis = _analysis()
    analysis[field] = value
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis))

    with pytest.raises(ValueError, match=message):
        load_analysis(path)


def test_load_analysis_rejects_unknown_schema_version(tmp_path):
    analysis = _analysis()
    analysis["schema"]["version"] = 2
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis))

    with pytest.raises(ValueError, match="schema version"):
        load_analysis(path)


def test_builds_small_canonical_segment_chunks():
    analysis = _analysis()
    analysis["segments"] = [{"index": index} for index in range(5)]

    chunks = build_analysis_evidence_chunks(analysis, segment_batch_size=2)

    assert chunks[0]["kind"] == "global"
    assert "segments" not in chunks[0]["evidence"]
    assert list(chunks[1]["canonical_segments"]) == ["$.segments[0]", "$.segments[1]"]
    assert list(chunks[-1]["canonical_segments"]) == ["$.segments[4]"]


def test_parses_json_claims_after_reasoning_text():
    response = 'Reasoning omitted.\\n```json\\n{"claims":[{"claim":"Observed","citations":[]}]}\\n```'

    assert _parse_claim_response(response)[0]["claim"] == "Observed"


def test_validates_exact_scalar_citations():
    analysis = _analysis()
    claims = [
        {
            "claim": "Player one has ten samples.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10}
            ],
        },
        {
            "claim": "Invented count.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 50}
            ],
        },
        {
            "claim": "Wrong scalar type.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10.0}
            ],
        },
        {
            "claim": "Player one has 999 samples.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10}
            ],
        },
        {
            "claim": "Fabricated speed is 999mph.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10}
            ],
        },
        {
            "claim": "Fabricated cost is USD999.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10}
            ],
        },
    ]

    assert validate_claims(analysis, claims) == [claims[0]]


def test_rejects_synthesis_numbers_without_evidence():
    analysis = _analysis()
    claims = [
        {
            "claim": "Player one has ten samples.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10}
            ],
        }
    ]

    for synthesis in (
        "The player has 999 samples.",
        "Fabricated speed is 9.99e2mph.",
        "Fabricated cost is USD999.",
    ):
        with pytest.raises(ValueError, match="999|9.99e2"):
            validate_synthesis(synthesis, analysis, claims)


def test_appends_deterministic_warnings_and_evidence():
    analysis = _analysis()
    claims = [
        {
            "claim": "Player one has 10 samples.",
            "citations": [
                {"path": "$.players.player_1.trajectory_samples", "value": 10}
            ],
        }
    ]

    result = validate_synthesis(
        "The player has 10 samples.\n1. Practice balanced movement.", analysis, claims
    )

    assert "Warning: Ball confidence is low." in result
    assert "`$.players.player_1.trajectory_samples` = `10`" in result
