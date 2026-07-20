import json
from pathlib import Path

import pytest

from run_coach_pipeline import (
    _compact_report_payload,
    _load_report_json_for_invoke,
    _resolve_azd_command,
)


def test_compact_report_retains_coaching_evidence_and_summarizes_unresolved_shots(tmp_path):
    report = {
        "schema": {
            "name": "tennis-coach-analysis",
            "version": 1,
            "sections": {"verbose": "documentation" * 100},
        },
        "source": {"segment_count": 1},
        "data_quality": {"shots": {"candidate_count": 3, "classified_count": 1}},
        "analysis_capabilities": {"supported": ["shot classification"], "unsupported": []},
        "target_player": {"player_id": None},
        "players": {"player_1": {"shot_counts": {"forehand": 1}}},
        "segments": [{
            "index": 1,
            "start": 1.0,
            "end": 2.0,
            "duration": 1.0,
            "motion": {"player_motion_max": 0.2},
            "audio": {"hit_count": 3},
            "ball": {"visible_ratio": 0.1},
            "players": {"player_1": {"trajectory_samples": 50}},
            "shots": [
                {
                    "time": 1.1,
                    "player_id": "player_1",
                    "classification": "forehand",
                    "confidence": 0.8,
                    "pose_landmarks": {"left_wrist": {"x": 0.2, "y": 0.3}},
                    "contact_point": {"x": 10, "y": 20},
                },
                {"time": 1.2, "player_id": None, "classification": "unknown", "reason": "no_nearby_ball"},
                {"time": 1.3, "player_id": None, "classification": "unknown", "reason": "no_nearby_ball"},
            ],
            "outcome": {"classification": "unknown"},
        }],
    }

    compact = _compact_report_payload(report)

    assert compact["schema"] == {"name": "tennis-coach-analysis", "version": 1}
    assert compact["players"] == report["players"]
    segment = compact["segments"][0]
    assert segment["shots"] == [{
        "time": 1.1,
        "player_id": "player_1",
        "classification": "forehand",
        "confidence": 0.8,
    }]
    assert segment["shot_candidate_summary"] == {
        "total": 3,
        "by_reason": {"classified": 1, "no_nearby_ball": 2},
    }
    assert "players" not in segment
    assert "pose_landmarks" not in json.dumps(compact)
    assert "contact_point" not in json.dumps(compact)

    report_path = tmp_path / "analysis.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    payload = _load_report_json_for_invoke(report_path, inline_json_max_chars=3000)
    assert json.loads(payload) == compact


def test_resolve_azd_command_uses_user_install_when_not_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr("run_coach_pipeline.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    azd = tmp_path / ".local" / "bin" / "azd"
    azd.parent.mkdir(parents=True)
    azd.touch()

    assert _resolve_azd_command() == str(azd)


def test_resolve_azd_command_reports_actionable_setup_error(monkeypatch, tmp_path):
    monkeypatch.setattr("run_coach_pipeline.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(RuntimeError, match=r"Azure Developer CLI.*azd auth login"):
        _resolve_azd_command()
