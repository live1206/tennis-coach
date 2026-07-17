from __future__ import annotations


TERMINAL_ERROR_EVENTS = {"net", "out"}


def _other_player(
    player_id: str,
    player_ids: set[str],
) -> str | None:
    others = sorted(player_ids - {player_id})
    return others[0] if len(others) == 1 else None


def infer_segment_outcome(
    segment: dict,
    minimum_outcome_confidence: float = 0.35,
) -> dict:
    shots = [dict(shot) for shot in segment.get("shots", [])]
    player_ids = {
        player_id
        for player_id in segment.get("players", {})
        if player_id
    }
    resolved_indices = [
        index for index, shot in enumerate(shots)
        if shot.get("player_id") is not None
    ]

    for position, shot_index in enumerate(resolved_indices):
        shot = shots[shot_index]
        next_shot = (
            shots[resolved_indices[position + 1]]
            if position + 1 < len(resolved_indices)
            else None
        )
        if (
            next_shot
            and next_shot["player_id"] != shot["player_id"]
            and float(next_shot["time"]) - float(shot["time"]) <= 4.0
        ):
            shot["outcome"] = "continued"
            shot["outcome_confidence"] = round(
                min(
                    float(shot.get("contact_confidence") or 0.0),
                    float(next_shot.get("contact_confidence") or 0.0),
                ),
                6,
            )
            shot["outcome_reason"] = "opponent_contact_followed"
        else:
            shot["outcome"] = "terminal_unknown"
            shot["outcome_confidence"] = 0.0
            shot["outcome_reason"] = "no_validated_terminal_event"

    if not resolved_indices:
        return {
            **segment,
            "shots": shots,
            "outcome": {
                "classification": "unknown",
                "winner_player_id": None,
                "confidence": 0.0,
                "reason": "no_resolved_contacts",
            },
        }

    last_shot = shots[resolved_indices[-1]]
    terminal_events = [
        observation
        for observation in segment.get("ball_trajectory", [])
        if observation.get("event") in TERMINAL_ERROR_EVENTS
        and float(observation["time"]) >= float(last_shot["time"])
    ]
    if terminal_events:
        event = min(terminal_events, key=lambda item: float(item["time"]))
        unresolved_later_contacts = [
            shot
            for shot in shots
            if shot.get("player_id") is None
            and float(last_shot["time"]) < float(shot["time"]) <= float(event["time"])
        ]
        if unresolved_later_contacts:
            return {
                **segment,
                "shots": shots,
                "outcome": {
                    "classification": "unknown",
                    "winner_player_id": None,
                    "confidence": 0.0,
                    "reason": "terminal_hitter_unresolved",
                },
            }
        winner = _other_player(last_shot["player_id"], player_ids)
        event_confidence = event.get("confidence")
        if event_confidence is None:
            event_confidence = 1.0
        confidence = min(
            float(last_shot.get("contact_confidence") or 0.0),
            float(event_confidence),
        )
        if confidence < minimum_outcome_confidence:
            return {
                **segment,
                "shots": shots,
                "outcome": {
                    "classification": "unknown",
                    "winner_player_id": None,
                    "confidence": round(confidence, 6),
                    "reason": "low_terminal_event_confidence",
                },
            }
        last_shot["outcome"] = "error"
        last_shot["outcome_confidence"] = round(confidence, 6)
        last_shot["outcome_reason"] = event["event"]
        return {
            **segment,
            "shots": shots,
            "outcome": {
                "classification": "point_ended",
                "winner_player_id": winner,
                "loser_player_id": last_shot["player_id"],
                "terminal_event": event["event"],
                "confidence": round(confidence, 6),
                "reason": None if winner else "opponent_identity_unavailable",
            },
        }

    return {
        **segment,
        "shots": shots,
        "outcome": {
            "classification": "unknown",
            "winner_player_id": None,
            "confidence": 0.0,
            "reason": "terminal_ball_event_unavailable",
        },
    }


def infer_outcomes(segments: list[dict]) -> list[dict]:
    return [infer_segment_outcome(segment) for segment in segments]
