from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


STATS_VERSION = 1


def _rounded_mean(values: list[float], digits: int = 6) -> float | None:
    return round(statistics.fmean(values), digits) if values else None


def _weighted_mean(values_and_weights: list[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _value, weight in values_and_weights)
    if total_weight == 0:
        return None
    return round(
        sum(value * weight for value, weight in values_and_weights) / total_weight,
        6,
    )


def _longest_missing_run(observations: list[dict]) -> int:
    longest = 0
    current = 0
    for observation in observations:
        if observation.get("visible"):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def summarize_ball_trajectory(observations: list[dict]) -> dict:
    if not observations:
        return {
            "available": False,
            "observation_count": 0,
            "visible_count": 0,
            "visible_ratio": None,
        }

    visible = [observation for observation in observations if observation.get("visible")]
    speeds = []
    travel = 0.0
    court_crossings = 0
    direction_change_candidates = 0
    previous_velocity = None
    previous_visible = None

    for observation in observations:
        if not observation.get("visible"):
            previous_visible = None
            previous_velocity = None
            continue
        if previous_visible is not None:
            delta_time = observation["time"] - previous_visible["time"]
            if delta_time > 0:
                delta_x = observation["x_normalized"] - previous_visible["x_normalized"]
                delta_y = observation["y_normalized"] - previous_visible["y_normalized"]
                distance = math.hypot(delta_x, delta_y)
                travel += distance
                velocity = (delta_x / delta_time, delta_y / delta_time)
                speeds.append(math.hypot(*velocity))

                previous_court = previous_visible.get("court_projection")
                current_court = observation.get("court_projection")
                if (
                    previous_court
                    and current_court
                    and previous_court["inside_court"]
                    and current_court["inside_court"]
                ):
                    previous_half = previous_court["y"] < 0.5
                    current_half = current_court["y"] < 0.5
                    if previous_half != current_half:
                        court_crossings += 1

                if previous_velocity is not None:
                    previous_speed = math.hypot(*previous_velocity)
                    current_speed = math.hypot(*velocity)
                    if previous_speed > 0 and current_speed > 0:
                        cosine = (
                            previous_velocity[0] * velocity[0]
                            + previous_velocity[1] * velocity[1]
                        ) / (previous_speed * current_speed)
                        if cosine < 0.5:
                            direction_change_candidates += 1
                previous_velocity = velocity
        previous_visible = observation

    return {
        "available": True,
        "observation_count": len(observations),
        "visible_count": len(visible),
        "visible_ratio": round(len(visible) / len(observations), 6),
        "longest_missing_run_observations": _longest_missing_run(observations),
        "normalized_image_travel": round(travel, 6),
        "speed_sample_count": len(speeds),
        "mean_normalized_image_speed_per_second": _rounded_mean(speeds),
        "max_normalized_image_speed_per_second": round(max(speeds), 6) if speeds else None,
        "court_half_crossings": court_crossings,
        "direction_change_candidates": direction_change_candidates,
        "mean_detection_confidence": _rounded_mean(
            [observation["confidence"] for observation in visible]
        ),
    }


def aggregate_ball_summaries(summaries: list[dict]) -> dict:
    available = [summary for summary in summaries if summary["available"]]
    if not available:
        return summarize_ball_trajectory([])

    observation_count = sum(summary["observation_count"] for summary in available)
    visible_count = sum(summary["visible_count"] for summary in available)
    speed_values = [
        (
            summary["mean_normalized_image_speed_per_second"],
            summary["speed_sample_count"],
        )
        for summary in available
        if summary["mean_normalized_image_speed_per_second"] is not None
    ]
    confidence_values = [
        (summary["mean_detection_confidence"], summary["visible_count"])
        for summary in available
        if summary["mean_detection_confidence"] is not None
    ]
    maximum_speeds = [
        summary["max_normalized_image_speed_per_second"]
        for summary in available
        if summary["max_normalized_image_speed_per_second"] is not None
    ]
    return {
        "available": True,
        "observation_count": observation_count,
        "visible_count": visible_count,
        "visible_ratio": round(visible_count / observation_count, 6),
        "longest_missing_run_observations": max(
            summary["longest_missing_run_observations"] for summary in available
        ),
        "normalized_image_travel": round(
            sum(summary["normalized_image_travel"] for summary in available),
            6,
        ),
        "speed_sample_count": sum(summary["speed_sample_count"] for summary in available),
        "mean_normalized_image_speed_per_second": _weighted_mean(speed_values),
        "max_normalized_image_speed_per_second": (
            round(max(maximum_speeds), 6) if maximum_speeds else None
        ),
        "court_half_crossings": sum(
            summary["court_half_crossings"] for summary in available
        ),
        "direction_change_candidates": sum(
            summary["direction_change_candidates"] for summary in available
        ),
        "mean_detection_confidence": _weighted_mean(confidence_values),
    }


def _court_positions(points: list[dict], inside_only: bool = False) -> list[dict]:
    return [
        point["court_position"]
        for point in points
        if point.get("court_position")
        and math.isfinite(point["court_position"]["x"])
        and math.isfinite(point["court_position"]["y"])
        and (not inside_only or point["court_position"]["inside_court"])
    ]


def _mean_court_position(
    points: list[dict],
    inside_only: bool = False,
) -> list[float] | None:
    positions = _court_positions(points, inside_only=inside_only)
    if not positions:
        return None
    return [
        round(statistics.fmean(position["x"] for position in positions), 6),
        round(statistics.fmean(position["y"] for position in positions), 6),
    ]


def _court_movement(points: list[dict]) -> float:
    positions = _court_positions(points)
    return round(
        sum(
            math.hypot(
                current["x"] - previous["x"],
                current["y"] - previous["y"],
            )
            for previous, current in zip(positions, positions[1:])
        ),
        6,
    )


def _segment_player_summary(segment: dict, player_id: str) -> dict:
    player = segment.get("players", {}).get(player_id, {"detected": False})
    points = segment.get("player_trajectories", {}).get(player_id, [])
    return {
        "detected": bool(player.get("detected")),
        "side": player.get("side"),
        "detection_confidence": player.get("detection_confidence"),
        "identity_confidence": player.get("identity_confidence"),
        "image_movement_distance_normalized": player.get("movement_distance"),
        "court_movement_normalized": _court_movement(points),
        "trajectory_samples": len(points),
        "mean_image_position": player.get("mean_position"),
        "mean_court_position": _mean_court_position(points),
        "mean_in_court_position": _mean_court_position(points, inside_only=True),
    }


def _aggregate_players(report: list[dict], player_ids: list[str]) -> dict:
    aggregates = {}
    segment_count = len(report)
    for player_id in player_ids:
        detected_players = []
        trajectory_points = []
        image_movement = []
        court_movement = []
        court_transition_samples = 0
        for segment in report:
            player = segment.get("players", {}).get(player_id)
            if player and player.get("detected"):
                detected_players.append(player)
                if player.get("movement_distance") is not None:
                    image_movement.append(player["movement_distance"])
            segment_points = segment.get("player_trajectories", {}).get(player_id, [])
            trajectory_points.extend(segment_points)
            court_movement.append(_court_movement(segment_points))
            court_transition_samples += max(len(_court_positions(segment_points)) - 1, 0)

        side_counts = {
            side: sum(point.get("side") == side for point in trajectory_points)
            for side in ("near", "far")
        }
        weighted_detection_confidence = [
            (player["detection_confidence"], player.get("sample_count", 1))
            for player in detected_players
            if player.get("detection_confidence") is not None
        ]
        weighted_identity_confidence = [
            (player["identity_confidence"], player.get("sample_count", 1))
            for player in detected_players
            if player.get("identity_confidence") is not None
        ]
        post_initialization_identity = weighted_identity_confidence[1:]
        aggregates[player_id] = {
            "segments_detected": len(detected_players),
            "segment_detection_rate": (
                round(len(detected_players) / segment_count, 6) if segment_count else None
            ),
            "trajectory_samples": len(trajectory_points),
            "side_sample_counts": side_counts,
            "mean_detection_confidence": _weighted_mean(weighted_detection_confidence),
            "mean_identity_confidence": _weighted_mean(weighted_identity_confidence),
            "mean_identity_confidence_after_initialization": _weighted_mean(
                post_initialization_identity
            ),
            "total_image_movement_distance_normalized": round(sum(image_movement), 6),
            "total_court_movement_normalized": round(sum(court_movement), 6),
            "court_transition_samples": court_transition_samples,
            "mean_court_position": _mean_court_position(trajectory_points),
            "mean_in_court_position": _mean_court_position(
                trajectory_points,
                inside_only=True,
            ),
        }
    return aggregates


def build_llm_statistics(report: list[dict]) -> dict:
    player_ids = sorted({
        player_id
        for segment in report
        for player_id in segment.get("players", {})
    })
    segments = []
    ball_summaries = []
    for segment in report:
        ball_trajectory = segment.get("ball_trajectory", [])
        ball_summary = summarize_ball_trajectory(ball_trajectory)
        ball_summaries.append(ball_summary)
        segments.append({
            "index": segment.get("index"),
            "start": segment["start"],
            "end": segment["end"],
            "duration": round(segment["end"] - segment["start"], 6),
            "motion": {
                key: segment.get("features", {}).get(key)
                for key in (
                    "player_motion_max",
                    "player_motion_var",
                    "near_motion_mean",
                    "far_motion_mean",
                    "motion_sample_count",
                )
            },
            "players": {
                player_id: _segment_player_summary(segment, player_id)
                for player_id in player_ids
            },
            "ball": ball_summary,
        })

    players = _aggregate_players(report, player_ids)
    ball_summary = aggregate_ball_summaries(ball_summaries)
    warnings = []
    if not ball_summary["available"]:
        warnings.append("Ball tracking is unavailable in this report.")
    low_identity_players = [
        player_id
        for player_id, player in players.items()
        if player["mean_identity_confidence_after_initialization"] is not None
        and player["mean_identity_confidence_after_initialization"] < 0.6
    ]
    if low_identity_players:
        warnings.append(
            "Low identity confidence affects: " + ", ".join(low_identity_players) + "."
        )
    warnings.extend([
        "Segment boundaries are accepted from the input report and are not guaranteed to be rallies.",
        "Court projection assumes points lie on the court plane; airborne ball projections are approximate.",
    ])
    comparable_movement_players = sum(
        player["segments_detected"] > 0
        and player["court_transition_samples"] > 0
        for player in players.values()
    )
    has_player_positions = any(
        player["mean_court_position"] is not None
        for player in players.values()
    )
    comparable_activity_segments = sum(
        bool(segment["motion"]["motion_sample_count"])
        for segment in segments
    )
    supported = []
    if comparable_movement_players >= 2:
        supported.append("player movement comparison")
    if has_player_positions:
        supported.append("approximate player positioning and court coverage")
    if comparable_activity_segments >= 2:
        supported.append("segment activity comparison")
    if ball_summary["observation_count"] > 0:
        supported.append("ball visibility analysis")
    if ball_summary.get("speed_sample_count", 0) > 0:
        supported.append("ball image-space trajectory analysis")
    if (
        ball_summary.get("speed_sample_count", 0) >= 2
        and any(
            summary.get("court_half_crossings", 0)
            or summary.get("direction_change_candidates", 0)
            for summary in ball_summaries
        )
    ):
        supported.append("candidate court crossings and direction changes")

    return {
        "stats_version": STATS_VERSION,
        "source": {
            "segment_count": len(report),
            "start": min((segment["start"] for segment in report), default=None),
            "end": max((segment["end"] for segment in report), default=None),
            "duration": (
                round(
                    max(segment["end"] for segment in report)
                    - min(segment["start"] for segment in report),
                    6,
                )
                if report
                else 0.0
            ),
        },
        "data_quality": {
            "ball": ball_summary,
            "warnings": warnings,
        },
        "analysis_capabilities": {
            "supported": supported,
            "unsupported": [
                "forehand/backhand classification",
                "shot success ratios",
                "winner/error attribution",
                "serve and return classification",
                "target-player identification from a key frame",
                "physical ball speed or 3D trajectory",
            ],
        },
        "players": players,
        "segments": segments,
    }


def generate_stats_file(report_path: str | Path, output_path: str | Path) -> dict:
    report = json.loads(Path(report_path).read_text())
    if not isinstance(report, list):
        raise ValueError("Report JSON must contain a list of segments")
    stats = build_llm_statistics(report)
    Path(output_path).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate compact LLM-ready tennis statistics.")
    parser.add_argument("report", help="Path to enriched report JSON")
    parser.add_argument("-o", "--output", default="stats.json", help="Output stats JSON path")
    args = parser.parse_args(argv)
    stats = generate_stats_file(args.report, args.output)
    print(
        f"Wrote statistics for {stats['source']['segment_count']} segments "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
