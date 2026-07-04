from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.update.behavior_window import BehaviorWindowEvaluator

NOW = datetime(2026, 7, 4, tzinfo=timezone.utc)


def _case(location: str = "home", case_id: str = "current") -> BehaviorCase:
    return BehaviorCase(
        user_id="u1",
        scene_key="hot_room",
        observation={"scene_key": "hot_room", "location": location, "environment": {"temperature": 30}},
        case_id=case_id,
        created_at=NOW.isoformat(),
    )


def _history(case_id: str, days_ago: int, location: str = "home") -> dict:
    case = _case(location=location, case_id=case_id)
    case.created_at = (NOW - timedelta(days=days_ago)).isoformat()
    return BehaviorWindowEvaluator().historical_record(
        f"memoryos://user/u1/behavior/cases/hot_room/{case_id}",
        case.to_dict(),
    )


def test_single_case_does_not_create_cluster_or_pattern() -> None:
    decision = BehaviorWindowEvaluator().evaluate("hot_room", [_case()], [], now=NOW)
    assert not decision.create_cluster
    assert not decision.create_pattern


def test_two_similar_cases_in_three_days_create_cluster() -> None:
    decision = BehaviorWindowEvaluator().evaluate("hot_room", [_case()], [_history("h1", 2)], now=NOW)
    assert decision.create_cluster
    assert not decision.create_pattern


def test_three_similar_cases_in_seven_days_create_pattern() -> None:
    decision = BehaviorWindowEvaluator().evaluate("hot_room", [_case()], [_history("h1", 2), _history("h2", 6)], now=NOW)
    assert decision.create_cluster
    assert decision.create_pattern


def test_thirty_day_stable_repetition_creates_pattern() -> None:
    decision = BehaviorWindowEvaluator().evaluate(
        "hot_room",
        [_case()],
        [_history("h1", 10), _history("h2", 20), _history("h3", 29)],
        now=NOW,
    )
    assert decision.create_pattern


def test_same_scene_with_different_context_tags_does_not_cluster() -> None:
    decision = BehaviorWindowEvaluator().evaluate("hot_room", [_case(location="home")], [_history("h1", 1, location="office")], now=NOW)
    assert not decision.create_cluster
    assert not decision.create_pattern
