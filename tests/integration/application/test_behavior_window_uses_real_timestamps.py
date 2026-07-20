from __future__ import annotations

from datetime import datetime, timedelta, timezone

from behavior.core.evaluation.behavior_window import BehaviorWindowEvaluator
from behavior.core.model.behavior_case import BehaviorCase
from behavior.projection.behavior_case import BehaviorCaseWriter

NOW = datetime(2026, 7, 4, tzinfo=timezone.utc)


def _case(case_id: str, observed_at: datetime) -> BehaviorCase:
    return BehaviorCase(
        user_id="u1",
        scene_key="hot_room",
        observation={"scene_key": "hot_room", "location": "home", "environment": {"temperature": 30}, "observed_at": observed_at.isoformat()},
        selected_action="turn_on_ac",
        case_id=case_id,
        created_at=observed_at.isoformat(),
    )


def test_behavior_case_metadata_writes_observed_at() -> None:
    case = _case("c1", NOW)
    payload = BehaviorCaseWriter().add_case(case).payload["context_object"]["metadata"]

    assert payload["observed_at"] == NOW.isoformat()
    assert payload["created_at"] == NOW.isoformat()


def test_behavior_window_uses_iso_datetime_boundaries() -> None:
    evaluator = BehaviorWindowEvaluator()
    current = _case("current", NOW)
    historical = [
        evaluator.historical_record("case-2d", _case("h2", NOW - timedelta(days=2)).to_dict()),
        evaluator.historical_record("case-6d", _case("h6", NOW - timedelta(days=6)).to_dict()),
        evaluator.historical_record("case-20d", _case("h20", NOW - timedelta(days=20)).to_dict()),
        evaluator.historical_record("case-40d", _case("h40", NOW - timedelta(days=40)).to_dict()),
    ]

    decision = evaluator.evaluate("hot_room", [current], historical, now=NOW)

    assert "case-2d" in decision.similar_refs_3d
    assert "case-6d" not in decision.similar_refs_3d
    assert "case-6d" in decision.similar_refs_7d
    assert "case-20d" in decision.similar_refs_30d
    assert "case-40d" not in decision.similar_refs_30d
