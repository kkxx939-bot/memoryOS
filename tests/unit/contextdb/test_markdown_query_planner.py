from __future__ import annotations

from datetime import datetime, timezone

from memoryos.application.context.query_planner import QueryPlanner
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent


def _planner(now: datetime) -> QueryPlanner:
    return QueryPlanner(now_provider=lambda: now)


def test_past_chat_day_number_uses_caller_timezone_and_event_half_open_range() -> None:
    plan = _planner(datetime(2026, 7, 17, 3, tzinfo=timezone.utc)).build(
        "我想看一下我11号和你讨论java的分布式方案，你还记得吗？",
        options=RetrievalOptions(timezone="Asia/Singapore"),
    )
    assert plan.query_intent is RetrievalQueryIntent.OPEN_RECALL
    assert plan.event_time_from == "2026-07-10T16:00:00+00:00"
    assert plan.event_time_to == "2026-07-11T16:00:00+00:00"


def test_future_day_number_with_past_cue_uses_previous_month_across_year() -> None:
    plan = _planner(datetime(2026, 1, 2, 12, tzinfo=timezone.utc)).build(
        "回顾一下3号聊过的方案",
        options=RetrievalOptions(timezone="UTC"),
    )
    assert plan.event_time_from == "2025-12-03T00:00:00+00:00"
    assert plan.event_time_to == "2025-12-04T00:00:00+00:00"


def test_day_missing_from_current_month_falls_back_to_previous_month() -> None:
    plan = _planner(datetime(2026, 2, 10, 12, tzinfo=timezone.utc)).build(
        "回顾一下31号聊过的方案",
        options=RetrievalOptions(timezone="UTC"),
    )
    assert plan.event_time_from == "2026-01-31T00:00:00+00:00"
    assert plan.event_time_to == "2026-02-01T00:00:00+00:00"


def test_day_number_without_past_cue_does_not_guess_month() -> None:
    plan = _planner(datetime(2026, 7, 17, tzinfo=timezone.utc)).build(
        "11号提醒我",
        options=RetrievalOptions(timezone="Asia/Singapore"),
    )
    assert plan.event_time_from is None
    assert plan.event_time_to is None


def test_explicit_event_range_wins_over_natural_language_date() -> None:
    options = RetrievalOptions(
        timezone="UTC",
        event_time_from="2026-06-01",
        event_time_to="2026-06-03",
    )
    plan = _planner(datetime(2026, 7, 17, tzinfo=timezone.utc)).build(
        "回顾一下11号聊过的方案",
        options=options,
    )
    assert plan.event_time_from == "2026-06-01T00:00:00+00:00"
    assert plan.event_time_to == "2026-06-04T00:00:00+00:00"


def test_query_contract_contains_document_filters_and_no_removed_state_fields() -> None:
    options = RetrievalOptions(
        document_ids=("memdoc_1234567890abcdef",),
        document_kinds=("episode",),
        record_kinds=("memory_document", "memory_block"),
        query_intent=RetrievalQueryIntent.EXACT,
    )
    payload = options.to_dict()
    assert payload["document_kinds"] == ["episode"]
    assert payload["record_kinds"] == ["memory_document", "memory_block"]
    assert "canonical_resolution_mode" not in payload
    assert "valid_at" not in payload
