from __future__ import annotations

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import SessionArchiveEpisodeAdapter


def test_cross_category_events_are_sorted_by_valid_ingest_sequence_and_id() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="ordered",
        archive_uri="memoryos://user/u1/sessions/history/ordered",
        messages=[
            {
                "id": "message-late",
                "role": "user",
                "content": "latest message",
                "occurred_at": "2026-01-01T10:00:03Z",
                "ingested_at": "2026-01-01T10:00:04Z",
                "sequence": 4,
            }
        ],
        observations=[
            {
                "id": "observation-first",
                "role": "sensor",
                "raw_text": "first observation",
                "occurred_at": "2026-01-01T10:00:00Z",
                "ingested_at": "2026-01-01T10:00:05Z",
                "sequence": 9,
            }
        ],
        tool_results=[
            {
                "id": "tool-middle",
                "role": "tool",
                "tool_output": "middle result",
                "occurred_at": "2026-01-01T10:00:02Z",
                "ingested_at": "2026-01-01T10:00:02Z",
                "sequence": 3,
            }
        ],
        feedback=[
            {
                "id": "feedback-second",
                "role": "user",
                "content": "second feedback",
                "occurred_at": "2026-01-01T10:00:01Z",
                "ingested_at": "2026-01-01T10:00:06Z",
                "sequence": 10,
            }
        ],
        created_at="2026-01-01T10:00:10Z",
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    assert [event.event_id for event in episode.events] == [
        "observation-first",
        "feedback-second",
        "tool-middle",
        "message-late",
    ]
    assert episode.started_at.isoformat() == "2026-01-01T10:00:00+00:00"
    assert episode.ended_at.isoformat() == "2026-01-01T10:00:03+00:00"


def test_equal_times_use_ingest_sequence_then_event_id_stably() -> None:
    common = {
        "role": "user",
        "occurred_at": "2026-01-01T10:00:00Z",
        "ingested_at": "2026-01-01T10:00:00Z",
    }
    archive = SessionArchive(
        user_id="u1",
        session_id="ties",
        archive_uri="memoryos://user/u1/sessions/history/ties",
        messages=[
            {**common, "id": "z", "sequence": 2, "content": "z"},
            {**common, "id": "b", "sequence": 1, "content": "b"},
        ],
        feedback=[{**common, "id": "a", "sequence": 1, "content": "a"}],
        created_at="2026-01-01T10:00:00Z",
    )
    first = SessionArchiveEpisodeAdapter().adapt(archive)
    second = SessionArchiveEpisodeAdapter().adapt(archive)
    assert [event.event_id for event in first.events] == ["a", "b", "z"]
    assert [event.digest for event in first.events] == [event.digest for event in second.events]


def test_late_historical_event_keeps_valid_time_separate_from_ingest_time() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="late",
        archive_uri="memoryos://user/u1/sessions/history/late",
        messages=[
            {
                "id": "current",
                "role": "user",
                "content": "current",
                "occurred_at": "2026-07-01T00:00:00Z",
                "ingested_at": "2026-07-01T00:00:00Z",
                "sequence": 2,
            },
            {
                "id": "late-history",
                "role": "user",
                "content": "historical",
                "occurred_at": "2025-01-01T00:00:00Z",
                "ingested_at": "2026-07-02T00:00:00Z",
                "sequence": 3,
            },
        ],
        created_at="2026-07-02T00:00:00Z",
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    historical = episode.event("late-history")
    assert historical is not None
    assert episode.events[0].event_id == "late-history"
    assert historical.occurred_at.year == 2025
    assert historical.ingested_at is not None and historical.ingested_at.year == 2026


def test_missing_and_invalid_fields_are_deterministically_defaulted_and_marked() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="inferred",
        archive_uri="memoryos://user/u1/sessions/history/inferred",
        messages=[{"id": "m1", "content": "hello", "occurred_at": "not-a-time", "sequence": "bad"}],
        created_at="2026-01-01T00:00:00Z",
    )
    first = SessionArchiveEpisodeAdapter().adapt(archive).events[0]
    second = SessionArchiveEpisodeAdapter().adapt(archive).events[0]
    assert first.digest == second.digest
    assert first.actor.kind == "user"
    assert first.actor.id_inferred and first.actor.role_inferred
    assert first.subjects[0].inferred
    assert first.occurred_at_inferred and first.ingested_at_inferred and first.sequence_inferred
    assert set(first.metadata["invalid_fields"]) == {"occurred_at", "sequence"}
    assert set(first.metadata["inferred_fields"]) >= {
        "actor.id",
        "actor.role",
        "subjects",
        "occurred_at",
        "ingested_at",
        "sequence",
    }
