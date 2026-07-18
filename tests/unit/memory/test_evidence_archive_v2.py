from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum

import pytest

from memoryos.contextdb.session.session_archive import (
    EvidenceArchiveConflictError,
    EvidenceArchiveIntegrityError,
    SessionArchiveStore,
)
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.core.integrity import canonical_digest
from memoryos.memory.evidence import SessionArchiveEpisodeAdapter


class _Kind(Enum):
    RULE = "rule"


def _archive(*, content: str = "I confirm SQLite.", actor_id: str = "u1") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "actor_id": actor_id,
                "event_type": "DECISION",
                "occurred_at": "2026-01-01T10:00:00Z",
                "ingested_at": "2026-01-01T10:00:01Z",
                "sequence": 7,
                "content": content,
            }
        ],
        metadata={"tenant_id": "t1", "subjects": [{"kind": "person", "id": "u1"}]},
        task_id="session_commit_stable",
        created_at="2026-01-01T10:00:01Z",
    )


def test_event_digest_covers_full_envelope_and_deterministic_types() -> None:
    base = SessionArchiveEpisodeAdapter().adapt(_archive()).events[0]
    other_actor = SessionArchiveEpisodeAdapter().adapt(_archive(actor_id="u2")).events[0]
    other_tenant_archive = _archive()
    other_tenant_archive.metadata["tenant_id"] = "t2"
    other_tenant = SessionArchiveEpisodeAdapter().adapt(other_tenant_archive).events[0]
    other_subject_archive = _archive()
    other_subject_archive.metadata["subjects"] = [{"kind": "person", "id": "u2"}]
    other_subject = SessionArchiveEpisodeAdapter().adapt(other_subject_archive).events[0]
    other_time_archive = _archive()
    other_time_archive.messages[0]["occurred_at"] = "2026-01-02T10:00:00Z"
    other_time = SessionArchiveEpisodeAdapter().adapt(other_time_archive).events[0]
    other_type_archive = _archive()
    other_type_archive.messages[0]["event_type"] = "PREFERENCE"
    other_type = SessionArchiveEpisodeAdapter().adapt(other_type_archive).events[0]
    assert (
        len(
            {
                base.digest,
                other_actor.digest,
                other_tenant.digest,
                other_subject.digest,
                other_time.digest,
                other_type.digest,
            }
        )
        == 6
    )
    ordered_subjects = _archive()
    ordered_subjects.metadata["subjects"] = [
        {"kind": "person", "id": "u1"},
        {"kind": "asset", "id": "device-1"},
        {"kind": "person", "id": "u1"},
        {"kind": "person", "id": "u1", "inferred": True},
    ]
    reversed_subjects = _archive()
    reversed_subjects.metadata["subjects"] = list(reversed(ordered_subjects.metadata["subjects"]))
    assert (
        SessionArchiveEpisodeAdapter().adapt(ordered_subjects).events[0].digest
        == SessionArchiveEpisodeAdapter().adapt(reversed_subjects).events[0].digest
    )
    normalized = SessionArchiveEpisodeAdapter().adapt(ordered_subjects).events[0]
    assert [(item.kind, item.id, item.inferred) for item in normalized.subjects] == [
        ("asset", "device-1", False),
        ("person", "u1", False),
        ("person", "u1", True),
    ]

    left = {
        "map": {"b": 2, "a": 1},
        "set": {"z", "a"},
        "time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "enum": _Kind.RULE,
    }
    right = {
        "enum": _Kind.RULE,
        "time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "set": {"a", "z"},
        "map": {"a": 1, "b": 2},
    }
    assert canonical_digest(left) == canonical_digest(right)


def test_event_envelope_takes_a_deep_immutable_snapshot() -> None:
    archive = _archive()
    event = SessionArchiveEpisodeAdapter().adapt(archive).events[0]
    original_digest = event.digest
    archive.messages[0]["content"] = "caller mutated the source dict"
    archive.metadata["subjects"][0]["id"] = "other"
    assert event.text() == "I confirm SQLite."
    assert event.digest == original_digest
    with pytest.raises(TypeError):
        event.content["content"] = "forbidden"


def test_inferred_envelope_fields_are_explicitly_marked() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="inferred",
        archive_uri="memoryos://user/u1/sessions/history/inferred",
        messages=[{"id": "m1", "content": "I prefer concise answers."}],
        created_at="2026-01-01T00:00:00Z",
    )
    event = SessionArchiveEpisodeAdapter().adapt(archive).events[0]
    assert event.actor.role_inferred
    assert event.subjects[0].inferred
    assert event.occurred_at_inferred
    assert event.ingested_at_inferred
    assert event.sequence_inferred


def test_content_addressed_archive_is_idempotent_and_old_manifests_remain_readable(tmp_path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="t1")
    archive = _archive()
    directory = store.write_sync_archive(archive)
    first_manifest = archive.manifest_digest
    first_event = SessionArchiveEpisodeAdapter().adapt(archive).events[0].digest
    assert not (directory / "messages.jsonl").exists()
    assert not (directory / "commit_manifest.json").exists()

    store.write_sync_archive(archive)
    assert len(list((directory / "evidence" / "events").glob("*.json"))) == 1
    assert len(list((directory / "evidence" / "manifests").glob("*.json"))) == 1

    archive.messages[0]["content"] = "I correct the choice to PostgreSQL."
    store.write_sync_archive(archive)
    second_manifest = archive.manifest_digest
    assert second_manifest != first_manifest
    assert not (directory / "messages.jsonl").exists()
    assert len(list((directory / "evidence" / "events").glob("*.json"))) == 2
    assert (
        store.read_archive_at_manifest(archive.archive_uri, first_manifest, tenant_id="t1").messages[0]["content"]
        == "I confirm SQLite."
    )
    assert store.read_archive(archive.archive_uri, tenant_id="t1").messages[0]["content"] == (
        "I correct the choice to PostgreSQL."
    )
    assert store.read_event(archive.archive_uri, first_event, tenant_id="t1")["event_digest"] == first_event


def test_archive_write_rejects_broken_commit_head_symlink(tmp_path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="t1")
    archive = _archive()
    directory = store._dir(archive.archive_uri, tenant_id="t1")
    directory.mkdir(parents=True, exist_ok=True)
    head = directory / "commit_head.json"
    missing_target = tmp_path / "missing-archive-head.json"
    head.symlink_to(missing_target)

    with pytest.raises(EvidenceArchiveIntegrityError, match="symbolic link"):
        store.write_sync_archive(archive)

    assert head.is_symlink()
    assert not missing_target.exists()


def test_async_commit_reloads_archived_manifest_instead_of_overwriting_from_caller(tmp_path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="t1")
    service = SessionCommitService(store, InMemoryQueueStore())
    archive = _archive(content="hello")
    service.sync_archive(archive, enqueue_commit_job=False)
    archived_manifest = archive.manifest_digest
    archive.messages[0]["content"] = "caller mutation after sync archive"
    result = service.async_commit(archive)
    persisted = store.read_archive_at_manifest(archive.archive_uri, archived_manifest, tenant_id="t1")
    assert result.memory_committed
    assert persisted.messages[0]["content"] == "hello"
    assert store.read_archive(archive.archive_uri, tenant_id="t1").manifest_digest == archived_manifest


def test_existing_event_or_manifest_with_same_digest_and_different_bytes_fails_closed(tmp_path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="t1")
    archive = _archive()
    directory = store.write_sync_archive(archive)
    event = SessionArchiveEpisodeAdapter().adapt(archive).events[0]
    event_path = directory / "evidence" / "events" / f"{event.digest}.json"
    event_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(EvidenceArchiveConflictError, match="different content"):
        store.write_sync_archive(archive)

    # Restore through a separate archive, then prove immutable manifest verification too.
    other = _archive(content="I confirm PostgreSQL.")
    other.session_id = "s2"
    other.archive_uri = "memoryos://user/u1/sessions/history/s2"
    second_directory = store.write_sync_archive(other)
    manifest_path = second_directory / "evidence" / "manifests" / f"{other.manifest_digest}.json"
    manifest_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(EvidenceArchiveIntegrityError, match="manifest digest mismatch"):
        store.read_archive(other.archive_uri, tenant_id="t1")


def test_episode_lookup_preserves_actor_subject_and_independent_event_text() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="field-spans",
        archive_uri="memoryos://user/u1/sessions/history/field-spans",
        messages=[
            {
                "id": "identity",
                "role": "user",
                "actor_id": "u1",
                "content": "The storage backend is under discussion.",
                "occurred_at": "2026-01-01T10:00:00Z",
                "sequence": 1,
            },
            {
                "id": "value",
                "role": "user",
                "actor_id": "u1",
                "content": "I confirm SQLite as the current choice.",
                "occurred_at": "2026-01-01T10:00:01Z",
                "sequence": 2,
            },
        ],
        metadata={"tenant_id": "t1", "subjects": [{"kind": "person", "id": "u1"}]},
        created_at="2026-01-01T10:00:02Z",
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    identity_event = episode.event("identity")
    value_event = episode.event("value")
    assert identity_event is not None and value_event is not None
    assert identity_event.actor.id == "u1"
    assert identity_event.actor.role == "user"
    assert identity_event.subjects[0].id == "u1"
    assert identity_event.text() == "The storage backend is under discussion."
    assert value_event.text() == "I confirm SQLite as the current choice."
    assert identity_event.digest != value_event.digest


def test_v1_archive_layout_is_rejected_instead_of_implicitly_migrated(tmp_path) -> None:
    directory = tmp_path / "tenants" / "default" / "users" / "u1" / "sessions" / "history" / "v1"
    directory.mkdir(parents=True)
    (directory / "messages.jsonl").write_text('{"id":"m1","content":"old"}\n', encoding="utf-8")
    for name in ("observations", "predictions", "action_results", "feedback", "tool_results"):
        (directory / f"{name}.jsonl").write_text("", encoding="utf-8")
    (directory / "commit_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "v1-task",
                "user_id": "u1",
                "session_id": "v1",
                "archive_uri": "memoryos://user/u1/sessions/history/v1",
                "created_at": "2025-01-01T00:00:00Z",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceArchiveIntegrityError, match="missing evidence archive object"):
        SessionArchiveStore(tmp_path).read_archive("memoryos://user/u1/sessions/history/v1")
