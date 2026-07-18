from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from memoryos.adapters.persistence.sqlite import SQLiteIndexStore
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import (
    AUTHORITATIVE_REMEMBER,
    READ_CONTEXT,
    TrustedRequestContext,
)
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.session.session_archive import EvidenceArchiveIntegrityError
from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.memory.documents.layout import user_memory_root
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.runtime.readiness import RuntimeReadinessState


def _caller(tenant_id: str) -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id=tenant_id,
        user_id="same-user",
        actor_kind="user",
        actor_id="same-user",
        capabilities=frozenset({READ_CONTEXT, AUTHORITATIVE_REMEMBER}),
    )


def _remember(client: MemoryOSClient, tenant_id: str, value: str) -> dict:
    return client.remember(
        f"The primary storage backend is {value}.",
        target_hint="topic:primary storage backend",
        caller=_caller(tenant_id),
    )


def test_document_memory_closure_is_tenant_local_and_rebuild_verifies_each_owner(tmp_path: Path) -> None:
    tenant_a = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")
    assert tenant_a.readiness.state == tenant_b.readiness.state == RuntimeReadinessState.READY
    a = _remember(tenant_a, "tenant-a", "PostgreSQL")
    b = _remember(tenant_b, "tenant-b", "SQLite")

    projected_a = tenant_a.memory_projection_worker.process_pending(limit=20)
    projected_b = tenant_b.memory_projection_worker.process_pending(limit=20)
    assert projected_a.processed and projected_a.failed == ()
    assert projected_b.processed and projected_b.failed == ()

    root_a = user_memory_root(tmp_path, "tenant-a", "same-user")
    root_b = user_memory_root(tmp_path, "tenant-b", "same-user")
    raw_a = (root_a / a["relative_path"]).read_text(encoding="utf-8")
    raw_b = (root_b / b["relative_path"]).read_text(encoding="utf-8")
    assert "PostgreSQL" in raw_a and "SQLite" not in raw_a
    assert "SQLite" in raw_b and "PostgreSQL" not in raw_b
    assert a["document_uri"] != b["document_uri"]

    index_a = cast(SQLiteIndexStore, tenant_a.index_store)
    index_b = cast(SQLiteIndexStore, tenant_b.index_store)
    a_records = index_a.list_catalog(
        tenant_id="tenant-a",
        filters={
            "owner_user_id": "same-user",
            "document_ids": (a["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    )
    b_records = index_b.list_catalog(
        tenant_id="tenant-b",
        filters={
            "owner_user_id": "same-user",
            "document_ids": (b["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    )
    assert {record.record_kind for record in a_records} >= {"memory_document", "memory_block"}
    assert {record.record_kind for record in b_records} >= {"memory_document", "memory_block"}
    assert {record.source_digest for record in a_records} == {a["source_digest"]}
    assert {record.source_digest for record in b_records} == {b["source_digest"]}
    assert index_b.list_catalog(
        tenant_id="tenant-b",
        filters={"owner_user_id": "same-user", "document_ids": (a["document_id"],)},
        limit=100,
    ) == []

    assert "PostgreSQL" in tenant_a.read(a["document_uri"], caller=_caller("tenant-a"))["content"]
    with pytest.raises(FileNotFoundError):
        tenant_b.read(a["document_uri"], caller=_caller("tenant-b"))

    assert tenant_b.recovery_worker.process_all()["recovered_count"] == 0
    rebuilt_a = tenant_a.memory_projection_worker.rebuild_owner("tenant-a", "same-user")
    rebuilt_b = tenant_b.memory_projection_worker.rebuild_owner("tenant-b", "same-user")
    verified_a = tenant_a.memory_projection_worker.verify_owner("tenant-a", "same-user")
    verified_b = tenant_b.memory_projection_worker.verify_owner("tenant-b", "same-user")
    assert rebuilt_a["documents"] == verified_a["verified"] == verified_a["projected"]
    assert rebuilt_b["documents"] == verified_b["verified"] == verified_b["projected"]
    assert rebuilt_a["skipped"] >= 1 and rebuilt_b["skipped"] >= 1

    for tenant_id in ("tenant-a", "tenant-b"):
        artifact_root = tmp_path / "tenants" / tenant_id
        assert (artifact_root / "system" / "runtime-layout.json").exists()
        assert not (artifact_root / "system" / "migrations").exists()
        assert not list((artifact_root / "system" / "redo").glob("*.json"))


def test_callerless_archive_and_session_commit_use_client_tenant(tmp_path: Path) -> None:
    tenant_a = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    result = tenant_a.commit_agent_session(
        user_id="same-user",
        session_id="tenant-session",
        messages=[{"id": "m1", "role": "user", "content": "hello"}],
        project_id="memoryos",
        async_commit=False,
    )
    archive = tenant_a.archive_read(result.archive_uri)
    assert archive["archive"]["metadata"]["tenant_id"] == "tenant-a"

    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")
    with pytest.raises(FileNotFoundError):
        tenant_b.archive_read(result.archive_uri)


def test_process_observation_persists_and_queues_in_client_tenant(tmp_path: Path) -> None:
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a", queue_store=queue)
    request = PredictionRequest(
        user_id="same-user",
        episode_id="tenant-observation",
        observation="hot room",
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        request_id="tenant-request",
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    )

    result = client.process_observation(request, archive_session=True, async_commit=False)

    assert result.archive_uri is not None
    persisted = client.archive_read(result.archive_uri)
    assert persisted["archive"]["metadata"]["tenant_id"] == "tenant-a"
    job = queue.lease("session_commit", lease_owner="test", limit=1)[0]
    assert job.payload["tenant_id"] == "tenant-a"
    default_store = SessionArchiveStore(tmp_path)
    default_head = default_store._dir(result.archive_uri) / "commit_head.json"
    assert not default_head.exists()


def test_trusted_caller_rejects_explicit_cross_tenant_override_before_document_write(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    caller = _caller("tenant-a")

    with pytest.raises(PermissionError, match="tenant_id does not match trusted caller"):
        client.remember(
            "The distributed storage backend is CockroachDB.",
            target_hint="topic:distributed storage backend",
            tenant_id="tenant-b",
            caller=caller,
        )

    assert not user_memory_root(tmp_path, "tenant-a", "same-user").exists()
    assert not user_memory_root(tmp_path, "tenant-b", "same-user").exists()
    assert client.queue_store.stats(queue_name="memory_projection").get("pending", 0) == 0


@pytest.mark.parametrize(
    ("metadata", "method"),
    [
        ({"tenant_id": "tenant-b"}, "sync_archive"),
        ({"scope": {"tenant_id": "tenant-b"}}, "sync_archive"),
        ({"tenant_id": "tenant-a", "scope": {"tenant_id": "tenant-b"}}, "async_commit"),
    ],
)
def test_session_commit_rejects_cross_tenant_archive_before_any_artifact(
    tmp_path: Path,
    metadata: dict,
    method: str,
) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    queue = InMemoryQueueStore()
    service = SessionCommitService(store, queue)
    archive = SessionArchive(
        user_id="same-user",
        session_id="cross-tenant",
        archive_uri="memoryos://user/same-user/sessions/history/cross-tenant",
        messages=[{"id": "m1", "role": "user", "content": "hello"}],
        metadata=metadata,
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    with pytest.raises(PermissionError, match="bound store"):
        getattr(service, method)(archive)

    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert queue.lease("session_commit", lease_owner="test", limit=1) == []


def test_session_commit_materializes_missing_tenant_from_bound_store(tmp_path: Path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    queue = InMemoryQueueStore()
    service = SessionCommitService(store, queue)
    archive = SessionArchive(
        user_id="same-user",
        session_id="bound-default",
        archive_uri="memoryos://user/same-user/sessions/history/bound-default",
        messages=[{"id": "m1", "role": "user", "content": "hello"}],
        metadata={"project_id": "memoryos"},
    )

    service.sync_archive(archive)

    assert archive.metadata["tenant_id"] == "tenant-a"
    persisted = store.read_archive(archive.archive_uri, tenant_id="tenant-a")
    assert persisted.metadata["tenant_id"] == "tenant-a"
    job = queue.lease("session_commit", lease_owner="test", limit=1)[0]
    assert job.payload["tenant_id"] == "tenant-a"
    default_head = store._dir(archive.archive_uri, tenant_id="default") / "commit_head.json"
    assert not default_head.exists()


def test_session_archive_store_rejects_cross_tenant_write_before_any_artifact(tmp_path: Path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    archive = SessionArchive(
        user_id="same-user",
        session_id="raw-cross-tenant",
        archive_uri="memoryos://user/same-user/sessions/history/raw-cross-tenant",
        messages=[{"id": "m1", "role": "user", "content": "hello"}],
        metadata={"tenant_id": "tenant-b"},
    )

    with pytest.raises(EvidenceArchiveIntegrityError, match="bound archive store"):
        store.write_sync_archive(archive)

    assert not list(tmp_path.rglob("*"))
