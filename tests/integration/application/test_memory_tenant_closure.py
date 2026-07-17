from __future__ import annotations

from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.session.session_archive import EvidenceArchiveIntegrityError
from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.runtime.readiness import RuntimeReadinessState


def _remember(client: MemoryOSClient, value: str) -> dict:
    return client.remember(
        user_id="same-user",
        content=value,
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )


def test_callerless_sdk_keeps_all_memory_closure_artifacts_tenant_local(tmp_path: Path) -> None:
    tenant_a = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")
    assert tenant_a.readiness.state == tenant_b.readiness.state == RuntimeReadinessState.READY
    a = _remember(tenant_a, "PostgreSQL")
    b = _remember(tenant_b, "SQLite")

    a_rows = tenant_a.search_context(
        "storage",
        user_id="same-user",
        project_id="memoryos",
        context_type="memory",
    )
    b_rows = tenant_b.search_context(
        "storage",
        user_id="same-user",
        project_id="memoryos",
        context_type="memory",
    )
    assert {item["metadata"]["canonical_value"] for item in a_rows} == {"PostgreSQL"}
    assert {item["metadata"]["canonical_value"] for item in b_rows} == {"SQLite"}
    assert a["uri"] != b["uri"]

    pending = _remember(tenant_a, "MySQL")
    assert pending["status"] == "PENDING"
    reviewable = tenant_a.list_pending(user_id="same-user")[0]
    assert tenant_b.list_pending(user_id="same-user") == []
    with pytest.raises(FileNotFoundError):
        tenant_b.review_pending(
            user_id="same-user",
            pending_uri=reviewable["uri"],
            decision="REJECT",
            expected_lifecycle_revision=reviewable["lifecycle_revision"],
            expected_proposal_fingerprint=reviewable["proposal_fingerprint"],
            command_id="cross-tenant-review",
        )

    assert tenant_b.recovery_worker.process_all()["recovered_count"] == 0
    assert tenant_a.memory_projection_worker.verify_current_projections()["verified"] == 1
    assert tenant_b.memory_projection_worker.verify_current_projections()["verified"] == 1
    tenant_a.context_db.rebuild_index()
    tenant_b.context_db.rebuild_index()
    assert a["uri"] in tenant_a.index_store.indexed_uris()
    assert a["uri"] not in tenant_b.index_store.indexed_uris()
    assert b["uri"] in tenant_b.index_store.indexed_uris()
    assert b["uri"] not in tenant_a.index_store.indexed_uris()

    for tenant_id in ("tenant-a", "tenant-b"):
        artifact_root = tmp_path / "tenants" / tenant_id
        assert (artifact_root / "system" / "migrations" / "memory-closure-v1.json").exists()
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


def test_callerless_explicit_tenant_override_routes_to_a_tenant_bound_runtime(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")

    committed = client.remember(
        user_id="same-user",
        content="CockroachDB",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "distributed storage backend"},
        tenant_id="tenant-b",
    )

    assert (
        client.search_context(
            "CockroachDB",
            user_id="same-user",
            project_id="memoryos",
            context_type="memory",
        )
        == []
    )
    routed = client.search_context(
        "CockroachDB",
        user_id="same-user",
        project_id="memoryos",
        context_type="memory",
        tenant_id="tenant-b",
    )
    assert [item["uri"] for item in routed] == [committed["uri"]]
    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")
    assert tenant_b.read(committed["uri"])["object"]["tenant_id"] == "tenant-b"


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
    service = SessionCommitService(store, queue, allow_plan_only=True)
    archive = SessionArchive(
        user_id="same-user",
        session_id="cross-tenant",
        archive_uri="memoryos://user/same-user/sessions/history/cross-tenant",
        messages=[{"id": "m1", "role": "user", "content": "hello"}],
        metadata=metadata,
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    with pytest.raises(PermissionError, match="bound archive store"):
        getattr(service, method)(archive)

    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert queue.lease("session_commit", lease_owner="test", limit=1) == []


def test_session_commit_materializes_missing_tenant_from_bound_store(tmp_path: Path) -> None:
    store = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    queue = InMemoryQueueStore()
    service = SessionCommitService(store, queue, allow_plan_only=True)
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
