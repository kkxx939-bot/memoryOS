from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.index_consistency import IndexConsistencyService
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.canonical.current_head import head_set_path, load_current_head
from memoryos.memory.canonical.projection_state import ProjectionIntegrityError
from memoryos.memory.canonical.visibility import (
    CommittedStateIntegrityError,
    list_committed_canonical,
)
from memoryos.providers.embedding import HashingEmbeddingProvider
from memoryos.runtime.readiness import RuntimeNotReadyError, RuntimeReadinessState
from memoryos.workers.reindex_worker import ReindexWorker


def _derived_snapshot(client: MemoryOSClient, vectors: InMemoryVectorStore) -> dict:
    view_root = Path(client.memory_projection_worker.projector.root) / "views"
    return {
        "index": {
            uri: copy.deepcopy(client.index_store.get_index_metadata(uri))
            for uri in sorted(client.index_store.indexed_uris())
        },
        "vectors": copy.deepcopy(vectors.rows),
        "views": {
            str(path.relative_to(view_root)): path.read_bytes()
            for path in sorted(view_root.rglob("*"))
            if path.is_file()
        }
        if view_root.exists()
        else {},
    }


def _tamper_rebuild_authority(client: MemoryOSClient, claim_uri: str, artifact: str) -> None:
    if artifact == "current_head":
        path = head_set_path(
            client.memory_projection_worker.projector.root,
            claim_uri.rsplit("/claims/", 1)[0],
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["heads"][claim_uri]["head_digest"] = "0" * 64
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return
    if artifact == "projection_record":
        store = client.memory_projection_worker.projector.record_store
        record = store.load_current(claim_uri, source_revision=1)
        assert record is not None
        path = store.attempt_path_for(record)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["record_digest"] = "0" * 64
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return
    if artifact == "source_bundle":
        object_dir = client.source_store._object_dir(claim_uri)  # type: ignore[attr-defined]
        pointer = json.loads((object_dir / ".bundle-current.json").read_text(encoding="utf-8"))
        content = object_dir / ".bundle-generations" / str(pointer["generation_id"]) / "content.md"
        content.write_text(content.read_text(encoding="utf-8") + "\ntampered", encoding="utf-8")
        return
    if artifact == "projection_publication":
        client.memory_projection_worker.validate_projection_proofs()
        head, _receipt, _snapshot = load_current_head(
            client.memory_projection_worker.projector.root,
            claim_uri,
        )
        path = client.memory_projection_worker.proof_store.publication_path(str(head["current_transaction_id"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["publication_digest"] = "0" * 64
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return
    raise AssertionError(f"unsupported authority artifact: {artifact}")


def test_generic_rebuild_preserves_formal_projection_and_excludes_raw_canonical(
    tmp_path: Path,
) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(str(tmp_path), vector_store=vectors)
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    before = client.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )
    assert [item["uri"] for item in before] == [committed["uri"]]

    raw_uri = "memoryos://user/u1/memories/canonical/slots/raw/claims/uncommitted"
    raw = ContextObject.from_dict(client.source_store.read_object(committed["uri"]).to_dict())
    raw.uri = raw_uri
    raw.title = "raw canonical bypass"
    client.source_store.write_object(raw, content="raw uncommitted canonical")
    stale_uri = "memoryos://user/u1/memories/canonical/slots/stale/claims/stale"
    stale = ContextObject.from_dict(raw.to_dict())
    stale.uri = stale_uri
    stale.title = "stale canonical row"
    schema_only_uri = "memoryos://user/u1/memories/schema-only-uncommitted"
    schema_only = ContextObject.from_dict(raw.to_dict())
    schema_only.uri = schema_only_uri
    schema_only.title = "schema only canonical bypass"
    schema_only.metadata.pop("canonical_kind", None)
    client.source_store.write_object(schema_only, content="schema only canonical bypass")
    vectors.upsert_vector(schema_only_uri, [1.0], metadata={"schema_version": "generic_vector_v1"})

    def seed_raw_index_rows() -> None:
        client.index_store.upsert_index(raw, content="raw uncommitted canonical")
        client.index_store.upsert_index(stale, content="stale")
        client.index_store.upsert_index(schema_only, content="schema only canonical bypass")

    projection_record = client.memory_projection_worker.projector.record_store.load_current(
        committed["uri"],
        source_revision=1,
    )
    assert projection_record is not None
    projected_index_metadata = client.index_store.get_index_metadata(committed["uri"])
    projected_vector_metadata = vectors.get_vector_metadata(committed["uri"])
    client.memory_projection_worker.validate_projection_proofs()
    projection_proofs = client.memory_projection_worker.validate_projection_proofs()
    assert client.memory_projection_worker.verify_current_projections() == {"verified": 1}

    seed_raw_index_rows()
    assert (
        client.context_db.search(
            "raw uncommitted canonical",
            owner_user_id="u1",
            context_type=ContextType.MEMORY,
        )
        == []
    )

    IndexConsistencyService(
        client.source_store,
        client.index_store,
        client.relation_store,
    ).rebuild()
    assert committed["uri"] in client.index_store.indexed_uris()
    assert raw_uri not in client.index_store.indexed_uris()
    assert stale_uri not in client.index_store.indexed_uris()
    assert schema_only_uri not in client.index_store.indexed_uris()
    assert client.index_store.get_index_metadata(committed["uri"]) == projected_index_metadata
    assert vectors.get_vector_metadata(committed["uri"]) == projected_vector_metadata
    assert (
        client.memory_projection_worker.projector.record_store.load_current(
            committed["uri"],
            source_revision=1,
        )
        == projection_record
    )
    assert client.memory_projection_worker.verify_current_projections() == {"verified": 1}
    assert client.memory_projection_worker.validate_projection_proofs() == projection_proofs

    seed_raw_index_rows()
    ReindexWorker(client.source_store, client.index_store).rebuild()
    assert committed["uri"] in client.index_store.indexed_uris()
    assert raw_uri not in client.index_store.indexed_uris()
    assert stale_uri not in client.index_store.indexed_uris()
    assert schema_only_uri not in client.index_store.indexed_uris()
    assert client.index_store.get_index_metadata(committed["uri"]) == projected_index_metadata
    assert vectors.get_vector_metadata(committed["uri"]) == projected_vector_metadata
    assert (
        client.memory_projection_worker.projector.record_store.load_current(
            committed["uri"],
            source_revision=1,
        )
        == projection_record
    )
    assert client.memory_projection_worker.verify_current_projections() == {"verified": 1}
    assert client.memory_projection_worker.validate_projection_proofs() == projection_proofs

    client.context_db.rebuild_index()
    indexed = set(client.index_store.indexed_uris())
    assert committed["uri"] in indexed
    assert raw_uri not in indexed
    assert stale_uri not in indexed
    assert schema_only_uri not in indexed
    assert schema_only_uri not in vectors.vector_uris()
    rebuilt_record = client.memory_projection_worker.projector.record_store.load_current(
        committed["uri"],
        source_revision=1,
    )
    assert rebuilt_record is not None
    assert rebuilt_record.projection_attempt_id != projection_record.projection_attempt_id
    assert client.memory_projection_worker.verify_current_projections() == {"verified": 1}
    after = client.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )
    assert [item["uri"] for item in after] == [item["uri"] for item in before]
    assert [item["metadata"]["canonical_value"] for item in after] == [
        item["metadata"]["canonical_value"] for item in before
    ]


def test_generic_rebuild_fails_before_mutation_for_corrupt_committed_projection(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    ordinary = ContextObject(
        uri="memoryos://user/u1/memories/profile/ordinary",
        context_type=ContextType.MEMORY,
        title="ordinary",
        owner_user_id="u1",
    )
    client.source_store.write_object(ordinary, content="ordinary content")
    client.index_store.upsert_index(ordinary, content="ordinary content")

    raw_claim = client.source_store.read_object(committed["uri"])
    client.index_store.upsert_index(raw_claim, content="raw Source is not a formal projection")

    with pytest.raises(ProjectionIntegrityError, match="invalid canonical index projection"):
        IndexConsistencyService(client.source_store, client.index_store).rebuild()

    assert ordinary.uri in client.index_store.indexed_uris()
    assert committed["uri"] in client.index_store.indexed_uris()


@pytest.mark.parametrize(
    "artifact",
    ["current_head", "projection_record", "source_bundle", "projection_publication"],
)
@pytest.mark.parametrize("action", ["verify", "rebuild"])
def test_contextdb_consistency_fails_closed_before_derived_mutation(
    tmp_path: Path,
    artifact: str,
    action: str,
) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": f"{artifact}-{action}"},
    )
    before = _derived_snapshot(client, vectors)
    _tamper_rebuild_authority(client, str(committed["uri"]), artifact)

    operation = client.context_db.verify_consistency if action == "verify" else client.context_db.rebuild_index
    with pytest.raises(RuntimeError):
        operation()

    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    assert _derived_snapshot(client, vectors) == before


@pytest.mark.parametrize(
    "artifact",
    ["current_head", "projection_record", "source_bundle", "projection_publication"],
)
def test_projector_rebuild_preflights_before_clearing_derived_state(
    tmp_path: Path,
    artifact: str,
) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": f"direct-projector-{artifact}"},
    )
    before = _derived_snapshot(client, vectors)
    _tamper_rebuild_authority(client, str(committed["uri"]), artifact)

    with pytest.raises(RuntimeError):
        client.memory_projection_worker.projector.rebuild(clear_views=True)

    assert _derived_snapshot(client, vectors) == before


def test_contextdb_rebuild_repairs_only_derived_projection_state(tmp_path: Path) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "derived projection repair"},
    )
    claim_uri = str(committed["uri"])
    before = client.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )
    original = client.memory_projection_worker.projector.record_store.load_current(
        claim_uri,
        source_revision=1,
    )
    assert original is not None
    client.index_store.delete_index(claim_uri)
    vectors.delete_vector(claim_uri)

    inconsistent = client.context_db.verify_consistency()

    assert inconsistent["consistent"] is False
    assert inconsistent["canonical_projection_error"]
    assert client.readiness.state == RuntimeReadinessState.READY

    rebuilt = client.context_db.rebuild_index()

    assert rebuilt["consistent"] is True
    assert rebuilt["canonical_projection_validation"]["verified"] == 1
    assert client.index_store.get_index_metadata(claim_uri) is not None
    assert vectors.get_vector_metadata(claim_uri) is not None
    current = client.memory_projection_worker.projector.record_store.load_current(
        claim_uri,
        source_revision=1,
    )
    assert current is not None
    assert current.projection_attempt_id != original.projection_attempt_id
    assert client.memory_projection_worker.verify_current_projections() == {"verified": 1}
    assert client.readiness.state == RuntimeReadinessState.READY
    after = client.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )
    assert [item["uri"] for item in after] == [item["uri"] for item in before]


def test_contextdb_rebuild_ignores_unrelated_session_queue_terminal_state(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "queue kind isolation"},
    )
    unrelated = client.queue_store.enqueue(
        QueueJob(
            job_id="unrelated-session-dead-letter",
            queue_name="session_commit",
            action="commit_session",
            target_uri="memoryos://user/u1/sessions/history/unrelated",
        )
    )
    leased = client.queue_store.lease(
        unrelated.queue_name,
        lease_owner="failed-session-worker",
        job_ids=(unrelated.job_id,),
    )[0]
    client.queue_store.fail(leased, "unrelated session failure")

    assert client.queue_store.stats().get("dead_letter") == 1
    assert client.queue_store.stats(queue_name="memory_projection").get("dead_letter", 0) == 0
    assert client.context_db.verify_consistency()["consistent"] is True
    assert client.context_db.rebuild_index()["consistent"] is True
    assert client.index_store.get_index_metadata(str(committed["uri"])) is not None
    assert client.readiness.state == RuntimeReadinessState.READY


def test_contextdb_rebuild_rejects_projection_queue_terminal_before_mutation(tmp_path: Path) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "projection queue terminal"},
    )
    head, _receipt, _snapshot = load_current_head(
        client.memory_projection_worker.projector.root,
        str(committed["uri"]),
    )
    job_id = f"outbox_{head['current_transaction_id']}"
    queue_path = getattr(client.queue_store, "path", None)
    assert isinstance(queue_path, Path)
    with sqlite3.connect(queue_path) as connection:
        connection.execute(
            "UPDATE queue_jobs SET status = 'dead_letter', last_error = 'injected' WHERE job_id = ?",
            (job_id,),
        )
    before = _derived_snapshot(client, vectors)

    with pytest.raises(RuntimeError, match="terminal before publication"):
        client.context_db.rebuild_index()

    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    assert _derived_snapshot(client, vectors) == before


def test_startup_recovers_expired_projection_lease_before_rebuild_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    monkeypatch.setattr(
        client.memory_projection_worker,
        "process_pending",
        lambda *args, **kwargs: {
            "processed": [],
            "stale": [],
            "failed": [],
            "dead_letter": [],
            "quarantine": [],
            "released": [],
        },
    )
    with pytest.raises(RuntimeError, match="serving projection remains pending"):
        client.remember(
            user_id="u1",
            content="PostgreSQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields={"decision_topic": "expired projection lease recovery"},
        )
    lease = client.queue_store.lease(
        "memory_projection",
        lease_owner="crashed-projection-worker",
        limit=1,
    )[0]
    job_id = lease.job_id
    committed_uri = lease.target_uri
    queue_path = getattr(client.queue_store, "path", None)
    assert isinstance(queue_path, Path)
    with sqlite3.connect(queue_path) as connection:
        connection.execute(
            "UPDATE queue_jobs SET leased_until = '1970-01-01T00:00:00+00:00' WHERE job_id = ?",
            (job_id,),
        )

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons
    settled = restarted.queue_store.get(job_id)
    assert settled is not None and settled.status == "done"
    assert settled.lease_generation == lease.lease_generation + 1
    assert restarted.readiness.details["queue_lease_recovery"] == {"recovered_expired": 1}
    current_rows = restarted.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "default",
            "canonical_slot_ids": (committed_uri.rsplit("/", 1)[-1],),
            "record_kinds": ("current_slot",),
        },
        limit=2,
    )
    assert len(current_rows) == 1
    assert current_rows[0].canonical_slot_uri == committed_uri
    assert restarted.memory_projection_worker.verify_current_projections() == {"verified": 1}


def test_startup_never_reports_ready_with_unconsumed_projection_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    empty_run: dict[str, list[str]] = {
        "processed": [],
        "stale": [],
        "failed": [],
        "dead_letter": [],
        "quarantine": [],
        "released": [],
    }
    monkeypatch.setattr(
        client.memory_projection_worker,
        "process_pending",
        lambda *args, **kwargs: dict(empty_run),
    )
    with pytest.raises(RuntimeError, match="serving projection remains pending"):
        client.remember(
            user_id="u1",
            content="PostgreSQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields={"decision_topic": "unconsumed startup projection"},
        )
    leased = client.queue_store.lease(
        "memory_projection",
        lease_owner="test-unconsumed-projection",
        limit=1,
    )[0]
    job_id = leased.job_id
    client.queue_store.release(leased, "leave durable work for startup gate")
    queued = client.queue_store.get(job_id)
    assert queued is not None and queued.status == "pending"
    worker_type = type(client.memory_projection_worker)
    monkeypatch.setattr(
        worker_type,
        "_process_pending_during_startup",
        lambda *args, **kwargs: dict(empty_run),
    )

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "still contains pending work" in " ".join(restarted.readiness.reasons)
    remaining = restarted.queue_store.get(job_id)
    assert remaining is not None and remaining.status == "pending"


def test_rebuild_accepts_equivalent_resolved_projection_record_paths() -> None:
    # macOS exposes the same temporary directory through /var and
    # /private/var.  Projection identity is the resolved file, not the spelling
    # used by the process that originally published it.
    with TemporaryDirectory() as root:
        client = MemoryOSClient(root)
        committed = client.remember(
            user_id="u1",
            content="PostgreSQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields={"decision_topic": "resolved projection path"},
        )

        assert client.context_db.verify_consistency()["consistent"] is True
        rebuilt = client.context_db.rebuild_index()

        assert rebuilt["consistent"] is True
        assert client.index_store.get_index_metadata(str(committed["uri"])) is not None
        assert client.memory_projection_worker.verify_current_projections() == {"verified": 1}


def test_contextdb_read_rejects_raw_canonical_without_current_head(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    raw_uri = "memoryos://user/u1/memories/canonical/slots/raw/claims/uncommitted"
    raw = ContextObject(
        uri=raw_uri,
        context_type=ContextType.MEMORY,
        title="raw canonical bypass",
        owner_user_id="u1",
        metadata={"canonical_kind": "claim", "revision": 1, "state": "ACTIVE"},
    )
    client.source_store.write_object(raw, content="raw")
    with pytest.raises(PermissionError, match="cannot be seeded through ContextDB"):
        client.context_db.seed_object(raw, content="raw staged bootstrap")
    assert raw_uri not in client.index_store.indexed_uris()

    try:
        client.context_db.read_object(raw_uri)
    except FileNotFoundError as exc:
        assert "not committed" in str(exc)
    else:  # pragma: no cover - explicit fail message is clearer than a context manager here.
        raise AssertionError("raw canonical Source state became visible through ContextDB")


def test_hybrid_search_rejects_canonical_metadata_at_a_noncanonical_uri(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    raw = ContextObject.from_dict(client.source_store.read_object(committed["uri"]).to_dict())
    raw.uri = "memoryos://user/u1/memories/raw-canonical-metadata"
    raw.title = "raw proof bypass token"
    client.source_store.write_object(raw, content="raw proof bypass token")
    client.index_store.upsert_index(raw, content="raw proof bypass token")
    schema_only = ContextObject.from_dict(raw.to_dict())
    schema_only.uri = "memoryos://user/u1/memories/raw-canonical-schema"
    schema_only.title = "schema proof bypass token"
    schema_only.metadata.pop("canonical_kind", None)
    client.source_store.write_object(schema_only, content="schema proof bypass token")
    client.index_store.upsert_index(schema_only, content="schema proof bypass token")

    hits = HybridSearch(client.index_store, source_store=client.source_store).search(
        "raw proof bypass token",
        filters={"owner_user_id": "u1", "tenant_id": "default"},
        context_type=ContextType.MEMORY,
    )

    assert hits == []
    assert (
        client.context_db.search(
            "schema proof bypass token",
            owner_user_id="u1",
            tenant_id="default",
            context_type=ContextType.MEMORY,
        )
        == []
    )
    with pytest.raises(PermissionError, match="cannot be seeded through ContextDB"):
        client.context_db.seed_object(schema_only, content="schema proof bypass token")
    with pytest.raises(PermissionError, match="canonical relations"):
        client.context_db.add_relation(
            ContextRelation(
                source_uri=schema_only.uri,
                relation_type="related_to",
                target_uri="memoryos://user/u1/memories/ordinary",
            )
        )


def test_sdk_read_fails_closed_and_marks_runtime_not_ready_on_unproved_source_tamper(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    claim_uri = result["uri"]
    assert client.read(claim_uri, layer="L0")["content"]
    raw = client.source_store.read_object(claim_uri)
    raw.title = "UNCOMMITTED MySQL"
    client.source_store.write_object(raw, content="UNCOMMITTED MySQL")

    with pytest.raises(RuntimeError, match="without an in-flight redo proof"):
        client.read(claim_uri)
    assert client.readiness.state.value == "NOT_READY"
    with pytest.raises(RuntimeError, match="runtime is NOT_READY"):
        client.search_context(
            "PostgreSQL",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
        )


def test_missing_current_head_for_committed_slot_makes_startup_not_ready(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    slot_uri = result["uri"].rsplit("/claims/", 1)[0]
    head_set_path(tmp_path, slot_uri).unlink()

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state.value == "NOT_READY"
    assert any("current head" in reason and "missing" in reason for reason in restarted.readiness.reasons)


def test_deleted_canonical_head_fails_closed_for_search_and_head_enumeration(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="PostgreSQL enumeration proof",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "enumeration storage backend"},
    )
    slot_uri = result["uri"].rsplit("/claims/", 1)[0]
    head_set_path(tmp_path, slot_uri).unlink()

    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        client.search_context(
            "PostgreSQL enumeration proof",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
        )
    assert client.readiness.state.value == "NOT_READY"

    with pytest.raises(CommittedStateIntegrityError, match="required current head is missing or stale"):
        list_committed_canonical(
            client.source_store,
            client.relation_store,
            kinds=("slot", "claim"),
        )


def test_deleted_pending_head_fails_closed_for_list_and_restart(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "pending enumeration backend"},
    )
    pending_result = client.remember(
        user_id="u1",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "pending enumeration backend"},
    )
    assert pending_result["status"] == "PENDING"
    pending_uri = pending_result["uri"]
    head_set_path(tmp_path, pending_uri).unlink()

    with pytest.raises(CommittedStateIntegrityError, match="required current head is missing or stale"):
        client.list_pending(user_id="u1", lifecycle_states=["PENDING"])
    assert client.readiness.state.value == "NOT_READY"

    restarted = MemoryOSClient(str(tmp_path))
    assert restarted.readiness.state.value == "NOT_READY"
    assert any("required current head is missing" in reason for reason in restarted.readiness.reasons)


def test_empty_filesystem_has_no_receipt_head_coverage_error(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))

    assert client.search_context("nothing", user_id="u1", context_type="memory") == []
    assert client.list_pending(user_id="u1") == []
    assert list_committed_canonical(client.source_store, client.relation_store) == ()
    assert client.readiness.state.value == "READY"


def test_first_receipt_before_head_remains_invisible_while_exact_redo_exists(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))

    def crash_after_receipt(stage: str, _transaction_id: str) -> None:
        if stage == "after_receipt":
            raise SystemExit("crash after immutable receipt")

    client.committer.test_hook = crash_after_receipt
    with pytest.raises(SystemExit, match="after immutable receipt"):
        client.remember(
            user_id="u1",
            content="SQLite in-flight enumeration proof",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields={"decision_topic": "in-flight enumeration backend"},
        )

    assert list_committed_canonical(client.source_store, client.relation_store) == ()
    assert (
        client.search_context(
            "SQLite in-flight enumeration proof",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
        )
        == []
    )
    assert client.readiness.state.value == "READY"

    restarted = MemoryOSClient(str(tmp_path))
    assert restarted.readiness.state.value == "READY"
    assert (
        len(
            restarted.search_context(
                "SQLite in-flight enumeration proof",
                user_id="u1",
                project_id="memoryos",
                context_type="memory",
            )
        )
        == 1
    )


def test_live_read_of_committed_uri_with_deleted_head_fails_closed(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    slot_uri = result["uri"].rsplit("/claims/", 1)[0]
    head_set_path(tmp_path, slot_uri).unlink()

    with pytest.raises(RuntimeError, match="required current head is missing"):
        client.read(result["uri"])
    assert client.readiness.state.value == "NOT_READY"


def test_retrieval_ignores_a_live_tampered_projection_and_uses_committed_source(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    record = client.memory_projection_worker.projector.record_store.load_current(
        result["uri"],
        source_revision=1,
    )
    assert record is not None
    client.source_store.write_content(record.l0_uri, "tampered projection bait")

    visible = client.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )

    assert [item["uri"] for item in visible] == [result["uri"]]
    assert visible[0]["text"] == "PostgreSQL"
    assert visible[0]["layer"] == "canonical_source"
    assert (
        client.search_context(
            "tampered projection bait",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
        )
        == []
    )


def test_startup_rebuilds_canonical_relation_store_from_current_receipts(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    claim_uri = result["uri"]
    slot_uri = claim_uri.rsplit("/claims/", 1)[0]
    client.relation_store.delete_relation(claim_uri, "belongs_to_slot", slot_uri)
    client.relation_store.delete_relation(slot_uri, "has_claim", claim_uri)
    client.relation_store.add_relation(
        ContextRelation(
            source_uri=claim_uri,
            relation_type="stale_uncommitted",
            target_uri="memoryos://user/u1/memories/canonical/slots/stale",
            metadata={"tenant_id": "default", "owner_user_id": "u1"},
        )
    )

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.health()["runtime"]["state"] == "READY"
    claim_relations = restarted.context_db.relations_of(
        claim_uri,
        tenant_id="default",
        owner_user_id="u1",
    )
    slot_relations = restarted.context_db.relations_of(
        slot_uri,
        tenant_id="default",
        owner_user_id="u1",
    )
    expected_relations = {
        ("belongs_to_slot", slot_uri),
        ("has_claim", claim_uri),
    }
    assert {(item.relation_type, item.target_uri) for item in claim_relations} == expected_relations
    assert {(item.relation_type, item.target_uri) for item in slot_relations} == expected_relations
    assert all(item.relation_type != "stale_uncommitted" for item in restarted.relation_store.relations_of(claim_uri))

    second_restart = MemoryOSClient(str(tmp_path))
    relation_recovery = second_restart.health()["runtime"]["details"]["canonical_relations"]
    assert relation_recovery["written"] == 0
    assert relation_recovery["deleted"] == 0


def test_startup_removes_scope_and_taxonomy_views_without_a_committed_head(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    stale_paths = (
        tmp_path / "views" / "scope" / "stale" / "current.json",
        tmp_path / "views" / "taxonomy" / "stale" / "current.json",
    )
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"claim_uri":"memoryos://user/u1/memories/canonical/slots/stale/claims/stale"}')

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state.value == "READY"
    assert all(not path.exists() for path in stale_paths)
