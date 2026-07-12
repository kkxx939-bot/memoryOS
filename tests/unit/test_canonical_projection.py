from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
)
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.canonical import (
    CanonicalMemoryProjector,
    EvidenceRef,
    MemoryClaim,
    MemoryProjectionWorker,
    MemoryRevision,
    ProjectionIntegrityError,
    ProjectionRecord,
    TransitionProfile,
)
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.operations.commit.effect_marker import (
    atomic_write_json,
    build_marker,
    object_effect_from_store,
)
from memoryos.operations.commit.outbox_envelope import build_outbox, planned_effect_manifest
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class FailingVectorStore(InMemoryVectorStore):
    def upsert_vector(self, uri, embedding, metadata=None):  # noqa: ANN001, ANN201
        raise RuntimeError("vector unavailable")


class CountingIndexStore(InMemoryIndexStore):
    def __init__(self) -> None:
        super().__init__()
        self.upsert_calls = 0

    def upsert_index(self, obj, content=""):  # noqa: ANN001, ANN201
        self.upsert_calls += 1
        return super().upsert_index(obj, content)


class SQLiteTestVectorStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors "
                "(uri TEXT PRIMARY KEY, embedding_json TEXT NOT NULL, metadata_json TEXT NOT NULL)"
            )

    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None:
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute(
                "INSERT INTO vectors VALUES (?, ?, ?) ON CONFLICT(uri) DO UPDATE SET "
                "embedding_json=excluded.embedding_json, metadata_json=excluded.metadata_json",
                (uri, json.dumps(embedding), json.dumps(metadata or {}, sort_keys=True)),
            )

    def delete_vector(self, uri: str) -> None:
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("DELETE FROM vectors WHERE uri = ?", (uri,))

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[Any]:  # noqa: ARG002
        return []


def _claim(revision: int, state: str = "ACTIVE") -> MemoryClaim:
    revisions = tuple(
        MemoryRevision(
            revision=index,
            state=state if index == revision else "PROPOSED",
            value_fields={"canonical_value": "SQLite"},
            evidence_refs=(EvidenceRef("e1", None, "hash"),),
            proposal_id=f"p{index}",
            relation="UNRELATED",
            epistemic_status="EXPLICIT",
        )
        for index in range(1, revision + 1)
    )
    return MemoryClaim(
        "claim1",
        "memoryos://user/u1/memories/canonical/slots/slot1/claims/claim1",
        "slot1",
        "sqlite",
        TransitionProfile.AUTHORITATIVE_STATE,
        revisions,
    )


def _scope() -> dict:
    workspace: dict[str, object] = {
        "namespace": "memoryos",
        "kind": "workspace",
        "id": "memoryos",
        "parent_id": None,
        "attributes": {},
        "confidence": 1.0,
        "source": "explicit",
        "inferred": False,
    }
    return {
        "canonical_subject": workspace,
        "applicability": {"all_of": [workspace]},
        "visibility": {
            "tenant_id": "t1",
            "allowed_principal_ids": [],
            "allowed_service_ids": [],
            "private": False,
        },
        "authority": {
            "principal_ids": ["u1"],
            "service_ids": [],
            "inferred": False,
        },
        "origin_refs": [],
    }


def _artifact_root(root):  # noqa: ANN001, ANN202
    return root / "tenants" / "t1"


def _persist_committed_claim(source, artifact_root, obj, content: str) -> None:  # noqa: ANN001
    revision = int(obj.metadata.get("revision", 0))
    idempotency_key = f"projection-fixture-revision-{revision}"
    transaction_id = f"projection-fixture-transaction-{revision}"
    obj.metadata = {
        **obj.metadata,
        "canonical_idempotency_key": idempotency_key,
        "canonical_transaction_id": transaction_id,
    }
    source.write_object(obj, content=content)
    marker = artifact_root / "system" / "transactions" / f"{idempotency_key}.json"
    operation_id = f"projection-fixture-operation-{revision}"
    atomic_write_json(
        marker,
        build_marker(
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            tenant_id="t1",
            user_id="u1",
            operation_ids=[operation_id],
            object_effects=[object_effect_from_store(source, obj.uri, operation_type="fixture")],
            relation_effects=[],
            diff={"user_id": "u1", "operations": [], "diff_id": f"fixture-{revision}"},
            operations=[],
        ),
    )


def _enqueue(queue, tmp_path, revision: int, job_id: str) -> None:  # noqa: ANN001
    outbox = tmp_path / "system" / "outbox" / f"{job_id}.json"
    uri = "memoryos://user/u1/memories/canonical/slots/slot1/claims/claim1"
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        target_uri=uri,
        operation_id=f"op-{job_id}",
        payload={
            "transaction_id": job_id,
            "idempotency_key": f"projection-fixture-revision-{revision}",
            "tenant_id": "t1",
            "context_object": {
                "uri": uri,
                "metadata": {"revision": revision},
            },
            "content": "",
        },
    )
    effect = planned_effect_manifest(operation, {})
    atomic_write_json(
        outbox,
        build_outbox(
            transaction_id=job_id,
            idempotency_key=f"projection-fixture-revision-{revision}",
            tenant_id="t1",
            user_id="u1",
            operations=[operation],
            status="committed",
            before_images=[],
            effect_manifests=[effect],
            claim_revisions=[{"uri": uri, "claim_id": "claim1", "revision": revision}],
            commit_group_id="",
        ),
    )
    queue.enqueue(
        QueueJob(
            job_id=f"outbox_{job_id}",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u1/memories/canonical/slots/slot1",
            payload={
                "transaction_id": job_id,
                "outbox_path": str(outbox),
                "operation_ids": [f"op-{job_id}"],
            },
        )
    )


def _multiprocess_project(
    root: str,
    barrier: Any,
    vector_path: str,
    results: Any,
) -> None:
    artifact_root = Path(root) / "tenants" / "t1"
    source = FileSystemSourceStore(root, tenant_id="t1")
    index = SQLiteIndexStore(artifact_root / "indexes" / "projection-test.sqlite3")

    def hook(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_read":
            barrier.wait()

    projector = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=SQLiteTestVectorStore(vector_path),
        test_hook=hook,
    )
    result = projector.project(_claim(1).uri, 1)
    results.put((result.status, result.projection_attempt_id))


def test_projection_sidecar_retry_idempotency_stale_guard_and_rebuild(tmp_path) -> None:  # noqa: ANN001
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = CountingIndexStore()
    queue = InMemoryQueueStore()
    claim = _claim(1)
    obj = claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    _persist_committed_claim(
        source,
        artifact_root,
        obj,
        json.dumps({"claim": "SQLite", "revision": 1}),
    )
    canonical_before = source.read_object(claim.uri).to_dict()
    status_events: list[ProjectionRecord] = []
    projector = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=FailingVectorStore(),
        status_callback=status_events.append,
    )
    worker = MemoryProjectionWorker(projector, queue)
    _enqueue(queue, artifact_root, 1, "projection-1")

    first = worker.process_pending()
    assert first["failed"] == ["outbox_projection-1"]
    assert queue.jobs["outbox_projection-1"].status == "pending"
    failed_record = projector.record_store.load(claim.uri, 1)
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert failed_record.retryable is True
    assert "RuntimeError" in failed_record.failure_reason
    assert failed_record.index_status == "pending"
    assert failed_record.vector_status == "failed"
    assert status_events[-1].status == "failed"
    assert source.read_object(claim.uri).to_dict() == canonical_before

    vectors = InMemoryVectorStore()
    projector.vector_store = vectors
    second = worker.process_pending()
    assert second["processed"] == ["outbox_projection-1"]
    assert index.upsert_calls == 1, "retry must not repeat a completed projection component"
    projected_record = projector.record_store.load_current(claim.uri, source_revision=1)
    assert projected_record is not None
    assert projected_record.current is True
    assert projected_record.status == "completed"
    assert projected_record.retryable is False
    assert projected_record.failure_reason == ""
    assert status_events[-1].status == "completed"
    assert projected_record.attempt_count == 2
    assert {
        projected_record.index_status,
        projected_record.vector_status,
        projected_record.relation_status,
        projected_record.scope_status,
        projected_record.taxonomy_status,
    } == {"completed"}
    assert source.read_object(claim.uri).to_dict() == canonical_before
    assert source.read_object(claim.uri).layers.l0_uri is None
    assert "projection_revision" not in source.read_object(claim.uri).metadata

    manifest = json.loads(source.read_content(projected_record.manifest_uri))
    assert manifest["claim_uri"] == claim.uri
    assert manifest["slot_uri"].endswith("/slots/slot1")
    assert manifest["source_revision"] == 1
    assert manifest["projection_revision"] == 1
    assert manifest["status"] == "completed"
    assert manifest["current"] is False
    assert manifest["projection_levels"] == ["L0", "L1", "L2"]
    assert {item["projection_level"] for item in manifest["projections"]} == {"L0", "L1", "L2"}
    assert vectors.rows[claim.uri][1]["source_revision"] == 1
    assert claim.uri in index.indexed_uris()

    record_path = projector.record_store.record_path(claim.uri, 1)
    record_bytes = record_path.read_bytes()
    manifest_bytes = source.read_content(projected_record.manifest_uri)
    projector.project(claim.uri, 1)
    assert record_path.read_bytes() == record_bytes
    assert source.read_content(projected_record.manifest_uri) == manifest_bytes

    scope_current = next((artifact_root / "views" / "scope").glob("**/current.json"))
    taxonomy_current = next((artifact_root / "views" / "taxonomy").glob("**/current.json"))
    assert json.loads(scope_current.read_text(encoding="utf-8"))["source_revision"] == 1
    assert json.loads(taxonomy_current.read_text(encoding="utf-8"))["source_revision"] == 1

    updated_claim = _claim(2)
    updated_obj = updated_claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    _persist_committed_claim(
        source,
        artifact_root,
        updated_obj,
        json.dumps({"claim": "SQLite", "revision": 2}),
    )
    _enqueue(queue, artifact_root, 1, "stale-projection")
    stale = worker.process_pending()
    assert stale["stale"] == ["outbox_stale-projection"]
    old = projector.record_store.load(claim.uri, 1)
    assert old is not None and old.current is False and old.status == "stale"
    assert index.rows[claim.uri][0].metadata["projection_source_revision"] == 1

    _enqueue(queue, artifact_root, 2, "projection-2")
    worker.process_pending()
    current = projector.record_store.load_current(claim.uri, source_revision=2)
    assert current is not None
    new_scope_current = next((artifact_root / "views" / "scope").glob("**/current.json"))
    new_taxonomy_current = next((artifact_root / "views" / "taxonomy").glob("**/current.json"))
    assert json.loads(new_scope_current.read_text(encoding="utf-8"))["source_revision"] == 2
    assert json.loads(new_taxonomy_current.read_text(encoding="utf-8"))["source_revision"] == 2

    new_scope_current.unlink()
    rebuilt = projector.rebuild()
    assert rebuilt == {"projected": 1, "skipped": 0}
    assert next((artifact_root / "views" / "scope").glob("**/current.json")).exists()


def test_late_projection_cannot_overwrite_new_canonical_revision(tmp_path) -> None:  # noqa: ANN001
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    claim = _claim(1)
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "revision one",
    )
    projector_entered = threading.Event()
    resume_projector = threading.Event()

    def hook(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_read":
            projector_entered.set()
            assert resume_projector.wait(timeout=5)

    projector = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        test_hook=hook,
    )
    results = []
    errors = []

    def run_projection() -> None:
        try:
            results.append(projector.project(claim.uri, 1))
        except BaseException as exc:  # pragma: no cover - assertion reports the captured error.
            errors.append(exc)

    thread = threading.Thread(target=run_projection)
    thread.start()
    assert projector_entered.wait(timeout=5)
    revision_two = _claim(2).to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    _persist_committed_claim(source, artifact_root, revision_two, "revision two")
    canonical_revision_two = source.read_object(claim.uri).to_dict()
    resume_projector.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert [result.status for result in results] == ["skipped_stale"]
    assert source.read_object(claim.uri).to_dict() == canonical_revision_two
    assert source.read_content(claim.uri) == "revision two"
    assert claim.uri not in index.indexed_uris()
    assert claim.uri not in vectors.rows
    stale = ProjectionRecordStore(artifact_root).load(claim.uri, 1)
    assert stale is not None and stale.status == "stale" and stale.current is False


def test_successful_attempt_survives_later_duplicate_failure(tmp_path: Path) -> None:
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    claim = _claim(1)
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "revision one",
    )
    both_started = threading.Barrier(2)
    published = threading.Event()

    def successful_hook(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_read":
            both_started.wait(timeout=5)
        if stage == "after_publish":
            published.set()

    def failing_hook(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_read":
            both_started.wait(timeout=5)
        if stage == "after_artifacts":
            assert published.wait(timeout=5)
            raise RuntimeError("duplicate failed after peer publication")

    successful = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        test_hook=successful_hook,
    )
    failing = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        test_hook=failing_hook,
    )
    successes: list[Any] = []
    failures: list[BaseException] = []

    def run(projector: CanonicalMemoryProjector, output: list[Any], errors: list[BaseException]) -> None:
        try:
            output.append(projector.project(claim.uri, 1))
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    thread_a = threading.Thread(target=run, args=(successful, successes, failures))
    thread_b = threading.Thread(target=run, args=(failing, [], failures))
    thread_a.start()
    thread_b.start()
    thread_a.join(10)
    thread_b.join(10)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], RuntimeError)

    current = ProjectionRecordStore(artifact_root).load_current(claim.uri, source_revision=1)
    assert current is not None and current.projection_attempt_id == successes[0].projection_attempt_id
    attempts = ProjectionRecordStore(artifact_root).attempts(claim.uri, 1)
    assert len(attempts) == 2
    failed = next(item for item in attempts if item.projection_attempt_id != current.projection_attempt_id)
    assert failed.status == "failed" and failed.current is False
    assert index.rows[claim.uri][0].metadata["projection_attempt_id"] == current.projection_attempt_id
    assert vectors.rows[claim.uri][1]["projection_attempt_id"] == current.projection_attempt_id
    view_currents = list((artifact_root / "views").glob("**/current.json"))
    assert view_currents
    assert {
        json.loads(path.read_text(encoding="utf-8"))["projection_attempt_id"] for path in view_currents
    } == {current.projection_attempt_id}


def test_old_revision_failure_cannot_revoke_new_revision(tmp_path: Path) -> None:
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    claim = _claim(1)
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "revision one",
    )
    entered = threading.Event()
    resume = threading.Event()

    def old_hook(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_read":
            entered.set()
            assert resume.wait(timeout=5)
        if stage == "after_artifacts":
            raise RuntimeError("old attempt failed")

    old_projector = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        test_hook=old_hook,
    )
    errors: list[BaseException] = []

    def run_old() -> None:
        try:
            old_projector.project(claim.uri, 1)
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    thread = threading.Thread(target=run_old)
    thread.start()
    assert entered.wait(timeout=5)
    revision_two = _claim(2).to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    _persist_committed_claim(source, artifact_root, revision_two, "revision two")
    newer = CanonicalMemoryProjector(source, index, artifact_root, vector_store=vectors).project(claim.uri, 2)
    resume.set()
    thread.join(10)
    assert len(errors) == 1

    current = ProjectionRecordStore(artifact_root).load_current(claim.uri, source_revision=2)
    assert current is not None and current.projection_attempt_id == newer.projection_attempt_id
    assert index.rows[claim.uri][0].metadata["projection_attempt_id"] == current.projection_attempt_id
    assert vectors.rows[claim.uri][1]["projection_attempt_id"] == current.projection_attempt_id
    assert {
        json.loads(path.read_text(encoding="utf-8"))["source_revision"]
        for path in (artifact_root / "views").glob("**/current.json")
    } == {2}


def test_projection_crash_boundaries_and_same_revision_integrity_conflict(tmp_path: Path) -> None:
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    claim = _claim(1)
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "revision one",
    )

    def before_publish(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "before_publish":
            raise RuntimeError("crash before publish")

    with pytest.raises(RuntimeError, match="crash before publish"):
        CanonicalMemoryProjector(
            source,
            index,
            artifact_root,
            vector_store=vectors,
            test_hook=before_publish,
        ).project(claim.uri, 1)
    assert ProjectionRecordStore(artifact_root).load_current(claim.uri) is None
    assert claim.uri not in index.rows and claim.uri not in vectors.rows
    assert not list((artifact_root / "views").glob("**/current.json"))

    def after_pointer(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_publish":
            raise RuntimeError("crash after current pointer")

    with pytest.raises(RuntimeError, match="crash after current pointer"):
        CanonicalMemoryProjector(
            source,
            index,
            artifact_root,
            vector_store=vectors,
            test_hook=after_pointer,
        ).project(claim.uri, 1)
    current = ProjectionRecordStore(artifact_root).load_current(claim.uri, source_revision=1)
    assert current is not None and current.current and current.usable
    replay = CanonicalMemoryProjector(source, index, artifact_root, vector_store=vectors).project(claim.uri, 1)
    assert replay.projection_attempt_id == current.projection_attempt_id

    tampered = source.read_object(claim.uri)
    tampered.title = "same revision, different effect"
    source.write_object(tampered, content="tampered")
    with pytest.raises(ProjectionIntegrityError, match="different input effect"):
        CanonicalMemoryProjector(source, index, artifact_root, vector_store=vectors).project(claim.uri, 1)
    still_current = ProjectionRecordStore(artifact_root).load_current(claim.uri, source_revision=1)
    assert still_current is not None and still_current.projection_attempt_id == current.projection_attempt_id


@pytest.mark.parametrize("crash_stage", ["after_index", "before_view_publish", "after_view_publish"])
def test_projection_publish_stage_crash_recovers_one_consistent_attempt(
    tmp_path: Path,
    crash_stage: str,
) -> None:
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    claim = _claim(1)
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "revision one",
    )

    def crash(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == crash_stage:
            raise RuntimeError(f"injected {crash_stage} crash")

    with pytest.raises(RuntimeError, match=crash_stage):
        CanonicalMemoryProjector(
            source,
            index,
            artifact_root,
            vector_store=vectors,
            test_hook=crash,
        ).project(claim.uri, 1)

    record_store = ProjectionRecordStore(artifact_root)
    assert record_store.load_current(claim.uri) is None
    assert not list((artifact_root / "views").glob("**/current.json"))
    failed = record_store.attempts(claim.uri, 1)
    assert len(failed) == 1 and failed[0].status == "failed" and failed[0].current is False

    recovered = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
    ).project(claim.uri, 1)
    current = record_store.load_current(claim.uri, source_revision=1)
    assert current is not None and current.projection_attempt_id == recovered.projection_attempt_id
    assert index.rows[claim.uri][0].metadata["projection_attempt_id"] == current.projection_attempt_id
    assert vectors.rows[claim.uri][1]["projection_attempt_id"] == current.projection_attempt_id
    view_currents = list((artifact_root / "views").glob("**/current.json"))
    assert view_currents
    assert {
        json.loads(path.read_text(encoding="utf-8"))["projection_attempt_id"]
        for path in view_currents
    } == {current.projection_attempt_id}


def test_corrupt_projection_pointer_is_quarantined_once(tmp_path: Path) -> None:
    store = ProjectionRecordStore(tmp_path)
    claim_uri = _claim(1).uri
    pointer = store.current_path(claim_uri)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("{broken", encoding="utf-8")

    with pytest.raises(ProjectionIntegrityError, match="invalid projection state"):
        store.load_current(claim_uri)

    assert not pointer.exists()
    originals = list((tmp_path / "system" / "quarantine" / "projection_record").glob("*.original"))
    assert len(originals) == 1
    assert originals[0].read_text(encoding="utf-8") == "{broken"
    assert store.load_current(claim_uri) is None
    assert list((tmp_path / "system" / "quarantine" / "projection_record").glob("*.original")) == originals


def test_multiprocess_duplicate_projection_has_one_current_attempt(tmp_path: Path) -> None:
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    claim = _claim(1)
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "revision one",
    )
    index_path = artifact_root / "indexes" / "projection-test.sqlite3"
    vector_path = artifact_root / "indexes" / "projection-vectors.sqlite3"
    SQLiteIndexStore(index_path)
    SQLiteTestVectorStore(vector_path)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_multiprocess_project,
            args=(str(tmp_path), barrier, str(vector_path), results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in processes]
    current = ProjectionRecordStore(artifact_root).load_current(claim.uri, source_revision=1)
    assert current is not None
    assert {attempt_id for _, attempt_id in outcomes} == {current.projection_attempt_id}
    assert len(ProjectionRecordStore(artifact_root).attempts(claim.uri, 1)) == 2

    with sqlite3.connect(index_path) as conn:
        metadata = json.loads(
            conn.execute("SELECT metadata_json FROM contexts WHERE uri = ?", (claim.uri,)).fetchone()[0]
        )
    assert metadata["projection_attempt_id"] == current.projection_attempt_id
    assert metadata["projection_input_effect_hash"] == current.input_effect_hash
    with sqlite3.connect(vector_path) as conn:
        vector_metadata = json.loads(
            conn.execute("SELECT metadata_json FROM vectors WHERE uri = ?", (claim.uri,)).fetchone()[0]
        )
    assert vector_metadata["projection_attempt_id"] == current.projection_attempt_id
    views = [json.loads(path.read_text(encoding="utf-8")) for path in (artifact_root / "views").glob("**/current.json")]
    assert views and {item["projection_attempt_id"] for item in views} == {current.projection_attempt_id}
