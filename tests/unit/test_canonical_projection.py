from __future__ import annotations

import json
import threading

from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
)
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.canonical import (
    CanonicalMemoryProjector,
    EvidenceRef,
    MemoryClaim,
    MemoryProjectionWorker,
    MemoryRevision,
    ProjectionRecord,
    TransitionProfile,
)
from memoryos.memory.canonical.projection_state import ProjectionRecordStore


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
    return {
        "applicability": {
            "all_of": [
                {
                    "namespace": "memoryos",
                    "kind": "workspace",
                    "id": "memoryos",
                    "parent_id": None,
                    "attributes": {},
                }
            ]
        },
        "visibility": {
            "tenant_id": "t1",
            "allowed_principal_ids": [],
            "allowed_service_ids": [],
            "private": False,
        },
        "origin_refs": [],
    }


def _enqueue(queue, tmp_path, revision: int, job_id: str) -> None:  # noqa: ANN001
    outbox = tmp_path / "system" / "outbox" / f"{job_id}.json"
    outbox.parent.mkdir(parents=True, exist_ok=True)
    outbox.write_text(
        json.dumps(
            {
                "event_type": "MemoryCommitted",
                "status": "committed",
                "transaction_id": job_id,
                "operation_ids": [f"op-{job_id}"],
                "claim_revisions": [
                    {
                        "uri": "memoryos://user/u1/memories/canonical/slots/slot1/claims/claim1",
                        "claim_id": "claim1",
                        "revision": revision,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    queue.enqueue(
        QueueJob(
            job_id=f"outbox_{job_id}",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u1/memories/canonical/slots/slot1",
            payload={"outbox_path": str(outbox)},
        )
    )


def test_projection_sidecar_retry_idempotency_stale_guard_and_rebuild(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = CountingIndexStore()
    queue = InMemoryQueueStore()
    claim = _claim(1)
    obj = claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    source.write_object(obj, content=json.dumps({"claim": "SQLite", "revision": 1}))
    canonical_before = source.read_object(claim.uri).to_dict()
    status_events: list[ProjectionRecord] = []
    projector = CanonicalMemoryProjector(
        source,
        index,
        tmp_path,
        vector_store=FailingVectorStore(),
        status_callback=status_events.append,
    )
    worker = MemoryProjectionWorker(projector, queue)
    _enqueue(queue, tmp_path, 1, "projection-1")

    first = worker.process_pending()
    assert first["failed"] == ["outbox_projection-1"]
    assert queue.jobs["outbox_projection-1"].status == "pending"
    failed_record = projector.record_store.load(claim.uri, 1)
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert failed_record.retryable is True
    assert "RuntimeError" in failed_record.failure_reason
    assert failed_record.index_status == "completed"
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

    scope_current = next((tmp_path / "views" / "scope").glob("**/current.json"))
    taxonomy_current = next((tmp_path / "views" / "taxonomy").glob("**/current.json"))
    assert json.loads(scope_current.read_text(encoding="utf-8"))["source_revision"] == 1
    assert json.loads(taxonomy_current.read_text(encoding="utf-8"))["source_revision"] == 1

    updated_claim = _claim(2)
    updated_obj = updated_claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    source.write_object(updated_obj, content=json.dumps({"claim": "SQLite", "revision": 2}))
    _enqueue(queue, tmp_path, 1, "stale-projection")
    stale = worker.process_pending()
    assert stale["stale"] == ["outbox_stale-projection"]
    old = projector.record_store.load(claim.uri, 1)
    assert old is not None and old.current is False and old.status == "stale"
    assert claim.uri not in index.indexed_uris()

    _enqueue(queue, tmp_path, 2, "projection-2")
    worker.process_pending()
    current = projector.record_store.load_current(claim.uri, source_revision=2)
    assert current is not None
    new_scope_current = next((tmp_path / "views" / "scope").glob("**/current.json"))
    new_taxonomy_current = next((tmp_path / "views" / "taxonomy").glob("**/current.json"))
    assert json.loads(new_scope_current.read_text(encoding="utf-8"))["source_revision"] == 2
    assert json.loads(new_taxonomy_current.read_text(encoding="utf-8"))["source_revision"] == 2

    new_scope_current.unlink()
    rebuilt = projector.rebuild()
    assert rebuilt == {"projected": 1, "skipped": 0}
    assert next((tmp_path / "views" / "scope").glob("**/current.json")).exists()


def test_late_projection_cannot_overwrite_new_canonical_revision(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    claim = _claim(1)
    source.write_object(
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        content="revision one",
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
        tmp_path,
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
    source.write_object(revision_two, content="revision two")
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
    stale = ProjectionRecordStore(tmp_path).load(claim.uri, 1)
    assert stale is not None and stale.status == "stale" and stale.current is False
