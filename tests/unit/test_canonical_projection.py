from __future__ import annotations

import json

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
    TransitionProfile,
)


class FailingVectorStore(InMemoryVectorStore):
    def upsert_vector(self, uri, embedding, metadata=None):  # noqa: ANN001, ANN201
        raise RuntimeError("vector unavailable")


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
            job_id=job_id,
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u1/memories/canonical/slots/slot1",
            payload={"outbox_path": str(outbox)},
        )
    )


def test_projection_revision_binding_retry_idempotency_stale_guard_and_rebuild(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    queue = InMemoryQueueStore()
    claim = _claim(1)
    obj = claim.to_context_object(tenant_id="t1", owner_user_id="u1", memory_type="project_decision", scope=_scope())
    source.write_object(obj, content=json.dumps({"claim": "SQLite", "revision": 1}))
    projector = CanonicalMemoryProjector(source, index, tmp_path, vector_store=FailingVectorStore())
    worker = MemoryProjectionWorker(projector, queue)
    _enqueue(queue, tmp_path, 1, "projection-1")

    first = worker.process_pending()
    assert first["failed"] == ["projection-1"]
    assert queue.jobs["projection-1"].status == "pending"
    vectors = InMemoryVectorStore()
    projector.vector_store = vectors
    second = worker.process_pending()
    assert second["processed"] == ["projection-1"]
    projected = source.read_object(claim.uri)
    assert projected.metadata["projection_revision"] == 1
    assert projected.layers.l0_uri is not None
    assert "/projections/rev-1/" in projected.layers.l0_uri
    manifest = json.loads(source.read_content(projected.metadata["projection_manifest_uri"]))
    assert manifest["source_revision"] == 1
    assert manifest["slot_id"] == "slot1"
    assert manifest["projection_levels"] == ["L0", "L1", "L2"]
    assert {item["projection_level"] for item in manifest["projections"]} == {"L0", "L1", "L2"}
    assert {item["slot_id"] for item in manifest["projections"]} == {"slot1"}
    assert all(item["source_revision"] == 1 and item["created_at"] for item in manifest["projections"])
    assert vectors.rows[claim.uri][1]["source_revision"] == 1
    assert claim.uri in index.indexed_uris()
    scope_view = next((tmp_path / "views" / "scope").glob("**/claim1.json"))
    taxonomy_view = next((tmp_path / "views" / "taxonomy").glob("**/claim1.json"))

    updated_claim = _claim(2)
    updated_obj = updated_claim.to_context_object(
        tenant_id="t1", owner_user_id="u1", memory_type="project_decision", scope=_scope()
    )
    source.write_object(updated_obj, content=json.dumps({"claim": "SQLite", "revision": 2}))
    _enqueue(queue, tmp_path, 1, "stale-projection")
    stale = worker.process_pending()
    assert stale["stale"] == ["stale-projection"]
    assert json.loads(scope_view.read_text(encoding="utf-8"))["source_revision"] == 1

    _enqueue(queue, tmp_path, 2, "projection-2")
    worker.process_pending()
    assert json.loads(scope_view.read_text(encoding="utf-8"))["source_revision"] == 2
    assert json.loads(taxonomy_view.read_text(encoding="utf-8"))["source_revision"] == 2
    scope_view.unlink()
    rebuilt = projector.rebuild()
    assert rebuilt["projected"] == 1 and scope_view.exists()
