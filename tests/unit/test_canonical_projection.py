from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from memoryos.contextdb.catalog import CatalogRecordKind
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
)
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, VectorStore, vector_row_id
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
from memoryos.memory.canonical.current_head import publish_current_head_sets
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.memory.canonical.visibility import CommittedStateIntegrityError
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.operations.commit.outbox_envelope import build_outbox, planned_effect_manifest
from memoryos.operations.commit.receipt import build_transaction_receipt
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

    def get_vector_metadata(self, uri: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT metadata_json FROM vectors WHERE uri = ?", (uri,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def vector_uris(self) -> list[str]:
        with sqlite3.connect(self.path) as conn:
            return [str(row[0]) for row in conn.execute("SELECT uri FROM vectors ORDER BY uri")]


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
    transaction_id = f"projection-{revision}"
    obj.metadata = {
        **obj.metadata,
        "canonical_idempotency_key": idempotency_key,
        "canonical_transaction_id": transaction_id,
    }
    source.write_object(obj, content=content)
    operation_id = f"projection-fixture-operation-{revision}"
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD if revision == 1 else OperationAction.UPDATE,
        target_uri=obj.uri,
        operation_id=operation_id,
        payload={
            "canonical_memory": True,
            "transaction_id": transaction_id,
            "idempotency_key": idempotency_key,
            "tenant_id": "t1",
            "commit_group_id": f"projection-fixture-{revision}",
            "planning_digest": canonical_digest(["projection-fixture", revision]),
            "expected_revision": max(0, revision - 1),
            "context_object": obj.to_dict(),
            "content": content,
        },
    )
    diff = {
        "user_id": "u1",
        "operations": [operation.to_dict()],
        "pending_operations": [],
        "rejected_operations": [],
        "diff_id": f"fixture-{revision}",
    }
    effect = planned_effect_manifest(operation, {})
    prepared = build_outbox(
        transaction_id=transaction_id,
        idempotency_key=idempotency_key,
        tenant_id="t1",
        user_id="u1",
        operations=[operation],
        status="prepared",
        before_images=[],
        effect_manifests=[effect],
        claim_revisions=[
            {
                "uri": obj.uri,
                "claim_id": str(obj.metadata["claim_id"]),
                "revision": revision,
            }
        ],
        commit_group_id=f"projection-fixture-{revision}",
    )
    receipt = build_transaction_receipt(
        transaction_id=transaction_id,
        idempotency_key=idempotency_key,
        tenant_id="t1",
        user_id="u1",
        commit_group_id=f"projection-fixture-{revision}",
        operations=[operation],
        diff=diff,
        planning_digest=str(operation.payload["planning_digest"]),
        prepared_intent_digest=str(prepared["prepared_intent_digest"]),
    )
    marker = artifact_root / "system" / "transactions" / f"{idempotency_key}.json"
    atomic_write_json(marker, receipt, artifact_root=artifact_root)
    publish_current_head_sets(artifact_root, marker, receipt)


def _enqueue(queue, tmp_path, revision: int, job_id: str) -> None:  # noqa: ANN001
    receipt = json.loads(
        (tmp_path / "system" / "transactions" / f"projection-fixture-revision-{revision}.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["transaction_id"] == job_id
    operation = ContextOperation.from_dict(dict(receipt["operations"][0]))
    uri = operation.target_uri
    effect = planned_effect_manifest(operation, {})
    outbox = tmp_path / "system" / "outbox" / f"{job_id}.json"
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
            commit_group_id=f"projection-fixture-{revision}",
            receipt_path=f"system/transactions/projection-fixture-revision-{revision}.json",
            receipt_digest=str(receipt["receipt_digest"]),
        ),
        artifact_root=tmp_path,
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
                "operation_ids": [operation.operation_id],
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
        vector_store=cast(VectorStore, SQLiteTestVectorStore(vector_path)),
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
    stale = projector.project(claim.uri, 1)
    assert stale.status == "skipped_stale"
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
    assert rebuilt == {"projected": 1, "skipped": 0, "retired": 0, "historical_restored": 0}
    assert next((artifact_root / "views" / "scope").glob("**/current.json")).exists()


def test_canonical_projection_sanitizes_layers_vector_index_and_view_metadata(tmp_path) -> None:  # noqa: ANN001
    class CapturingEmbedding:
        model_name = "capture-v1"
        dimension = 2

        def __init__(self) -> None:
            self.texts: list[str] = []

        def embed(self, text: str) -> list[float]:
            self.texts.append(text)
            return [1.0, 0.5]

    secret = "Authorization: Token canonical-secret read /Users/u1/Desktop/private-plan.md"
    claim = MemoryClaim(
        "claim-secret",
        "memoryos://user/u1/memories/canonical/slots/slot-secret/claims/claim-secret",
        "slot-secret",
        secret,
        TransitionProfile.AUTHORITATIVE_STATE,
        (
            MemoryRevision(
                revision=1,
                state="ACTIVE",
                value_fields={"canonical_value": secret},
                evidence_refs=(EvidenceRef("e-secret", None, "digest-secret"),),
                proposal_id="proposal-secret",
                relation="UNRELATED",
                epistemic_status="EXPLICIT",
                qualifiers={"display_fields": {"details": secret}},
            ),
        ),
    )
    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    embedding = CapturingEmbedding()
    obj = claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope=_scope(),
    )
    obj.metadata["identity_fields"] = {
        "decision_topic": "Authorization: Token tree-path-secret",
    }
    _persist_committed_claim(source, artifact_root, obj, secret)

    result = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        embedding_provider=embedding,
    ).project(claim.uri, 1)

    record = ProjectionRecordStore(artifact_root).load_current(claim.uri, source_revision=1)
    assert result.status == "projected" and record is not None
    derived = "\n".join(
        (
            source.read_content(record.l0_uri),
            source.read_content(record.l1_uri),
            source.read_content(record.l2_uri),
            repr(index.rows[claim.uri]),
            repr(vectors.get_vector_metadata(claim.uri)),
            "\n".join(embedding.texts),
            "\n".join(path.read_text(encoding="utf-8") for path in (artifact_root / "views").glob("**/*.json")),
        )
    )
    assert "canonical-secret" not in derived
    assert "/Users/u1" not in derived
    assert "tree-path-secret" not in derived
    vector_metadata = vectors.get_vector_metadata(claim.uri)
    assert vector_metadata is not None
    assert {
        "catalog_record_key",
        "tenant_id",
        "owner_user_id",
        "workspace_id",
        "session_id",
        "adapter_id",
        "context_type",
        "source_kind",
        "record_kind",
        "lifecycle_state",
        "primary_tree_path",
        "tree_paths",
        "scope_keys",
        "created_at",
        "updated_at",
        "event_time",
        "ingested_at",
        "transaction_time",
        "valid_from",
        "valid_to",
        "source_uri",
        "source_digest",
        "source_revision",
        "canonical_slot_id",
        "canonical_claim_id",
        "canonical_revision",
        "canonical_state",
        "canonical_head_digest",
        "receipt_digest",
        "projection_effect_hash",
        "serving_tier",
        "projection_status",
    } <= vector_metadata.keys()
    assert vector_metadata["catalog_record_key"] == "claim:claim-secret:revision:1"
    assert vector_metadata["workspace_id"] == "memoryos"
    assert tuple(vector_metadata["scope_keys"]) == ("memoryos:workspace:memoryos",)
    assert vector_metadata["record_kind"] == "claim_revision"
    assert vector_metadata["source_revision"] == 1
    assert vector_metadata["projection_effect_hash"] == result.input_effect_hash
    assert vector_metadata["canonical_head_digest"]
    assert vector_metadata["receipt_digest"]
    assert vector_metadata["tree_paths"][0].startswith("memories/decisions/")
    assert "projects/memoryos" in vector_metadata["tree_paths"]


def test_claim_projection_derives_previous_valid_to_without_rewriting_source(tmp_path) -> None:  # noqa: ANN001
    class CountingEmbedding:
        model_name = "counting-validity-v1"
        dimension = 2

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:
            del text
            self.calls += 1
            return [1.0, 0.5]

    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = SQLiteIndexStore(artifact_root / "indexes" / "validity.sqlite3")
    vectors = InMemoryVectorStore()
    embedding = CountingEmbedding()
    revision_one = MemoryRevision(
        revision=1,
        state="ACTIVE",
        value_fields={"canonical_value": "SQLite"},
        evidence_refs=(EvidenceRef("e1", None, "hash-1"),),
        proposal_id="p1",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        valid_from="2026-07-14T00:00:00+00:00",
    )
    claim_one = MemoryClaim(
        "claim-validity",
        "memoryos://user/u1/memories/canonical/slots/slot-validity/claims/claim-validity",
        "slot-validity",
        "SQLite",
        TransitionProfile.AUTHORITATIVE_STATE,
        (revision_one,),
    )
    projector = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        embedding_provider=embedding,
    )
    _persist_committed_claim(
        source,
        artifact_root,
        claim_one.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "SQLite",
    )
    assert projector.project(claim_one.uri, 1).status == "projected"
    published_revision_one = projector.record_store.load_current(claim_one.uri, source_revision=1)
    assert published_revision_one is not None
    assert embedding.calls == 1

    revision_two = MemoryRevision(
        revision=2,
        state="SUPERSEDED",
        value_fields={"canonical_value": "SQLite"},
        evidence_refs=(EvidenceRef("e2", None, "hash-2"),),
        proposal_id="p2",
        relation="CORRECTS",
        epistemic_status="EXPLICIT",
        previous_revision=1,
        valid_from="2026-07-15T00:00:00+00:00",
    )
    claim_two = claim_one.with_revision(revision_two)
    _persist_committed_claim(
        source,
        artifact_root,
        claim_two.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_decision",
            scope=_scope(),
        ),
        "SQLite",
    )
    assert projector.project(claim_two.uri, 2).status == "projected"

    # A later failed retry is deliberately newer than the published revision
    # 1 attempt.  Refresh must follow the Catalog's exact attempt id, never the
    # record store's arbitrary preferred/latest attempt for a non-current
    # revision.
    failed_attempt_id = "f" * 32
    failed_base = f"{claim_one.uri}/projections/rev-1/attempt-{failed_attempt_id}"
    failed_attempt = projector.record_store.start(
        claim_uri=claim_one.uri,
        slot_uri=claim_one.uri.rsplit("/claims/", 1)[0],
        source_revision=1,
        projection_revision=1,
        projection_attempt_id=failed_attempt_id,
        input_effect_hash=published_revision_one.input_effect_hash,
        l0_uri=f"{failed_base}/l0.md",
        l1_uri=f"{failed_base}/l1.md",
        l2_uri=f"{failed_base}/l2.json",
        relations_uri=f"{failed_base}/relations.json",
        manifest_uri=f"{failed_base}/manifest.json",
        current_claim_revision=1,
    )
    projector.record_store.fail(failed_attempt, "deliberate later failed retry")
    assert projector.record_store.load(claim_one.uri, 1).projection_attempt_id == failed_attempt_id  # type: ignore[union-attr]
    current_obj = source.read_object(claim_two.uri)
    projector._reconcile_claim_catalog_projections(  # noqa: SLF001
        current_obj,
        dict(current_obj.metadata or {}),
        published_revision=2,
    )
    # Revision 2 publishes once; both the automatic and explicit repair paths
    # refresh revision 1 exactly once.
    assert embedding.calls == 4

    old_record = index.get_catalog("claim:claim-validity:revision:1", tenant_id="t1")
    assert old_record is not None
    assert old_record.valid_to == revision_two.valid_from
    assert old_record.metadata["valid_to"] == revision_two.valid_from
    assert old_record.metadata["validity_end_derived"] is True
    assert old_record.metadata["projection_attempt_id"] == published_revision_one.projection_attempt_id
    old_vector = vectors.get_vector_metadata(vector_row_id("t1", old_record.record_key))
    assert old_vector is not None
    assert old_vector["valid_to"] == revision_two.valid_from

    committed = source.read_object(claim_two.uri)
    source_revisions = list(committed.metadata["revisions"])
    assert source_revisions[0]["valid_to"] is None


def test_late_historical_claim_catalog_binds_requested_revision_without_changing_legacy_current(
    tmp_path,
) -> None:  # noqa: ANN001
    class CapturingEmbedding:
        model_name = "capturing-late-history-v1"
        dimension = 2

        def __init__(self) -> None:
            self.inputs: list[str] = []

        def embed(self, text: str) -> list[float]:
            self.inputs.append(text)
            return [1.0, 0.5]

    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = SQLiteIndexStore(artifact_root / "indexes" / "late-history.sqlite3")
    vectors = InMemoryVectorStore()
    embedding = CapturingEmbedding()
    effective_current = MemoryRevision(
        revision=1,
        state="ACTIVE",
        value_fields={"canonical_value": "current effective rule"},
        evidence_refs=(EvidenceRef("current", None, "current-hash"),),
        proposal_id="current",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        qualifiers={"display_fields": {"summary": "current display"}},
        created_at="2026-07-14T00:10:00+00:00",
        transaction_time="2026-07-14T00:10:00+00:00",
        valid_from="2026-07-14T00:00:00+00:00",
    )
    late_history = MemoryRevision(
        revision=2,
        state="PROPOSED",
        value_fields={"canonical_value": "older historical rule"},
        evidence_refs=(EvidenceRef("history", None, "history-hash"),),
        proposal_id="history",
        relation="SUPPLEMENTS",
        epistemic_status="EXPLICIT",
        qualifiers={
            "non_current_historical": True,
            "display_fields": {"summary": "historical display"},
        },
        created_at="2026-07-15T08:30:00+00:00",
        transaction_time="2026-07-15T08:30:00+00:00",
        previous_revision=1,
        valid_from="2026-06-01T12:00:00+00:00",
    )
    claim = MemoryClaim(
        "claim-late-history",
        "memoryos://user/u1/memories/canonical/slots/slot-late-history/claims/claim-late-history",
        "slot-late-history",
        "current effective rule",
        TransitionProfile.AUTHORITATIVE_STATE,
        (effective_current, late_history),
    )
    projector = CanonicalMemoryProjector(
        source,
        index,
        artifact_root,
        vector_store=vectors,
        embedding_provider=embedding,
    )
    claim_one = MemoryClaim(
        claim.claim_id,
        claim.uri,
        claim.slot_id,
        claim.canonical_value,
        claim.profile,
        (effective_current,),
    )
    _persist_committed_claim(
        source,
        artifact_root,
        claim_one.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_rule",
            scope=_scope(),
        ),
        "current effective rule",
    )
    assert projector.project(claim.uri, 1).status == "projected"
    _persist_committed_claim(
        source,
        artifact_root,
        claim.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_rule",
            scope=_scope(),
        ),
        "current effective rule",
    )

    assert projector.project(claim.uri, 2).status == "projected"

    row = index.get_catalog("claim:claim-late-history:revision:2", tenant_id="t1")
    assert row is not None
    assert row.canonical_revision == 2
    assert row.canonical_state == "PROPOSED"
    assert row.metadata["revision"] == 2
    assert row.metadata["value_fields"] == {"canonical_value": "older historical rule"}
    assert [item["revision"] for item in row.metadata["revisions"]] == [2]
    assert row.metadata["state"] == "PROPOSED"
    assert row.metadata["canonical_value"] == "older historical rule"
    assert row.event_time == late_history.valid_from
    assert row.transaction_time == late_history.transaction_time
    assert "older historical rule" in row.l0_text
    assert "historical display" in row.l1_text
    assert "current effective rule" not in row.l1_text
    assert "older historical rule" in source.read_content(row.l2_uri)
    vector_metadata = vectors.get_vector_metadata(vector_row_id("t1", row.record_key))
    assert vector_metadata is not None
    assert vector_metadata["canonical_revision"] == 2
    assert vector_metadata["canonical_state"] == "PROPOSED"
    assert vector_metadata["event_time"] == late_history.valid_from
    assert vector_metadata["transaction_time"] == late_history.transaction_time
    assert any("older historical rule" in value for value in embedding.inputs)

    proof = projector.record_store.load_current(claim.uri, source_revision=2)
    assert proof is not None
    assert proof.current_claim_revision == 1
    assert "current effective rule" in source.read_content(proof.l0_uri)
    assert "older historical rule" not in source.read_content(proof.l0_uri)


def test_claim_revision_refresh_rebinds_owner_scope_paths_acl_fts_and_vector(tmp_path) -> None:  # noqa: ANN001
    def private_scope(
        visible_principal: str,
        workspace_id: str,
        *,
        tenant_id: str = "t1",
    ) -> dict[str, Any]:
        workspace: dict[str, object] = {
            "namespace": "memoryos",
            "kind": "workspace",
            "id": workspace_id,
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
                "tenant_id": tenant_id,
                "allowed_principal_ids": [visible_principal],
                "allowed_service_ids": [],
                "private": True,
            },
            "authority": {
                "principal_ids": ["u1"],
                "service_ids": [],
                "inferred": False,
            },
            "origin_refs": [],
        }

    artifact_root = _artifact_root(tmp_path)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = SQLiteIndexStore(artifact_root / "indexes" / "security-refresh.sqlite3")
    vectors = InMemoryVectorStore()
    projector = CanonicalMemoryProjector(source, index, artifact_root, vector_store=vectors)
    revision_one = MemoryRevision(
        revision=1,
        state="ACTIVE",
        value_fields={"canonical_value": "revision one searchable marker"},
        evidence_refs=(EvidenceRef("r1", None, "r1-hash"),),
        proposal_id="r1",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        created_at="2026-07-10T01:00:00+00:00",
        transaction_time="2026-07-10T01:00:00+00:00",
        valid_from="2026-07-10T00:00:00+00:00",
    )
    claim_one = MemoryClaim(
        "claim-security-refresh",
        "memoryos://user/u1/memories/canonical/slots/slot-security-refresh/claims/claim-security-refresh",
        "slot-security-refresh",
        "revision one searchable marker",
        TransitionProfile.AUTHORITATIVE_STATE,
        (revision_one,),
    )
    _persist_committed_claim(
        source,
        artifact_root,
        claim_one.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_rule",
            scope=private_scope("old-reader", "old-workspace"),
        ),
        "revision one searchable marker",
    )
    assert projector.project(claim_one.uri, 1).status == "projected"
    revision_one_key = "claim:claim-security-refresh:revision:1"
    stale_owner_row = index.get_catalog(revision_one_key, tenant_id="t1")
    assert stale_owner_row is not None
    stale_scope = private_scope("old-reader", "old-workspace", tenant_id="legacy-tenant")
    assert index.delete_catalog(revision_one_key, tenant_id="t1") is True
    index.upsert_catalog(
        replace(
            stale_owner_row,
            tenant_id="legacy-tenant",
            owner_user_id="stale-owner",
            metadata={
                **dict(stale_owner_row.metadata),
                "tenant_id": "legacy-tenant",
                "owner_user_id": "stale-owner",
                "scope": stale_scope,
            },
        )
    )
    old_vector = vectors.get_vector_metadata(vector_row_id("t1", revision_one_key))
    assert old_vector is not None
    vectors.delete_vector(vector_row_id("t1", revision_one_key))
    vectors.upsert_vector(
        vector_row_id("legacy-tenant", revision_one_key),
        [1.0, 0.5],
        metadata={
            **old_vector,
            "tenant_id": "legacy-tenant",
            "owner_user_id": "stale-owner",
            "workspace_id": "old-workspace",
        },
    )

    revision_two = MemoryRevision(
        revision=2,
        state="ACTIVE",
        value_fields={"canonical_value": "revision two searchable marker"},
        evidence_refs=(EvidenceRef("r2", None, "r2-hash"),),
        proposal_id="r2",
        relation="CORRECTS",
        epistemic_status="EXPLICIT",
        created_at="2026-07-15T02:00:00+00:00",
        transaction_time="2026-07-15T02:00:00+00:00",
        previous_revision=1,
        valid_from="2026-07-15T00:00:00+00:00",
    )
    claim_two = claim_one.with_revision(revision_two)
    _persist_committed_claim(
        source,
        artifact_root,
        claim_two.to_context_object(
            tenant_id="t1",
            owner_user_id="u1",
            memory_type="project_rule",
            scope=private_scope("new-reader", "new-workspace"),
        ),
        "revision two searchable marker",
    )
    assert projector.project(claim_two.uri, 2).status == "projected"

    old_key = revision_one_key
    refreshed = index.get_catalog(old_key, tenant_id="t1")
    assert refreshed is not None
    assert index.get_catalog(old_key, tenant_id="legacy-tenant") is None
    assert refreshed.owner_user_id == "u1"
    assert refreshed.workspace_id == "new-workspace"
    assert refreshed.primary_tree_path.startswith("memories/rules/")
    assert "projects/new-workspace" in refreshed.tree_paths
    assert "projects/old-workspace" not in refreshed.tree_paths
    assert "revision one searchable marker" in refreshed.l1_text
    assert "revision two searchable marker" not in refreshed.l1_text
    assert refreshed.transaction_time == revision_one.transaction_time
    assert refreshed.valid_to == revision_two.valid_from
    assert refreshed.metadata["scope"] == private_scope("new-reader", "new-workspace")

    old_candidates = index.search_catalog(
        "revision one searchable marker",
        filters={
            "tenant_id": "t1",
            "principal_owner_id": "old-reader",
            "workspace_access_ids": ("old-workspace",),
            "applicability_scope_keys": ("memoryos:workspace:old-workspace",),
            "target_paths": ("projects/old-workspace",),
            "record_kinds": (CatalogRecordKind.CLAIM_REVISION.value,),
            "include_inactive": True,
        },
        limit=10,
    )
    assert old_candidates == []
    new_candidates = index.search_catalog(
        "revision one searchable marker",
        filters={
            "tenant_id": "t1",
            "principal_owner_id": "new-reader",
            "workspace_access_ids": ("new-workspace",),
            "applicability_scope_keys": ("memoryos:workspace:new-workspace",),
            "target_paths": ("projects/new-workspace",),
            "record_kinds": (CatalogRecordKind.CLAIM_REVISION.value,),
            "include_inactive": True,
        },
        limit=10,
    )
    assert old_key in {item.metadata["catalog_record_key"] for item in new_candidates}
    vector_metadata = vectors.get_vector_metadata(vector_row_id("t1", old_key))
    assert vector_metadata is not None
    assert vectors.get_vector_metadata(vector_row_id("legacy-tenant", old_key)) is None
    assert vector_metadata["owner_user_id"] == "u1"
    assert vector_metadata["workspace_id"] == "new-workspace"
    assert "projects/new-workspace" in vector_metadata["tree_paths"]
    assert "projects/old-workspace" not in vector_metadata["tree_paths"]


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
    assert {json.loads(path.read_text(encoding="utf-8"))["projection_attempt_id"] for path in view_currents} == {
        current.projection_attempt_id
    }


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
    with pytest.raises(CommittedStateIntegrityError, match="without an in-flight redo proof"):
        CanonicalMemoryProjector(
            source,
            index,
            artifact_root,
            vector_store=vectors,
        ).project(claim.uri, 1)
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
    assert {json.loads(path.read_text(encoding="utf-8"))["projection_attempt_id"] for path in view_currents} == {
        current.projection_attempt_id
    }


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
            conn.execute(
                "SELECT metadata_json FROM vectors WHERE uri = ?",
                (vector_row_id("t1", "claim:claim1:revision:1"),),
            ).fetchone()[0]
        )
    assert vector_metadata["projection_attempt_id"] == current.projection_attempt_id
    views = [json.loads(path.read_text(encoding="utf-8")) for path in (artifact_root / "views").glob("**/current.json")]
    assert views and {item["projection_attempt_id"] for item in views} == {current.projection_attempt_id}
