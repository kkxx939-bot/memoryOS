from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from infrastructure.store.contracts.vector import vector_row_id
from infrastructure.store.filesystem.memory_document_store import (
    FileSystemMemoryDocumentStore,
)
from infrastructure.store.memory import (
    DocumentControlRecord,
    DocumentDeletionStatus,
    DocumentPublicationBarrier,
    MemoryDocumentControlStore,
    MemoryDocumentRevisionStore,
)
from infrastructure.store.memory.erasure_store import MemoryDocumentEraseStore
from infrastructure.store.model.catalog import CatalogRecordKind
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from memory.commit import DocumentEraseStatus, MemoryDocumentEraser
from memory.core import (
    ABSENT,
    PresentPath,
    new_document_id,
    render_new_document,
)
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.worker.projection.erase_backend import MemoryDocumentCatalogEraseBackend
from memory.worker.projection.worker import MemoryDocumentProjectionWorker
from tests.support.embedding import DeterministicEmbeddingProvider
from tests.support.persistence import InMemoryVectorStore
from tests.support.persistence.in_memory import InMemoryQueueStore, InMemoryRelationStore


def _components(root: Path):  # noqa: ANN202
    documents = FileSystemMemoryDocumentStore(root)
    controls = MemoryDocumentControlStore(root)
    catalog = SQLiteIndexStore(root / "catalog.sqlite3")
    worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        catalog,
        InMemoryQueueStore(),
    )
    return documents, catalog, worker


def _rich_components(root: Path):  # noqa: ANN202
    documents = FileSystemMemoryDocumentStore(root)
    controls = MemoryDocumentControlStore(root)
    catalog = SQLiteIndexStore(root / "catalog.sqlite3")
    relations = InMemoryRelationStore()
    vectors = InMemoryVectorStore()
    worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        catalog,
        InMemoryQueueStore(),
        vector_store=vectors,
        embedding_provider=DeterministicEmbeddingProvider(),
        relation_store=relations,
    )
    return documents, controls, catalog, relations, vectors, worker


def _write_soft_barrier(
    controls: MemoryDocumentControlStore,
    document_id: str,
    relative_path: str,
    *,
    generation: int = 2,
) -> DocumentPublicationBarrier:
    return controls.write_publication_barrier(
        DocumentPublicationBarrier(
            tenant_id="t1",
            owner_user_id="u1",
            document_id=document_id,
            relative_path=relative_path,
            deletion_generation=generation,
            deletion_event_digest="a" * 64,
            status=DocumentDeletionStatus.SOFT_FORGOTTEN,
            updated_at="2026-07-18T00:00:00+00:00",
        )
    )


def _document_records(catalog: SQLiteIndexStore, document_id: str, *, include_inactive: bool = False):  # noqa: ANN202
    return catalog.scan_catalog_batch(
        tenant_id="t1",
        filters={
            "owner_user_id": "u1",
            "document_id": document_id,
            "record_kind": CatalogRecordKind.MEMORY_DOCUMENT.value,
            "include_inactive": include_inactive,
        },
        limit=100,
    )


def test_full_scan_rebuild_restores_document_catalog_from_exact_live_markdown(tmp_path: Path) -> None:
    documents, catalog, worker = _components(tmp_path)
    document_id = new_document_id()
    raw = render_new_document(document_id, "# Durable preference\n\nUse concise answers.\n")
    created = documents.create("t1", "u1", "preferences.md", raw, expected=ABSENT)

    first = worker.rebuild_owner("t1", "u1")
    assert first == {"projected": 1, "skipped": 0, "deleted": 0, "documents": 1}
    before = _document_records(catalog, document_id)
    assert len(before) == 1
    assert before[0].source_digest == created.raw_sha256
    assert before[0].l2_uri == created.uri

    catalog.clear(tenant_id="t1")
    assert _document_records(catalog, document_id) == []

    rebuilt = worker.rebuild_owner("t1", "u1")
    assert rebuilt["projected"] == 1
    after = _document_records(catalog, document_id)
    assert len(after) == 1
    assert after[0].source_digest == created.raw_sha256
    assert worker.verify_owner("t1", "u1") == {"verified": 1, "projected": 1}


def test_external_delete_barrier_prevents_rebuild_resurrection(tmp_path: Path) -> None:
    documents, catalog, worker = _components(tmp_path)
    document_id = new_document_id()
    relative_path = "knowledge/topics/deleted.md"
    raw = render_new_document(document_id, "# Deleted\n\nDo not resurrect this text.\n")
    documents.create("t1", "u1", relative_path, raw, expected=ABSENT)
    worker.rebuild_owner("t1", "u1")

    state = documents.read_state("t1", "u1", relative_path)
    assert isinstance(state, PresentPath)
    documents.delete("t1", "u1", document_id, expected_state=state)
    _write_soft_barrier(worker.control_store, document_id, relative_path)
    deleted = worker.rebuild_owner("t1", "u1")
    assert deleted["deleted"] == 1
    projection_state = catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
    )
    assert projection_state is not None
    assert projection_state["deletion_status"] == "SOFT_FORGOTTEN"
    assert _document_records(catalog, document_id) == []

    documents.create("t1", "u1", relative_path, raw, expected=ABSENT)
    replay = worker.rebuild_owner("t1", "u1")
    assert replay["projected"] == 0
    assert replay["skipped"] == 1
    assert _document_records(catalog, document_id) == []

    barrier = worker.control_store.load_publication_barrier("t1", "u1", document_id)
    assert barrier is not None and barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
    for serving_file in tmp_path.glob("catalog.sqlite3*"):
        serving_file.unlink()
    recreated_catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    recreated_worker = MemoryDocumentProjectionWorker(
        documents,
        worker.control_store,
        recreated_catalog,
        InMemoryQueueStore(),
    )

    after_sqlite_loss = recreated_worker.rebuild_owner("t1", "u1")

    assert after_sqlite_loss["projected"] == 0
    assert after_sqlite_loss["skipped"] == 1
    recreated_state = recreated_catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
    )
    assert recreated_state is not None
    assert recreated_state["deletion_generation"] == barrier.deletion_generation
    assert recreated_state["deletion_event_digest"] == barrier.deletion_event_digest
    assert recreated_state["deletion_status"] == barrier.status.value
    assert _document_records(recreated_catalog, document_id) == []


def test_external_target_delete_rebuild_removes_exact_inbound_document_links(tmp_path: Path) -> None:
    documents, controls, _catalog, relations, _vectors, worker = _rich_components(tmp_path)
    source_id = new_document_id()
    target_id = new_document_id()
    source_path = "knowledge/topics/source.md"
    target_path = "knowledge/topics/target.md"
    documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nDelete externally.\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    target_uri = MemoryDocumentPathPolicy.document_uri("u1", target_id)
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1")

    documents.delete(
        "t1",
        "u1",
        target_id,
        expected_state=documents.read_state("t1", "u1", target_path),
    )
    _write_soft_barrier(controls, target_id, target_path)
    rebuilt = worker.rebuild_owner("t1", "u1")

    assert rebuilt["deleted"] == 1
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []
    barrier = controls.load_publication_barrier("t1", "u1", target_id)
    assert barrier is not None and barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
    worker.rebuild_owner("t1", "u1")
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []


def test_hard_erased_identity_stays_blocked_after_sqlite_recreation(tmp_path: Path) -> None:
    documents, catalog, worker = _components(tmp_path)
    controls = worker.control_store
    document_id = new_document_id()
    relative_path = "knowledge/topics/hard-erased.md"
    raw = render_new_document(document_id, "# Erased\n\nNever republish these old bytes.\n")
    created = documents.create("t1", "u1", relative_path, raw, expected=ABSENT)
    worker.rebuild_owner("t1", "u1")
    eraser = MemoryDocumentEraser(
        documents,
        controls,
        MemoryDocumentRevisionStore(tmp_path),
        cleanup_backends=(MemoryDocumentCatalogEraseBackend(worker),),
        erase_store=MemoryDocumentEraseStore(controls.root),
    )

    erased = eraser.hard_erase(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
        expected_source_digest=created.raw_sha256,
        relative_path=relative_path,
    )

    assert erased.completed is True
    barrier = controls.load_publication_barrier("t1", "u1", document_id)
    assert barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED
    documents.create("t1", "u1", relative_path, raw, expected=ABSENT)
    for serving_file in tmp_path.glob("catalog.sqlite3*"):
        serving_file.unlink()
    recreated_catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    recreated_worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        recreated_catalog,
        InMemoryQueueStore(),
    )

    rebuilt = recreated_worker.rebuild_owner("t1", "u1")

    assert rebuilt["projected"] == 0
    assert rebuilt["skipped"] == 1
    state = recreated_catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
    )
    assert state is not None
    assert state["deletion_generation"] == barrier.deletion_generation
    assert state["deletion_event_digest"] == barrier.deletion_event_digest
    assert state["deletion_status"] == "HARD_ERASED"
    assert _document_records(recreated_catalog, document_id) == []


def test_rebuild_replays_erasure_epoch_created_before_hard_barrier(tmp_path: Path) -> None:
    documents, _catalog_store, worker = _components(tmp_path)
    document_id = new_document_id()
    relative_path = "knowledge/topics/crash-before-barrier.md"
    raw = render_new_document(document_id, "# Secret\n\nCRASH_SECRET must not return.\n")
    created = documents.create("t1", "u1", relative_path, raw, expected=ABSENT)
    worker.rebuild_owner("t1", "u1")
    erase_store = MemoryDocumentEraseStore(tmp_path)
    erase_store.begin(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
        relative_path=relative_path,
        source_digest=created.raw_sha256,
        document_revision_floor=0,
        projection_generation_floor=0,
        backend_names=("local.live_source",),
        independent_evidence_retained=(),
        started_at="2026-07-18T00:00:00+00:00",
    )
    for serving_file in tmp_path.glob("catalog.sqlite3*"):
        serving_file.unlink()
    recreated_catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    recreated_worker = MemoryDocumentProjectionWorker(
        documents,
        worker.control_store,
        recreated_catalog,
        InMemoryQueueStore(),
    )

    rebuilt = recreated_worker.rebuild_owner("t1", "u1")

    assert rebuilt["projected"] == 0
    assert rebuilt["skipped"] == 1
    assert _document_records(recreated_catalog, document_id, include_inactive=True) == []
    barrier = worker.control_store.load_publication_barrier("t1", "u1", document_id)
    assert barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED
    state = recreated_catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
    )
    assert state is not None and state["deletion_status"] == "HARD_ERASED"


def test_hard_erase_seals_barrier_above_full_rebuild_serving_generation(tmp_path: Path) -> None:
    documents, catalog, worker = _components(tmp_path)
    document_id = new_document_id()
    relative_path = "knowledge/topics/generation-ahead.md"
    raw = render_new_document(document_id, "# Generation one\n\nOLD_SECRET\n")
    created = documents.create("t1", "u1", relative_path, raw, expected=ABSENT)
    worker.rebuild_owner("t1", "u1")
    updated_raw = render_new_document(document_id, "# Generation two\n\nNEW_SECRET\n")
    updated = documents.replace(
        "t1",
        "u1",
        document_id,
        updated_raw,
        expected_state=documents.read_state("t1", "u1", relative_path),
    )
    worker.rebuild_owner("t1", "u1")
    before = catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
    )
    assert before is not None and before["projection_generation"] == 2
    eraser = MemoryDocumentEraser(
        documents,
        worker.control_store,
        MemoryDocumentRevisionStore(tmp_path),
        cleanup_backends=(MemoryDocumentCatalogEraseBackend(worker),),
        erase_store=MemoryDocumentEraseStore(worker.control_store.root),
    )

    result = eraser.hard_erase(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
        expected_source_digest=updated.raw_sha256,
        relative_path=relative_path,
    )

    assert result.record.status is DocumentEraseStatus.ERASED
    barrier = worker.control_store.load_publication_barrier("t1", "u1", document_id)
    assert barrier is not None and barrier.deletion_generation >= 3
    state = catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=document_id,
    )
    assert state is not None
    assert state["deletion_generation"] == barrier.deletion_generation
    assert state["deletion_status"] == "HARD_ERASED"
    assert _document_records(catalog, document_id, include_inactive=True) == []
    assert created.raw_sha256 != updated.raw_sha256


def test_tombstone_journal_lookup_is_exact_owner_scoped(tmp_path: Path) -> None:
    documents, catalog, worker = _components(tmp_path)
    document_id = new_document_id()
    for owner in ("a-owner", "z-owner"):
        raw = render_new_document(document_id, f"# {owner}\n\nowner scoped\n")
        documents.create("t1", owner, "knowledge/topics/shared-id.md", raw, expected=ABSENT)
        worker.rebuild_owner("t1", owner)
    catalog.clear(tenant_id="t1")

    catalog.tombstone_memory_document_projection(
        tenant_id="t1",
        owner_user_id="z-owner",
        document_id=document_id,
        deletion_generation=1,
        deletion_event_digest="d" * 64,
        deletion_status="HARD_ERASED",
        relative_path="knowledge/topics/shared-id.md",
    )

    with catalog._connect() as connection:
        rows = connection.execute(
            "SELECT source_uri, owner_user_id, status FROM context_projection_journal "
            "WHERE tenant_id = 't1' AND projector_kind = 'memory_document' ORDER BY source_uri"
        ).fetchall()
    by_uri = {str(row["source_uri"]): (str(row["owner_user_id"]), str(row["status"])) for row in rows}
    a_uri = MemoryDocumentPathPolicy.document_uri("a-owner", document_id)
    z_uri = MemoryDocumentPathPolicy.document_uri("z-owner", document_id)
    assert by_uri[a_uri] == ("a-owner", "PENDING")
    assert by_uri[z_uri] == ("z-owner", "TOMBSTONED")


def test_hard_erase_replays_derived_cleanup_after_catalog_commit_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    documents = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    relations = InMemoryRelationStore()
    vectors = InMemoryVectorStore()
    worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        catalog,
        InMemoryQueueStore(),
        vector_store=vectors,
        embedding_provider=DeterministicEmbeddingProvider(),
        relation_store=relations,
    )
    target_id = new_document_id()
    source_id = new_document_id()
    target_path = "knowledge/topics/target.md"
    source_path = "knowledge/topics/source.md"
    target = documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nTARGET_SECRET\n"),
        expected=ABSENT,
    )
    documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    target_uri = MemoryDocumentPathPolicy.document_uri("u1", target_id)
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1")
    assert any(metadata.get("document_id") == target_id for _vector, metadata in vectors.rows.values())
    original_tombstone = catalog.tombstone_memory_document_projection
    crashed = False

    def crash_after_catalog_commit(**kwargs):  # noqa: ANN003, ANN202
        nonlocal crashed
        result = original_tombstone(**kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("fault after Catalog tombstone commit")
        return result

    monkeypatch.setattr(catalog, "tombstone_memory_document_projection", crash_after_catalog_commit)
    eraser = MemoryDocumentEraser(
        documents,
        controls,
        MemoryDocumentRevisionStore(tmp_path),
        cleanup_backends=(MemoryDocumentCatalogEraseBackend(worker),),
        erase_store=MemoryDocumentEraseStore(controls.root),
    )

    first = eraser.hard_erase(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=target_id,
        expected_source_digest=target.raw_sha256,
        relative_path=target_path,
    )
    assert first.record.status is DocumentEraseStatus.ERASE_PENDING
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1")
    assert any(metadata.get("document_id") == target_id for _vector, metadata in vectors.rows.values())

    replay = eraser.hard_erase(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=target_id,
        expected_source_digest=target.raw_sha256,
        relative_path=target_path,
    )

    assert replay.record.status is DocumentEraseStatus.ERASED
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []
    assert not any(metadata.get("document_id") == target_id for _vector, metadata in vectors.rows.values())
    assert any(metadata.get("document_id") == source_id for _vector, metadata in vectors.rows.values())


def test_owner_relation_lock_orders_cross_document_link_add_after_target_erase(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    documents = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    relations = InMemoryRelationStore()
    worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        catalog,
        InMemoryQueueStore(),
        relation_store=relations,
    )
    target_id = new_document_id()
    source_id = new_document_id()
    target_path = "knowledge/topics/target.md"
    source_path = "knowledge/topics/source.md"
    target = documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nErase me.\n"),
        expected=ABSENT,
    )
    source = documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    updated_source = documents.replace(
        "t1",
        "u1",
        source_id,
        render_new_document(source_id, "# Source updated\n\n[Target](target.md)\n"),
        expected_state=documents.read_state("t1", "u1", source_path),
    )
    link_ready = Event()
    release_link = Event()
    original_add = relations.add_relation

    def pause_before_add(relation, *, tenant_id):  # noqa: ANN001, ANN202
        if relation.source_uri == MemoryDocumentPathPolicy.document_uri("u1", source_id):
            link_ready.set()
            assert release_link.wait(5)
        return original_add(relation, tenant_id=tenant_id)

    monkeypatch.setattr(relations, "add_relation", pause_before_add)
    eraser = MemoryDocumentEraser(
        documents,
        controls,
        MemoryDocumentRevisionStore(tmp_path),
        cleanup_backends=(MemoryDocumentCatalogEraseBackend(worker),),
        erase_store=MemoryDocumentEraseStore(controls.root),
    )
    errors: list[BaseException] = []
    erase_results = []

    def rebuild_source() -> None:
        try:
            worker.rebuild_owner("t1", "u1")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def erase_target() -> None:
        try:
            erase_results.append(
                eraser.hard_erase(
                    tenant_id="t1",
                    owner_user_id="u1",
                    document_id=target_id,
                    expected_source_digest=target.raw_sha256,
                    relative_path=target_path,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    rebuild_thread = Thread(target=rebuild_source)
    rebuild_thread.start()
    assert link_ready.wait(5)
    erase_thread = Thread(target=erase_target)
    erase_thread.start()
    erase_thread.join(0.25)
    release_link.set()
    rebuild_thread.join(5)
    erase_thread.join(5)

    assert errors == []
    assert not rebuild_thread.is_alive() and not erase_thread.is_alive()
    assert erase_results and erase_results[0].record.status is DocumentEraseStatus.ERASED
    target_uri = MemoryDocumentPathPolicy.document_uri("u1", target_id)
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []
    assert updated_source.raw_sha256 != source.raw_sha256


def test_soft_forget_replays_exact_derivative_cleanup_after_catalog_commit_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    documents, controls, catalog, relations, vectors, worker = _rich_components(tmp_path)
    source_id = new_document_id()
    target_id = new_document_id()
    source_path = "knowledge/topics/source.md"
    target_path = "knowledge/topics/target.md"
    source = documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nKeep target derivatives.\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    source_uri = MemoryDocumentPathPolicy.document_uri("u1", source_id)
    target_uri = MemoryDocumentPathPolicy.document_uri("u1", target_id)
    assert relations.relations_of(source_uri, tenant_id="t1", owner_user_id="u1")
    source_vector_ids = {
        row_id for row_id, (_vector, metadata) in vectors.rows.items() if metadata.get("document_id") == source_id
    }
    target_vector_ids = {
        row_id for row_id, (_vector, metadata) in vectors.rows.items() if metadata.get("document_id") == target_id
    }
    assert source_vector_ids and target_vector_ids
    _write_soft_barrier(controls, source_id, source_path)
    documents.delete(
        "t1",
        "u1",
        source_id,
        expected_state=documents.read_state("t1", "u1", source_path),
    )
    original_delete = vectors.delete_vector
    failed = False

    def fail_first_vector_delete(row_id: str) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("fault after soft Catalog tombstone commit")
        original_delete(row_id)

    monkeypatch.setattr(vectors, "delete_vector", fail_first_vector_delete)
    with pytest.raises(RuntimeError, match="soft Catalog tombstone"):
        worker.rebuild_owner("t1", "u1")
    state = catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=source_id,
    )
    assert state is not None and state["deletion_status"] == "SOFT_FORGOTTEN"
    assert source_vector_ids <= set(vectors.rows)
    assert relations.relations_of(source_uri, tenant_id="t1", owner_user_id="u1")

    replay = worker.rebuild_owner("t1", "u1")

    assert replay["projected"] == 0
    assert not (source_vector_ids & set(vectors.rows))
    assert target_vector_ids <= set(vectors.rows)
    assert relations.relations_of(source_uri, tenant_id="t1", owner_user_id="u1") == []
    target_state = catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=target_id,
    )
    assert target_state is not None and target_state["deletion_status"] == ""
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []
    assert source.raw_sha256


def test_soft_barrier_rebuild_after_sqlite_loss_purges_orphan_derivatives(tmp_path: Path) -> None:
    documents, controls, _catalog, relations, vectors, worker = _rich_components(tmp_path)
    source_id = new_document_id()
    target_id = new_document_id()
    source_path = "knowledge/topics/source.md"
    target_path = "knowledge/topics/target.md"
    documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nKeep me.\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    source_uri = MemoryDocumentPathPolicy.document_uri("u1", source_id)
    source_vectors = {
        row_id for row_id, (_vector, metadata) in vectors.rows.items() if metadata.get("document_id") == source_id
    }
    target_vectors = {
        row_id for row_id, (_vector, metadata) in vectors.rows.items() if metadata.get("document_id") == target_id
    }
    assert source_vectors and target_vectors
    assert relations.relations_of(source_uri, tenant_id="t1", owner_user_id="u1")
    _write_soft_barrier(controls, source_id, source_path)
    documents.delete(
        "t1",
        "u1",
        source_id,
        expected_state=documents.read_state("t1", "u1", source_path),
    )
    for serving_file in tmp_path.glob("catalog.sqlite3*"):
        serving_file.unlink()
    recreated_catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    recreated_worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        recreated_catalog,
        InMemoryQueueStore(),
        vector_store=vectors,
        embedding_provider=DeterministicEmbeddingProvider(),
        relation_store=relations,
    )

    rebuilt = recreated_worker.rebuild_owner("t1", "u1")

    assert rebuilt["projected"] == 1
    assert not (source_vectors & set(vectors.rows))
    assert target_vectors <= set(vectors.rows)
    assert relations.relations_of(source_uri, tenant_id="t1", owner_user_id="u1") == []
    state = recreated_catalog.get_memory_document_projection_state(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=source_id,
    )
    assert state is not None and state["deletion_status"] == "SOFT_FORGOTTEN"


def test_soft_barrier_blocks_inbound_link_until_exact_explicit_restore(tmp_path: Path) -> None:
    documents, controls, catalog, relations, _vectors, worker = _rich_components(tmp_path)
    source_id = new_document_id()
    target_id = new_document_id()
    source_path = "knowledge/topics/source.md"
    target_path = "knowledge/topics/target.md"
    source = documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    target = documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nRestore lineage.\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    target_uri = MemoryDocumentPathPolicy.document_uri("u1", target_id)
    barrier = _write_soft_barrier(controls, target_id, target_path)
    worker._mirror_barrier(barrier)
    updated_source = documents.replace(
        "t1",
        "u1",
        source_id,
        render_new_document(source_id, "# Source updated\n\n[Target](target.md)\n"),
        expected_state=documents.read_state("t1", "u1", source_path),
    )
    worker._publish_live(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=source_id,
        relative_path=source_path,
        source_digest=updated_source.raw_sha256,
        document_revision=2,
        projection_generation=2,
        expected_previous_generation=1,
    )
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []
    controls.write_control(
        DocumentControlRecord(
            tenant_id="t1",
            owner_user_id="u1",
            document_id=target_id,
            relative_path=target_path,
            raw_sha256=target.raw_sha256,
            size=target.size,
            logical_revision=3,
            projection_generation=3,
            status="present",
            last_event_id=f"memchg_{'b' * 64}",
            updated_at="2026-07-18T00:00:01+00:00",
            restored_from_deletion_generation=barrier.deletion_generation,
        )
    )
    worker._publish_live(
        tenant_id="t1",
        owner_user_id="u1",
        document_id=target_id,
        relative_path=target_path,
        source_digest=target.raw_sha256,
        document_revision=3,
        projection_generation=3,
        expected_previous_generation=0,
        restored_from_deletion_generation=barrier.deletion_generation,
    )
    source_record = _document_records(catalog, source_id)[0]
    worker._replace_document_links(
        source_record,
        documents.read_raw("t1", "u1", document_id=source_id),
    )
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1")
    assert source.raw_sha256 != updated_source.raw_sha256


def test_soft_mirror_owner_lock_removes_link_added_before_fence_linearizes(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    documents, controls, _catalog, relations, _vectors, worker = _rich_components(tmp_path)
    source_id = new_document_id()
    target_id = new_document_id()
    source_path = "knowledge/topics/source.md"
    target_path = "knowledge/topics/target.md"
    documents.create(
        "t1",
        "u1",
        source_path,
        render_new_document(source_id, "# Source\n\n[Target](target.md)\n"),
        expected=ABSENT,
    )
    documents.create(
        "t1",
        "u1",
        target_path,
        render_new_document(target_id, "# Target\n\nSoft delete.\n"),
        expected=ABSENT,
    )
    worker.rebuild_owner("t1", "u1")
    documents.replace(
        "t1",
        "u1",
        source_id,
        render_new_document(source_id, "# Source changed\n\n[Target](target.md)\n"),
        expected_state=documents.read_state("t1", "u1", source_path),
    )
    link_ready = Event()
    release_link = Event()
    original_add = relations.add_relation

    def pause_before_add(relation, *, tenant_id):  # noqa: ANN001, ANN202
        if relation.source_uri == MemoryDocumentPathPolicy.document_uri("u1", source_id):
            link_ready.set()
            assert release_link.wait(5)
        original_add(relation, tenant_id=tenant_id)

    monkeypatch.setattr(relations, "add_relation", pause_before_add)
    errors: list[BaseException] = []

    def rebuild_source() -> None:
        try:
            worker.rebuild_owner("t1", "u1")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    mirror_results: list[tuple[str, ...]] = []

    def mirror_soft_barrier(barrier: DocumentPublicationBarrier) -> None:
        try:
            mirror_results.append(worker._mirror_barrier(barrier))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    rebuild_thread = Thread(target=rebuild_source)
    rebuild_thread.start()
    assert link_ready.wait(5)
    barrier = _write_soft_barrier(controls, target_id, target_path)
    mirror_thread = Thread(target=mirror_soft_barrier, args=(barrier,))
    mirror_thread.start()
    mirror_thread.join(0.25)
    release_link.set()
    rebuild_thread.join(5)
    mirror_thread.join(5)

    assert errors == []
    assert not rebuild_thread.is_alive() and not mirror_thread.is_alive()
    assert mirror_results
    target_uri = MemoryDocumentPathPolicy.document_uri("u1", target_id)
    assert relations.relations_of(target_uri, tenant_id="t1", owner_user_id="u1") == []


def test_remove_obsolete_drains_more_than_one_relation_adapter_batch(tmp_path: Path) -> None:
    documents = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    relations = InMemoryRelationStore()
    worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        catalog,
        InMemoryQueueStore(),
        relation_store=relations,
    )
    uri = MemoryDocumentPathPolicy.document_uri("u1", new_document_id())
    record_key = "memory-document:u1:obsolete"
    for index in range(1_501):
        relations.add_relation(
            ContextRelation(
                source_uri=uri,
                relation_type="links_to",
                target_uri=f"memoryos://resources/target/{index}",
                metadata={
                    "tenant_id": "t1",
                    "owner_user_id": "u1",
                    "catalog_record_key": record_key,
                },
            ),
            tenant_id="t1",
        )

    worker._remove_obsolete("t1", (record_key,), {record_key: uri})

    assert relations.relations_of(uri, tenant_id="t1", owner_user_id="u1") == []


def test_remove_obsolete_uses_exact_metadata_fallback_for_noncanonical_vector_row(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    documents = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    vectors = InMemoryVectorStore()
    worker = MemoryDocumentProjectionWorker(
        documents,
        controls,
        catalog,
        InMemoryQueueStore(),
        vector_store=vectors,
        embedding_provider=DeterministicEmbeddingProvider(),
    )
    record_key = "memory-document:u1:legacy-vector"
    expected_row_id = vector_row_id("t1", record_key)
    metadata = {"tenant_id": "t1", "catalog_record_key": record_key}
    vectors.upsert_vector(expected_row_id, [1.0], metadata)
    embedding, stored_metadata = vectors.rows.pop(expected_row_id)
    vectors._discard_metadata_identity(expected_row_id, stored_metadata)
    legacy_row_id = "memoryos-vector://legacy/noncanonical-row"
    vectors.rows[legacy_row_id] = (embedding, stored_metadata)
    vectors._index_metadata_identity(legacy_row_id, stored_metadata)
    original_delete = vectors.delete_vector

    def ignore_only_expected_row_id(row_id: str) -> None:
        if row_id != expected_row_id:
            original_delete(row_id)

    monkeypatch.setattr(vectors, "delete_vector", ignore_only_expected_row_id)

    worker._remove_obsolete("t1", (record_key,), {})

    assert legacy_row_id not in vectors.rows
