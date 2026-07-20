from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from infrastructure.store.contracts.vector import vector_row_id
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from tests.support.persistence import InMemoryVectorStore

_NOW = "2026-07-17T02:00:00+00:00"


def _ordinary_record(
    tenant_id: str,
    *,
    title: str,
    owner_user_id: str,
    record_key: str = "shared-record-key",
) -> CatalogRecord:
    return CatalogRecord(
        record_key=record_key,
        uri="memoryos://contexts/shared",
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        context_type="session",
        source_kind="message",
        record_kind=CatalogRecordKind.MESSAGE.value,
        primary_tree_path="sessions/shared-session",
        tree_paths=("sessions/shared-session",),
        created_at=_NOW,
        updated_at=_NOW,
        event_time=_NOW,
        ingested_at=_NOW,
        transaction_time=_NOW,
        title=title,
        l0_text=title,
        l1_text=f"{title} tenant-safe-lexical-token",
        source_uri="memoryos://sources/shared",
        source_digest=f"digest-{tenant_id}",
        source_revision=1,
    )


def _document_record(
    *,
    generation: int,
    digest: str,
    record_key: str = "document-record",
) -> CatalogRecord:
    return CatalogRecord(
        record_key=record_key,
        uri="memoryos://users/alice/memory/documents/document-1",
        tenant_id="tenant-a",
        owner_user_id="alice",
        context_type="memory",
        source_kind="markdown",
        record_kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
        primary_tree_path="memories/knowledge/topics/catalog",
        tree_paths=("memories/knowledge/topics/catalog",),
        created_at=_NOW,
        updated_at=_NOW,
        event_time=_NOW,
        ingested_at=_NOW,
        transaction_time=_NOW,
        title="Catalog document",
        l0_text="Catalog document",
        l1_text=f"document generation {generation}",
        source_uri="memoryos://sources/memory/document-1",
        source_digest=digest,
        source_revision=generation,
        document_id="document-1",
        document_kind="knowledge",
        document_revision=generation,
        projection_generation=generation,
        metadata={"relative_path": "knowledge/catalog.md"},
    )


def _block_record(
    block_id: str,
    *,
    generation: int,
    digest: str,
) -> CatalogRecord:
    return CatalogRecord(
        record_key=f"block-record-{block_id}",
        uri=f"memoryos://users/alice/memory/documents/document-1/blocks/{block_id}",
        tenant_id="tenant-a",
        owner_user_id="alice",
        context_type="memory",
        source_kind="markdown",
        record_kind=CatalogRecordKind.MEMORY_BLOCK.value,
        primary_tree_path="memories/knowledge/topics/catalog",
        tree_paths=("memories/knowledge/topics/catalog",),
        created_at=_NOW,
        updated_at=_NOW,
        event_time=_NOW,
        ingested_at=_NOW,
        transaction_time=_NOW,
        title=f"Block {block_id}",
        l0_text=f"Block {block_id}",
        l1_text=f"block {block_id} generation {generation} unique{block_id}projectiontoken",
        source_uri="memoryos://sources/memory/document-1",
        source_digest=digest,
        source_revision=generation,
        document_id="document-1",
        block_id=block_id,
        document_kind="knowledge",
        document_revision=generation,
        projection_generation=generation,
    )


def test_greenfield_schema_has_composite_identity_and_no_removed_tables(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)

    assert store.catalog_schema_version() == 1
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(contexts)")}
        primary = tuple(
            row[1]
            for row in sorted(conn.execute("PRAGMA table_info(contexts)"), key=lambda item: item[5])
            if row[5]
        )
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert primary == ("tenant_id", "record_key")
    assert {
        "document_id",
        "block_id",
        "document_kind",
        "document_revision",
        "projection_generation",
    } <= columns
    assert {
        "context_projection_journal",
        "memory_document_projection_state",
    } <= tables
    assert not any(
        token in table
        for table in tables
        for token in ("migration", "shadow", "equivalence", "validity")
    )


@pytest.mark.parametrize(
    "fts_map_columns",
    (
        "record_key TEXT PRIMARY KEY, fts_rowid INTEGER NOT NULL UNIQUE",
        (
            "tenant_id TEXT NOT NULL, record_key TEXT NOT NULL, "
            "fts_rowid INTEGER NOT NULL, PRIMARY KEY (tenant_id, record_key)"
        ),
    ),
    ids=("legacy-record-key-primary", "missing-rowid-unique-identity"),
)
def test_catalog_rejects_unsupported_fts_map_identity(
    tmp_path,
    fts_map_columns: str,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    SQLiteIndexStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE context_fts_map")
        conn.execute(f"CREATE TABLE context_fts_map ({fts_map_columns})")

    with pytest.raises(
        RuntimeError,
        match="unsupported Catalog auxiliary layout for context_fts_map",
    ):
        SQLiteIndexStore(path)


def test_same_record_key_fts_and_delete_are_tenant_isolated(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)
    tenant_a = _ordinary_record("tenant-a", title="Alpha document", owner_user_id="alice")
    tenant_b = _ordinary_record("tenant-b", title="Beta document", owner_user_id="bob")

    store.upsert_catalog(tenant_a, tenant_id="tenant-a")
    store.upsert_catalog(tenant_b, tenant_id="tenant-b")

    assert store.get_catalog(tenant_a.record_key, tenant_id="tenant-a").title == tenant_a.title  # type: ignore[union-attr]
    assert store.get_catalog(tenant_b.record_key, tenant_id="tenant-b").title == tenant_b.title  # type: ignore[union-attr]
    assert [hit.title for hit in store.search_catalog("tenant-safe-lexical-token", tenant_id="tenant-a")] == [
        "Alpha document"
    ]
    assert [hit.title for hit in store.search_catalog("tenant-safe-lexical-token", tenant_id="tenant-b")] == [
        "Beta document"
    ]

    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM context_fts_map WHERE record_key = ?",
            (tenant_a.record_key,),
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM context_paths WHERE record_key = ?",
            (tenant_a.record_key,),
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM context_acl_grants WHERE record_key = ?",
            (tenant_a.record_key,),
        ).fetchone()[0] == 2

    assert store.delete_catalog(tenant_a.record_key, tenant_id="tenant-a") is True
    assert store.get_catalog(tenant_a.record_key, tenant_id="tenant-a") is None
    assert store.get_catalog(tenant_b.record_key, tenant_id="tenant-b").title == tenant_b.title  # type: ignore[union-attr]
    assert store.search_catalog("tenant-safe-lexical-token", tenant_id="tenant-a") == []
    assert [hit.title for hit in store.search_catalog("tenant-safe-lexical-token", tenant_id="tenant-b")] == [
        "Beta document"
    ]

    with pytest.raises(TypeError):
        store.get_catalog(tenant_b.record_key)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        store.upsert_catalog(tenant_b)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        store.delete_catalog(tenant_b.record_key)  # type: ignore[call-arg]


def test_acl_and_path_subqueries_correlate_tenant_and_record_key(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    tenant_a = _ordinary_record("tenant-a", title="Alpha document", owner_user_id="alice")
    tenant_b = _ordinary_record("tenant-b", title="Beta document", owner_user_id="bob")
    store.upsert_catalog(tenant_a, tenant_id="tenant-a")
    store.upsert_catalog(tenant_b, tenant_id="tenant-b")

    assert store.list_catalog(
        tenant_id="tenant-a",
        filters={"principal_owner_id": "bob", "target_paths": ["sessions/shared-session"]},
    ) == []
    assert [
        record.title
        for record in store.list_catalog(
            tenant_id="tenant-b",
            filters={"principal_owner_id": "bob", "target_paths": ["sessions/shared-session"]},
        )
    ] == ["Beta document"]


def test_structured_filters_use_tenant_first_indexes(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    record = _ordinary_record("tenant-a", title="Alpha document", owner_user_id="alice")
    store.upsert_catalog(record, tenant_id="tenant-a")

    record_kind_plan = store.explain_structured_query(
        tenant_id="tenant-a",
        filters={"record_kind": CatalogRecordKind.MESSAGE.value},
    )
    path_plan = store.explain_structured_query(
        tenant_id="tenant-a",
        filters={"target_paths": ["sessions/shared-session"]},
    )
    document_plan = store.explain_structured_query(
        tenant_id="tenant-a",
        filters={"document_id": "document-1"},
    )

    assert any("idx_contexts_tenant_record_kind_updated" in line for line in record_kind_plan)
    assert any("idx_context_path_closure_ancestor" in line for line in path_plan)
    assert any("idx_contexts_tenant_document_id" in line for line in document_plan)


def test_document_projection_generation_cas_is_atomic_and_idempotent(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)
    document_v1 = _document_record(generation=1, digest="digest-v1")
    blocks_v1 = (
        _block_record("one", generation=1, digest="digest-v1"),
        _block_record("two", generation=1, digest="digest-v1"),
    )

    assert store.replace_memory_document_projection(
        document_v1,
        blocks_v1,
        tenant_id="tenant-a",
        owner_user_id="alice",
        expected_previous_generation=0,
    ) == ()
    assert store.get_memory_document_projection_state(
        tenant_id="tenant-a",
        owner_user_id="alice",
        document_id="document-1",
    ) == {
        "tenant_id": "tenant-a",
        "owner_user_id": "alice",
        "document_id": "document-1",
        "relative_path": "knowledge/catalog.md",
        "source_digest": "digest-v1",
        "projection_generation": 1,
        "projection_status": "PROJECTED",
        "deletion_generation": 0,
        "deletion_event_digest": "",
        "deletion_status": "",
    }
    assert store.replace_memory_document_projection(
        document_v1,
        blocks_v1,
        tenant_id="tenant-a",
        owner_user_id="alice",
        expected_previous_generation=0,
    ) == ()

    with pytest.raises(ValueError, match="digest conflicts"):
        store.replace_memory_document_projection(
            replace(document_v1, source_digest="different-digest"),
            tuple(replace(block, source_digest="different-digest") for block in blocks_v1),
            tenant_id="tenant-a",
            owner_user_id="alice",
            expected_previous_generation=1,
        )

    document_v2 = _document_record(generation=2, digest="digest-v2")
    blocks_v2 = (
        _block_record("two", generation=2, digest="digest-v2"),
        _block_record("three", generation=2, digest="digest-v2"),
    )
    assert store.replace_memory_document_projection(
        document_v2,
        blocks_v2,
        tenant_id="tenant-a",
        owner_user_id="alice",
        expected_previous_generation=1,
    ) == ("block-record-one",)

    with pytest.raises(ValueError, match="stale"):
        store.replace_memory_document_projection(
            _document_record(generation=3, digest="digest-v3"),
            (_block_record("four", generation=3, digest="digest-v3"),),
            tenant_id="tenant-a",
            owner_user_id="alice",
            expected_previous_generation=1,
        )

    current = store.list_catalog(
        tenant_id="tenant-a",
        filters={"document_id": "document-1"},
        limit=10,
    )
    assert {record.record_key for record in current} == {
        "document-record",
        "block-record-two",
        "block-record-three",
    }
    assert {record.projection_generation for record in current} == {2}
    assert store.search_catalog("uniqueoneprojectiontoken", tenant_id="tenant-a") == []
    assert [
        hit.metadata["catalog_record_key"]
        for hit in store.search_catalog("uniquethreeprojectiontoken", tenant_id="tenant-a")
    ] == ["block-record-three"]

    with sqlite3.connect(path) as conn:
        state = conn.execute(
            "SELECT projection_generation, source_digest FROM memory_document_projection_state "
            "WHERE tenant_id = 'tenant-a' AND owner_user_id = 'alice' AND document_id = 'document-1'"
        ).fetchone()
        journal = conn.execute(
            "SELECT source_digest FROM context_projection_journal WHERE tenant_id = 'tenant-a' "
            "AND projector_kind = 'memory_document'"
        ).fetchone()
    assert state == (2, "digest-v2")
    assert journal == ("digest-v2",)


def test_memory_projection_cannot_bypass_atomic_publication_api(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    document = _document_record(generation=1, digest="digest-v1")
    block = _block_record("one", generation=1, digest="digest-v1")
    store.replace_memory_document_projection(
        document,
        (block,),
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )

    ordinary_collision = replace(
        _ordinary_record("tenant-a", title="collision", owner_user_id="alice"),
        record_key=document.record_key,
    )
    with pytest.raises(ValueError, match="replace_memory_document_projection"):
        store.upsert_catalog(ordinary_collision, tenant_id="tenant-a")
    with pytest.raises(ValueError, match="tombstone_memory_document_projection"):
        store.delete_catalog(document.record_key, tenant_id="tenant-a")
    with pytest.raises(ValueError, match="tombstone_memory_document_projection"):
        store.delete_index(document.uri, tenant_id="tenant-a")
    with pytest.raises(ValueError, match="tombstone_memory_document_projection"):
        store.enqueue_tombstone(
            tenant_id="tenant-a",
            record_key=document.record_key,
            reason="generic-delete-must-not-bypass-document-barrier",
        )

    loaded_document = store.get_catalog(document.record_key, tenant_id="tenant-a")
    loaded_block = store.get_catalog(block.record_key, tenant_id="tenant-a")
    assert loaded_document is not None
    assert loaded_block is not None
    assert (loaded_document.document_id, loaded_document.source_digest) == (
        document.document_id,
        document.source_digest,
    )
    assert (loaded_block.block_id, loaded_block.source_digest) == (
        block.block_id,
        block.source_digest,
    )


def test_document_projection_cas_serializes_concurrent_publishers(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)
    store.replace_memory_document_projection(
        _document_record(generation=1, digest="digest-v1"),
        (_block_record("one", generation=1, digest="digest-v1"),),
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )

    def publish(digest: str) -> str:
        local = SQLiteIndexStore(path)
        try:
            local.replace_memory_document_projection(
                _document_record(generation=2, digest=digest),
                (_block_record("two", generation=2, digest=digest),),
                1,
                tenant_id="tenant-a",
                owner_user_id="alice",
            )
        except ValueError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(publish, ("digest-v2-a", "digest-v2-b")))

    assert sorted(outcomes) == ["conflict", "published"]
    records = store.list_catalog(
        tenant_id="tenant-a",
        filters={"document_id": "document-1"},
        limit=10,
    )
    assert {record.projection_generation for record in records} == {2}
    assert len({record.source_digest for record in records}) == 1


def test_document_projection_rolls_back_all_rows_if_state_publish_fails(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    first_document = _document_record(generation=1, digest="digest-v1")
    first_blocks = (_block_record("one", generation=1, digest="digest-v1"),)
    store.replace_memory_document_projection(
        first_document,
        first_blocks,
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )
    second_document = replace(
        _document_record(generation=1, digest="digest-document-2"),
        record_key="document-record-2",
        uri="memoryos://users/alice/memory/documents/document-2",
        source_uri="memoryos://sources/memory/document-2",
        document_id="document-2",
    )
    second_block = replace(
        _block_record("other", generation=1, digest="digest-document-2"),
        record_key="block-record-document-2",
        uri="memoryos://users/alice/memory/documents/document-2/blocks/other",
        source_uri="memoryos://sources/memory/document-2",
        document_id="document-2",
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_memory_document_projection(
            second_document,
            (second_block,),
            0,
            tenant_id="tenant-a",
            owner_user_id="alice",
        )

    assert {record.record_key for record in store.list_catalog(tenant_id="tenant-a")} == {
        "document-record",
        "block-record-one",
    }
    assert store.search_catalog("uniqueotherprojectiontoken", tenant_id="tenant-a") == []


def test_document_tombstone_is_atomic_and_soft_restore_is_explicit(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)
    document_v1 = _document_record(generation=1, digest="digest-v1")
    blocks_v1 = (
        _block_record("one", generation=1, digest="digest-v1"),
        _block_record("two", generation=1, digest="digest-v1"),
    )
    store.replace_memory_document_projection(
        document_v1,
        blocks_v1,
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )

    obsolete = store.tombstone_memory_document_projection(
        tenant_id="tenant-a",
        owner_user_id="alice",
        document_id="document-1",
        deletion_generation=2,
        deletion_event_digest="soft-delete-digest",
        deletion_status="SOFT_FORGOTTEN",
    )

    assert set(obsolete) == {"document-record", "block-record-one", "block-record-two"}
    assert store.list_catalog(tenant_id="tenant-a") == []
    assert store.search_catalog("uniqueoneprojectiontoken", tenant_id="tenant-a") == []
    with sqlite3.connect(path) as conn:
        serving_counts = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "contexts",
                "context_fts_map",
                "context_paths",
                "context_path_closure",
                "context_path_acl",
                "context_acl_grants",
            )
        )
        state = conn.execute(
            "SELECT projection_generation, projection_status, deletion_generation, "
            "deletion_event_digest, deletion_status FROM memory_document_projection_state "
            "WHERE tenant_id = 'tenant-a' AND owner_user_id = 'alice' AND document_id = 'document-1'"
        ).fetchone()
        journal = conn.execute(
            "SELECT status, source_digest FROM context_projection_journal "
            "WHERE tenant_id = 'tenant-a' AND projector_kind = 'memory_document'"
        ).fetchone()
    assert serving_counts == (0, 0, 0, 0, 0, 0)
    assert state == (0, "TOMBSTONED", 2, "soft-delete-digest", "SOFT_FORGOTTEN")
    assert journal == ("TOMBSTONED", "soft-delete-digest")
    assert store.tombstone_memory_document_projection(
        tenant_id="tenant-a",
        owner_user_id="alice",
        document_id="document-1",
        deletion_generation=2,
        deletion_event_digest="soft-delete-digest",
        deletion_status="SOFT_FORGOTTEN",
    ) == ()
    with pytest.raises(ValueError, match="conflicts"):
        store.tombstone_memory_document_projection(
            tenant_id="tenant-a",
            owner_user_id="alice",
            document_id="document-1",
            deletion_generation=2,
            deletion_event_digest="different-delete-digest",
            deletion_status="SOFT_FORGOTTEN",
        )

    document_v3 = _document_record(generation=3, digest="digest-v3")
    blocks_v3 = (_block_record("three", generation=3, digest="digest-v3"),)
    with pytest.raises(ValueError, match="blocked"):
        store.replace_memory_document_projection(
            document_v3,
            blocks_v3,
            0,
            tenant_id="tenant-a",
            owner_user_id="alice",
        )
    assert store.replace_memory_document_projection(
        document_v3,
        blocks_v3,
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
        restore_soft_deleted=True,
    ) == ()
    with sqlite3.connect(path) as conn:
        restored = conn.execute(
            "SELECT projection_generation, deletion_generation, deletion_status "
            "FROM memory_document_projection_state WHERE tenant_id = 'tenant-a' "
            "AND owner_user_id = 'alice' AND document_id = 'document-1'"
        ).fetchone()
    assert restored == (3, 2, "")
    with pytest.raises(ValueError, match="stale"):
        store.tombstone_memory_document_projection(
            tenant_id="tenant-a",
            owner_user_id="alice",
            document_id="document-1",
            deletion_generation=2,
            deletion_event_digest="soft-delete-digest",
            deletion_status="SOFT_FORGOTTEN",
        )
    assert {record.record_key for record in store.list_catalog(tenant_id="tenant-a")} == {
        "document-record",
        "block-record-three",
    }

    hard_obsolete = store.tombstone_memory_document_projection(
        tenant_id="tenant-a",
        owner_user_id="alice",
        document_id="document-1",
        deletion_generation=4,
        deletion_event_digest="hard-delete-digest",
        deletion_status="HARD_ERASED",
    )
    assert set(hard_obsolete) == {"document-record", "block-record-three"}
    assert store.tombstone_memory_document_projection(
        tenant_id="tenant-a",
        owner_user_id="alice",
        document_id="document-1",
        deletion_generation=4,
        deletion_event_digest="hard-delete-digest",
        deletion_status="HARD_ERASED",
    ) == ()
    hard_state = store.get_memory_document_projection_state(
        tenant_id="tenant-a",
        owner_user_id="alice",
        document_id="document-1",
    )
    assert hard_state is not None
    assert hard_state["relative_path"] == ""
    persisted = b"".join(
        candidate.read_bytes()
        for candidate in path.parent.glob(f"{path.name}*")
        if candidate.is_file()
    )
    assert b"knowledge/catalog.md" not in persisted
    with pytest.raises(ValueError, match="hard-erased"):
        store.replace_memory_document_projection(
            _document_record(generation=5, digest="digest-v5"),
            (_block_record("five", generation=5, digest="digest-v5"),),
            0,
            tenant_id="tenant-a",
            owner_user_id="alice",
            restore_soft_deleted=True,
        )


def test_tombstones_with_same_id_cannot_delete_another_tenant(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    tenant_a = _ordinary_record("tenant-a", title="Alpha document", owner_user_id="alice")
    tenant_b = _ordinary_record("tenant-b", title="Beta document", owner_user_id="bob")
    store.upsert_catalog(tenant_a, tenant_id="tenant-a")
    store.upsert_catalog(tenant_b, tenant_id="tenant-b")

    store.enqueue_tombstone(
        tenant_id="tenant-a",
        record_key=tenant_a.record_key,
        reason="delete",
        tombstone_id="shared-tombstone",
    )
    store.enqueue_tombstone(
        tenant_id="tenant-b",
        record_key=tenant_b.record_key,
        reason="delete",
        tombstone_id="shared-tombstone",
    )
    store.mark_tombstone_applied("shared-tombstone", tenant_id="tenant-a")

    assert store.get_catalog(tenant_a.record_key, tenant_id="tenant-a") is None
    assert store.get_catalog(tenant_b.record_key, tenant_id="tenant-b").title == tenant_b.title  # type: ignore[union-attr]
    assert store.get_pending_tombstones(tenant_id="tenant-a") == []
    assert [row["tenant_id"] for row in store.get_pending_tombstones(tenant_id="tenant-b")] == [
        "tenant-b"
    ]


def test_serving_cleanup_preserves_document_deletion_barrier(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)
    document = _document_record(generation=1, digest="digest-v1")
    blocks = (_block_record("one", generation=1, digest="digest-v1"),)
    store.replace_memory_document_projection(
        document,
        blocks,
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE memory_document_projection_state SET deletion_generation = 2, "
            "deletion_event_digest = 'delete-digest', deletion_status = 'SOFT_FORGOTTEN' "
            "WHERE tenant_id = 'tenant-a' AND owner_user_id = 'alice' AND document_id = 'document-1'"
        )

    store.clear(tenant_id="tenant-a")

    with sqlite3.connect(path) as conn:
        barrier = conn.execute(
            "SELECT deletion_generation, deletion_event_digest, deletion_status "
            "FROM memory_document_projection_state WHERE tenant_id = 'tenant-a' "
            "AND owner_user_id = 'alice' AND document_id = 'document-1'"
        ).fetchone()
    assert barrier == (2, "delete-digest", "SOFT_FORGOTTEN")
    with pytest.raises(ValueError, match="not newer"):
        store.replace_memory_document_projection(
            _document_record(generation=2, digest="digest-v2"),
            (_block_record("two", generation=2, digest="digest-v2"),),
            1,
            tenant_id="tenant-a",
            owner_user_id="alice",
        )
    with pytest.raises(ValueError, match="blocked"):
        store.replace_memory_document_projection(
            _document_record(generation=3, digest="digest-v3"),
            (_block_record("three", generation=3, digest="digest-v3"),),
            1,
            tenant_id="tenant-a",
            owner_user_id="alice",
        )


def test_serving_cleanup_allows_active_document_full_rebuild(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = SQLiteIndexStore(path)
    document = _document_record(generation=1, digest="digest-v1")
    blocks = (_block_record("one", generation=1, digest="digest-v1"),)
    store.replace_memory_document_projection(
        document,
        blocks,
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )

    store.clear(tenant_id="tenant-a")

    assert store.list_catalog(tenant_id="tenant-a") == []
    with sqlite3.connect(path) as conn:
        state = conn.execute(
            "SELECT projection_generation, source_digest, projection_status "
            "FROM memory_document_projection_state WHERE tenant_id = 'tenant-a' "
            "AND owner_user_id = 'alice' AND document_id = 'document-1'"
        ).fetchone()
    assert state == (0, "", "PENDING")
    assert store.replace_memory_document_projection(
        document,
        blocks,
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    ) == ()
    assert {record.record_key for record in store.list_catalog(tenant_id="tenant-a")} == {
        "document-record",
        "block-record-one",
    }


def test_vector_catalog_identity_is_tenant_qualified_and_has_no_public_uri_alias() -> None:
    store = InMemoryVectorStore()
    public_uri = "memoryos://contexts/shared"
    metadata_a = {
        "tenant_id": "tenant-a",
        "catalog_record_key": "shared-record-key",
        "public_uri": public_uri,
    }
    metadata_b = {**metadata_a, "tenant_id": "tenant-b"}
    row_a = vector_row_id("tenant-a", "shared-record-key")
    row_b = vector_row_id("tenant-b", "shared-record-key")

    with pytest.raises(ValueError, match="row ID"):
        store.upsert_vector(public_uri, [1.0, 0.0], metadata_a)
    with pytest.raises(ValueError, match="catalog_record_key"):
        store.upsert_vector(public_uri, [1.0, 0.0], {"tenant_id": "tenant-a"})
    store.upsert_vector(row_a, [1.0, 0.0], metadata_a)
    store.upsert_vector(row_b, [0.0, 1.0], metadata_b)

    assert row_a != row_b
    assert store.get_vector_metadata(public_uri) is None
    assert store.search_vector_candidates([1.0, 0.0], [public_uri]) == []
    assert [hit.uri for hit in store.search_vector_candidates([1.0, 0.0], [row_a, row_b])] == [
        row_a,
        row_b,
    ]
    with pytest.raises(ValueError, match="tenant_id"):
        store.delete_by_filter({"catalog_record_key": "shared-record-key"})
    assert store.delete_by_filter(
        {"tenant_id": "tenant-a", "catalog_record_key": "shared-record-key"}
    ) == 1
    assert store.get_vector_metadata(row_a) is None
    assert store.get_vector_metadata(row_b) == metadata_b
