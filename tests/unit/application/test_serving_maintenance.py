from __future__ import annotations

from dataclasses import dataclass

import pytest

from memoryos.application.context.maintenance import (
    CallbackDocumentServingMaintenance,
    CatalogDocumentProjectionVerifier,
    DerivedServingMaintenanceService,
)
from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.extensions import NoContextIndexPolicy, NoDomainOverlay
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.memory.documents.model import ManagedDocument, ScanGeneration, UnmanagedDocument


def _scan(*, owner: str = "alice", unmanaged: bool = False) -> ScanGeneration:
    managed = ManagedDocument(
        relative_path="profile.md",
        document_id="memdoc_0123456789abcdef01234567",
        raw_sha256="a" * 64,
        size=20,
    )
    registrations = (managed,)
    if unmanaged:
        registrations += (
            UnmanagedDocument(
                relative_path="topics/unmanaged.md",
                raw_sha256="b" * 64,
                size=12,
                reason="missing document id",
            ),
        )
    return ScanGeneration(
        generation_id="scan_0123456789abcdef",
        tenant_id="tenant-a",
        owner_user_id=owner,
        root_identity="root-a",
        observed_at="2026-07-18T00:00:00+00:00",
        complete=True,
        registrations=registrations,
    )


def test_document_maintenance_orders_safe_scan_rebuild_and_verification() -> None:
    calls: list[str] = []

    def full_scan(tenant: str, owner: str) -> ScanGeneration:
        calls.append(f"scan:{tenant}:{owner}")
        return _scan(owner=owner)

    def rebuild(tenant: str, owner: str) -> dict[str, int]:
        calls.append(f"rebuild:{tenant}:{owner}")
        return {"projected": 1}

    def verify(tenant: str, owner: str, scan: ScanGeneration) -> dict[str, object]:
        calls.append(f"verify:{tenant}:{owner}:{scan.generation_id}")
        return {"consistent": True, "projected_documents": len(scan.managed)}

    maintenance = CallbackDocumentServingMaintenance(
        full_scan=full_scan,
        rebuild_owner=rebuild,
        verify_owner=verify,
    )
    result = maintenance.rebuild_and_verify(
        tenant_id="tenant-a",
        owner_user_id="alice",
    )

    assert calls == [
        "scan:tenant-a:alice",
        "rebuild:tenant-a:alice",
        "scan:tenant-a:alice",
        "verify:tenant-a:alice:scan_0123456789abcdef",
    ]
    assert result["consistent"] is True
    assert result["documents"] == 1


def test_document_maintenance_rejects_unmanaged_scan_before_publication() -> None:
    rebuilt = False

    def rebuild(_tenant: str, _owner: str) -> dict[str, int]:
        nonlocal rebuilt
        rebuilt = True
        return {"projected": 0}

    maintenance = CallbackDocumentServingMaintenance(
        full_scan=lambda _tenant, owner: _scan(owner=owner, unmanaged=True),
        rebuild_owner=rebuild,
        verify_owner=lambda _tenant, _owner, _scan: {"consistent": True},
    )

    with pytest.raises(RuntimeError, match="unmanaged registrations"):
        maintenance.rebuild_and_verify(tenant_id="tenant-a", owner_user_id="alice")
    assert rebuilt is False


def test_document_maintenance_rejects_source_change_during_rebuild() -> None:
    scans = 0

    def full_scan(_tenant: str, owner: str) -> ScanGeneration:
        nonlocal scans
        scans += 1
        result = _scan(owner=owner)
        if scans == 1:
            return result
        changed = ManagedDocument(
            relative_path="profile.md",
            document_id=result.managed[0].document_id,
            raw_sha256="c" * 64,
            size=result.managed[0].size,
        )
        return ScanGeneration(
            generation_id="scan_changed",
            tenant_id=result.tenant_id,
            owner_user_id=result.owner_user_id,
            root_identity=result.root_identity,
            observed_at=result.observed_at,
            complete=True,
            registrations=(changed,),
        )

    maintenance = CallbackDocumentServingMaintenance(
        full_scan=full_scan,
        rebuild_owner=lambda _tenant, _owner: {"projected": 1},
        verify_owner=lambda _tenant, _owner, _scan: {"consistent": True},
    )

    with pytest.raises(RuntimeError, match="changed during"):
        maintenance.rebuild_and_verify(tenant_id="tenant-a", owner_user_id="alice")


def test_tenant_document_maintenance_requires_bounded_owner_enumeration() -> None:
    maintenance = CallbackDocumentServingMaintenance(
        full_scan=lambda _tenant, owner: _scan(owner=owner),
        rebuild_owner=lambda _tenant, _owner: {"projected": 1},
        verify_owner=lambda _tenant, _owner, _scan: {"consistent": True},
        owner_user_ids=lambda _tenant, _limit: ("bob", "alice"),
        max_owners=2,
    )

    result = maintenance.rebuild_and_verify(tenant_id="tenant-a")

    assert [item["owner_user_id"] for item in result["owner_results"]] == ["alice", "bob"]
    assert result["owners"] == 2


def test_catalog_document_verifier_binds_source_digest_path_and_generation(tmp_path) -> None:
    index = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    timestamp = "2026-07-18T00:00:00+00:00"
    record = CatalogRecord(
        record_key="memory-document:alice:memdoc_0123456789abcdef01234567",
        uri="memoryos://user/alice/memory/documents/memdoc_0123456789abcdef01234567",
        tenant_id="tenant-a",
        owner_user_id="alice",
        context_type="memory",
        source_kind="markdown_memory_document",
        record_kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
        tree_paths=("memories/profile",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="Profile",
        l1_text="Profile",
        source_uri="memoryos://user/alice/memory/documents/memdoc_0123456789abcdef01234567",
        source_digest="a" * 64,
        document_id="memdoc_0123456789abcdef01234567",
        document_kind="profile",
        document_revision=1,
        projection_generation=1,
        metadata={"relative_path": "profile.md"},
    )
    index.replace_memory_document_projection(
        record,
        (),
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )
    verifier = CatalogDocumentProjectionVerifier(index)

    assert verifier("tenant-a", "alice", _scan())["consistent"] is True
    changed = ScanGeneration(
        generation_id="scan_changed",
        tenant_id="tenant-a",
        owner_user_id="alice",
        root_identity="root-a",
        observed_at=timestamp,
        complete=True,
        registrations=(
            ManagedDocument(
                relative_path="profile.md",
                document_id="memdoc_0123456789abcdef01234567",
                raw_sha256="b" * 64,
                size=20,
            ),
        ),
    )
    mismatch = verifier("tenant-a", "alice", changed)
    assert mismatch["consistent"] is False
    assert mismatch["issues"] == [
        "projection source mismatch:memdoc_0123456789abcdef01234567"
    ]


@dataclass
class _DocumentServing:
    rebuilt_owner: str = ""

    def rebuild_and_verify(self, *, tenant_id: str, owner_user_id: str | None = None):  # noqa: ANN201
        assert tenant_id == "tenant-a"
        self.rebuilt_owner = str(owner_user_id or "")
        return {"consistent": True, "documents": 1}

    def verify(self, *, tenant_id: str, owner_user_id: str | None = None):  # noqa: ANN201
        assert tenant_id == "tenant-a"
        return {"consistent": True, "owner_user_id": owner_user_id}


def test_serving_service_rebuilds_ordinary_rows_and_injected_documents(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source", tenant_id="tenant-a")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    obj = ContextObject(
        uri="memoryos://resources/user/alice/notes/one",
        context_type=ContextType.RESOURCE,
        title="ordinary note",
        owner_user_id="alice",
        tenant_id="tenant-a",
    )
    source.write_object(obj, content="ordinary serving content")
    documents = _DocumentServing()
    service = DerivedServingMaintenanceService(
        source,
        index,
        relations,
        tenant_id="tenant-a",
        document_serving=documents,
        domain_overlay=NoDomainOverlay(),
        index_policy=NoContextIndexPolicy(),
    )

    result = service.rebuild_index(owner_user_id="alice")

    assert result["consistent"] is True
    assert result["missing_index"] == []
    assert documents.rebuilt_owner == "alice"
    assert index.search(
        "ordinary",
        tenant_id="tenant-a",
        filters={"owner_user_id": "alice"},
    )
