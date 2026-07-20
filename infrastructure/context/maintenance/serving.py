"""可重建 Serving 投影的离线修复与一致性校验。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Protocol

from infrastructure.context.contracts import ContextDomainOverlay, ContextIndexPolicy
from infrastructure.context.maintenance.index_consistency import IndexConsistencyService
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
)


def _trusted_scope_segment(value: object, name: str) -> str:
    segment = str(value).strip()
    if not segment or segment in {".", ".."} or any(char in segment for char in ("/", "\\", "\x00")):
        raise ValueError(f"{name} is invalid")
    return segment


class DocumentServingMaintenance(Protocol):
    """由运行时注入的有界 Markdown 原文重建边界。"""

    def rebuild_and_verify(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> Mapping[str, Any]: ...

    def verify(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DocumentOwnerServingResult:
    owner_user_id: str
    scan_generation_id: str
    scanned_documents: int
    rebuild: Mapping[str, Any]
    verification: Mapping[str, Any]


class CallbackDocumentServingMaintenance:
    """组合严格的全量扫描、发布和校验流程。

    该适配器不会递归遍历运行根来发现用户。租户级任务必须提供有界的所有者列表；
    校验阶段接收发布前的同一安全扫描代次，用它逐一比较实时文档身份、路径、原文
    摘要和可重建 Catalog 投影。
    """

    def __init__(
        self,
        *,
        full_scan: Callable[[str, str], Any],
        rebuild_owner: Callable[[str, str], Mapping[str, Any]],
        verify_owner: Callable[[str, str, Any], Mapping[str, Any]],
        owner_user_ids: Callable[[str, int], Sequence[str]] | None = None,
        max_owners: int = 1_000,
        max_documents_per_owner: int = 10_000,
    ) -> None:
        if not 1 <= max_owners <= 10_000:
            raise ValueError("max_owners must be between 1 and 10000")
        if not 1 <= max_documents_per_owner <= 100_000:
            raise ValueError("max_documents_per_owner must be between 1 and 100000")
        self.full_scan = full_scan
        self.rebuild_owner = rebuild_owner
        self.verify_owner = verify_owner
        self.owner_user_ids = owner_user_ids
        self.max_owners = max_owners
        self.max_documents_per_owner = max_documents_per_owner

    def rebuild_and_verify(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> Mapping[str, Any]:
        tenant = _trusted_scope_segment(tenant_id, "tenant_id")
        results: list[DocumentOwnerServingResult] = []
        for owner in self._owners(tenant, owner_user_id):
            before = self._safe_scan(tenant, owner)
            rebuilt = self.rebuild_owner(tenant, owner)
            if not isinstance(rebuilt, Mapping):
                raise TypeError("document projection rebuild returned an invalid result")
            scan = self._safe_scan(tenant, owner)
            if self._scan_fingerprint(before) != self._scan_fingerprint(scan):
                raise RuntimeError("Markdown source changed during document projection rebuild")
            verified = self.verify_owner(tenant, owner, scan)
            self._require_verified(verified)
            results.append(
                DocumentOwnerServingResult(
                    owner_user_id=owner,
                    scan_generation_id=str(scan.generation_id),
                    scanned_documents=len(scan.managed),
                    rebuild=dict(rebuilt),
                    verification=dict(verified),
                )
            )
        return self._report(tenant, results)

    def verify(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> Mapping[str, Any]:
        tenant = _trusted_scope_segment(tenant_id, "tenant_id")
        results: list[DocumentOwnerServingResult] = []
        for owner in self._owners(tenant, owner_user_id):
            scan = self._safe_scan(tenant, owner)
            verified = self.verify_owner(tenant, owner, scan)
            self._require_verified(verified)
            results.append(
                DocumentOwnerServingResult(
                    owner_user_id=owner,
                    scan_generation_id=str(scan.generation_id),
                    scanned_documents=len(scan.managed),
                    rebuild={},
                    verification=dict(verified),
                )
            )
        return self._report(tenant, results)

    def _owners(self, tenant_id: str, owner_user_id: str | None) -> tuple[str, ...]:
        if owner_user_id is not None:
            return (_trusted_scope_segment(owner_user_id, "owner_user_id"),)
        if self.owner_user_ids is None:
            raise RuntimeError("tenant document maintenance requires a bounded owner provider")
        raw = self.owner_user_ids(tenant_id, self.max_owners + 1)
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            raise TypeError("document owner provider returned an invalid result")
        if len(raw) > self.max_owners:
            raise RuntimeError("document owner enumeration exceeded its bound")
        owners = tuple(_trusted_scope_segment(item, "owner_user_id") for item in raw)
        if len(set(owners)) != len(owners):
            raise RuntimeError("document owner enumeration contains duplicates")
        return tuple(sorted(owners))

    def _safe_scan(self, tenant_id: str, owner_user_id: str) -> Any:
        scan = self.full_scan(tenant_id, owner_user_id)
        if str(getattr(scan, "tenant_id", "")) != tenant_id or str(getattr(scan, "owner_user_id", "")) != owner_user_id:
            raise PermissionError("document full scan crossed its tenant or owner boundary")
        if not bool(getattr(scan, "complete", False)):
            raise RuntimeError("document projection rebuild requires a complete full scan")
        if tuple(getattr(scan, "errors", ()) or ()):
            raise RuntimeError("document projection rebuild scan reported traversal errors")
        if tuple(getattr(scan, "unsafe_paths", ()) or ()):
            raise RuntimeError("document projection rebuild scan reported unsafe paths")
        registrations = tuple(getattr(scan, "registrations", ()) or ())
        managed = tuple(getattr(scan, "managed", ()) or ())
        if len(registrations) != len(managed):
            raise RuntimeError("document projection rebuild found unmanaged registrations")
        if len(managed) > self.max_documents_per_owner:
            raise RuntimeError("document projection rebuild exceeded its document bound")
        ids = tuple(str(getattr(item, "document_id", "")) for item in managed)
        paths = tuple(str(getattr(item, "relative_path", "")) for item in managed)
        if any(not item for item in ids + paths) or len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise RuntimeError("document projection rebuild scan contains duplicate identities or paths")
        return scan

    @staticmethod
    def _scan_fingerprint(scan: Any) -> tuple[str, tuple[tuple[str, str, str, int], ...]]:
        return (
            str(scan.root_identity),
            tuple(
                sorted(
                    (
                        str(item.document_id),
                        str(item.relative_path),
                        str(item.raw_sha256),
                        int(item.size),
                    )
                    for item in tuple(scan.managed)
                )
            ),
        )

    @staticmethod
    def _require_verified(result: Mapping[str, Any]) -> None:
        if not isinstance(result, Mapping):
            raise TypeError("document projection verifier returned an invalid result")
        if result.get("consistent") is not True:
            raise RuntimeError("document projection generation verification failed")

    @staticmethod
    def _report(
        tenant_id: str,
        results: Sequence[DocumentOwnerServingResult],
    ) -> Mapping[str, Any]:
        return {
            "tenant_id": tenant_id,
            "owners": len(results),
            "documents": sum(item.scanned_documents for item in results),
            "consistent": all(item.verification.get("consistent") is True for item in results),
            "owner_results": [asdict(item) for item in results],
        }


class CatalogDocumentProjectionVerifier:
    """比较一次安全 Markdown 扫描与租户 Catalog 投影代次。"""

    def __init__(self, catalog_store: Any, *, max_documents: int = 10_000) -> None:
        if not 1 <= max_documents <= 100_000:
            raise ValueError("max_documents must be between 1 and 100000")
        if not callable(getattr(catalog_store, "scan_catalog_batch", None)):
            raise TypeError("document projection verification requires bounded Catalog scans")
        if not callable(getattr(catalog_store, "get_memory_document_projection_state", None)):
            raise TypeError("document projection verification requires document state reads")
        self.catalog_store = catalog_store
        self.max_documents = max_documents

    def __call__(
        self,
        tenant_id: str,
        owner_user_id: str,
        scan: Any,
    ) -> Mapping[str, Any]:
        live = {str(item.document_id): item for item in tuple(scan.managed)}
        records = self._records(tenant_id, owner_user_id)
        projected: dict[str, CatalogRecord] = {}
        issues: list[str] = []
        for record in records:
            if record.tenant_id != tenant_id or record.owner_user_id != owner_user_id:
                raise PermissionError("document Catalog verification crossed its scope boundary")
            if not record.document_id or record.document_id in projected:
                issues.append(f"duplicate projection identity:{record.document_id}")
                continue
            projected[record.document_id] = record
        for document_id in sorted(live.keys() | projected.keys()):
            source = live.get(document_id)
            projected_record = projected.get(document_id)
            if source is None:
                issues.append(f"stale projection:{document_id}")
                continue
            if projected_record is None:
                issues.append(f"missing projection:{document_id}")
                continue
            state = self.catalog_store.get_memory_document_projection_state(
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                document_id=document_id,
            )
            if not isinstance(state, Mapping):
                issues.append(f"missing projection state:{document_id}")
                continue
            if str(state.get("deletion_status") or ""):
                issues.append(f"deleted projection is still live:{document_id}")
            expected = (
                str(source.raw_sha256),
                str(source.relative_path),
            )
            actual = (
                str(projected_record.source_digest),
                str(projected_record.metadata.get("relative_path") or ""),
            )
            state_actual = (
                str(state.get("source_digest") or ""),
                str(state.get("relative_path") or ""),
            )
            if actual != expected or state_actual != expected:
                issues.append(f"projection source mismatch:{document_id}")
            generation = int(state.get("projection_generation") or 0)
            if generation <= 0 or projected_record.projection_generation != generation:
                issues.append(f"projection generation mismatch:{document_id}")
        return {
            "consistent": not issues,
            "live_documents": len(live),
            "projected_documents": len(projected),
            "issues": sorted(issues),
        }

    def _records(self, tenant_id: str, owner_user_id: str) -> tuple[CatalogRecord, ...]:
        records: list[CatalogRecord] = []
        cursor = ""
        while True:
            batch = self.catalog_store.scan_catalog_batch(
                tenant_id=tenant_id,
                after_record_key=cursor,
                filters={
                    "tenant_id": tenant_id,
                    "owner_user_id": owner_user_id,
                    "record_kind": CatalogRecordKind.MEMORY_DOCUMENT.value,
                    "include_inactive": True,
                    "serving_tier": tuple(item.value for item in ServingTier),
                    "projection_status": tuple(item.value for item in CatalogProjectionStatus),
                },
                limit=min(1_000, self.max_documents + 1),
            )
            if not isinstance(batch, list) or any(not isinstance(item, CatalogRecord) for item in batch):
                raise TypeError("document Catalog scan returned an invalid batch")
            if not batch:
                break
            records.extend(batch)
            if len(records) > self.max_documents:
                raise RuntimeError("document Catalog verification exceeded its bound")
            next_cursor = batch[-1].record_key
            if next_cursor <= cursor:
                raise RuntimeError("document Catalog verification did not advance")
            cursor = next_cursor
        return tuple(records)


class DerivedServingMaintenanceService:
    """先修复普通记录，再重建 Markdown 文档投影。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        *,
        tenant_id: str,
        document_serving: DocumentServingMaintenance | None = None,
        retention_manager: Any | None = None,
        readiness: Any | None = None,
        domain_overlay: ContextDomainOverlay,
        index_policy: ContextIndexPolicy,
        serving_lock: RLock | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.tenant_id = _trusted_scope_segment(tenant_id, "tenant_id")
        self.document_serving = document_serving
        self.retention_manager = retention_manager
        self.readiness = readiness
        self.domain_overlay = domain_overlay
        self.index_policy = index_policy
        self.serving_lock = serving_lock or RLock()

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        """执行离线有界重建，不能把 Catalog 当作事实源。"""

        self._require_ready()
        owner = _trusted_scope_segment(owner_user_id, "owner_user_id") if owner_user_id is not None else None
        try:
            with self.serving_lock:
                ordinary = self._index_consistency().rebuild(owner_user_id=owner)
                documents: Mapping[str, Any] = {"configured": False, "consistent": True}
                if self.document_serving is not None:
                    documents = self.document_serving.rebuild_and_verify(
                        tenant_id=self.tenant_id,
                        owner_user_id=owner,
                    )
                retention = self._run_retention() if owner is None else {"configured": False}
                payload = self._consistency_payload(ordinary)
                payload.update(
                    {
                        "tenant_id": self.tenant_id,
                        "documents": dict(documents),
                        "retention": retention,
                    }
                )
                payload["consistent"] = bool(ordinary.consistent and documents.get("consistent") is True)
                return payload
        except Exception as exc:
            self._mark_not_ready(exc, artifact="serving_rebuild")
            raise

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        self._require_ready()
        owner = _trusted_scope_segment(owner_user_id, "owner_user_id") if owner_user_id is not None else None
        ordinary = self._index_consistency().verify(owner_user_id=owner)
        documents: Mapping[str, Any] = {"configured": False, "consistent": True}
        if self.document_serving is not None:
            documents = self.document_serving.verify(
                tenant_id=self.tenant_id,
                owner_user_id=owner,
            )
        payload = self._consistency_payload(ordinary)
        payload.update({"tenant_id": self.tenant_id, "documents": dict(documents)})
        payload["consistent"] = bool(ordinary.consistent and documents.get("consistent") is True)
        return payload

    def _index_consistency(self) -> IndexConsistencyService:
        return IndexConsistencyService(
            self.source_store,
            self.index_store,
            self.relation_store,
            tenant_id=self.tenant_id,
            domain_overlay=self.domain_overlay,
            index_policy=self.index_policy,
        )

    def _run_retention(self) -> dict[str, Any]:
        if self.retention_manager is None:
            return {"configured": False}
        tiers = self.retention_manager.apply_serving_tiers(tenant_id=self.tenant_id)
        vectors = self.retention_manager.gc_vectors(tenant_id=self.tenant_id)
        stale = self.retention_manager.gc_stale_projections(tenant_id=self.tenant_id)
        auxiliary = self.retention_manager.gc_auxiliary_state(tenant_id=self.tenant_id)
        if vectors.tombstones_failed or stale.tombstones_failed:
            raise RuntimeError("serving retention cleanup remained incomplete")
        return {
            "configured": True,
            "tiers": asdict(tiers),
            "vectors": asdict(vectors),
            "stale": asdict(stale),
            "auxiliary": asdict(auxiliary),
        }

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    def _mark_not_ready(self, error: BaseException, *, artifact: str) -> None:
        mark_not_ready = getattr(self.readiness, "mark_not_ready", None)
        if callable(mark_not_ready):
            mark_not_ready(
                f"serving consistency failure: {type(error).__name__}: {error}",
                details={"artifact": artifact, "error_type": type(error).__name__},
            )

    @staticmethod
    def _consistency_payload(result: Any) -> dict[str, Any]:
        return {
            "source_count": result.source_count,
            "indexed_count": result.index_count,
            "missing_index": result.missing_in_index,
            "dangling_index": result.orphan_index,
            "deleted_or_archived_in_default_search": result.deleted_or_archived_in_default_search,
            "broken_relations": result.broken_relations,
            "consistent": result.consistent,
        }


__all__ = [
    "CatalogDocumentProjectionVerifier",
    "CallbackDocumentServingMaintenance",
    "DerivedServingMaintenanceService",
    "DocumentOwnerServingResult",
    "DocumentServingMaintenance",
]
