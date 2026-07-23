"""可重建 Context Serving 投影的离线修复与一致性校验。"""

from __future__ import annotations

from dataclasses import asdict
from threading import RLock
from typing import Any

from infrastructure.context.contracts import ContextDomainOverlay, ContextIndexPolicy
from infrastructure.context.maintenance.index_consistency import IndexConsistencyService
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore


def _trusted_scope_segment(value: object, name: str) -> str:
    segment = str(value).strip()
    if not segment or segment in {".", ".."} or any(char in segment for char in ("/", "\\", "\x00")):
        raise ValueError(f"{name} is invalid")
    return segment


class DerivedServingMaintenanceService:
    """重建并验证普通 Context 的可重建 Serving 状态。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        *,
        tenant_id: str,
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
        self.retention_manager = retention_manager
        self.readiness = readiness
        self.domain_overlay = domain_overlay
        self.index_policy = index_policy
        self.serving_lock = serving_lock or RLock()

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        self._require_ready()
        owner = _trusted_scope_segment(owner_user_id, "owner_user_id") if owner_user_id is not None else None
        try:
            with self.serving_lock:
                ordinary = self._index_consistency().rebuild(owner_user_id=owner)
                retention = self._run_retention() if owner is None else {"configured": False}
                payload = self._consistency_payload(ordinary)
                payload.update({"tenant_id": self.tenant_id, "retention": retention})
                return payload
        except Exception as exc:
            self._mark_not_ready(exc, artifact="serving_rebuild")
            raise

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        self._require_ready()
        owner = _trusted_scope_segment(owner_user_id, "owner_user_id") if owner_user_id is not None else None
        ordinary = self._index_consistency().verify(owner_user_id=owner)
        payload = self._consistency_payload(ordinary)
        payload["tenant_id"] = self.tenant_id
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


__all__ = ["DerivedServingMaintenanceService"]
