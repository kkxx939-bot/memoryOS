"""上下文删除、冷热分层和派生投影清理服务。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from threading import RLock
from typing import Any

from infrastructure.context.maintenance.retention import CatalogRetentionManager
from infrastructure.context.maintenance.tombstone import ProjectionTombstoneService, TombstoneRunResult
from infrastructure.store.contracts.source import SourceStore


class ContextLifecycleService:
    """集中处理上下文生命周期，不再借用 ``ContextDB`` 暴露维护入口。"""

    def __init__(
        self,
        source_store: SourceStore,
        tombstone_service: ProjectionTombstoneService,
        *,
        tenant_id: str,
        retention_manager: CatalogRetentionManager | None = None,
        readiness: Any | None = None,
        serving_lock: RLock | None = None,
    ) -> None:
        self.source_store = source_store
        self.tombstone_service = tombstone_service
        self.retention_manager = retention_manager
        self.tenant_id = str(tenant_id or "").strip()
        if not self.tenant_id:
            raise ValueError("context lifecycle service requires an explicit tenant_id")
        self.readiness = readiness
        self.serving_lock = serving_lock or RLock()

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    @contextmanager
    def _mutation_fence(self) -> Iterator[None]:
        """让事实源退役、墓碑处理和 Serving 维护观察同一发布边界。"""

        with self.serving_lock:
            yield

    def delete_context(self, uri: str, *, reason: str = "context_deleted") -> dict[str, Any]:
        """先记录耐久墓碑，再退役事实源并清理派生投影。"""

        with self._mutation_fence():
            self._require_ready()
            obj = self.source_store.read_object(uri)
            tenant_id = str(obj.tenant_id or self.tenant_id)
            tombstones = self.tombstone_service.enqueue_uri(
                uri,
                tenant_id=tenant_id,
                reason=reason,
                require_source_retired=True,
            )
            self.source_store.soft_delete(uri, reason)
            result = self.tombstone_service.process_tombstones(tombstones, tenant_id=tenant_id)
            self._require_complete_cleanup(result)
            return {
                "uri": uri,
                "tenant_id": tenant_id,
                "tombstone_ids": list(tombstones),
                "processed": list(result.processed),
                "stale": list(result.stale),
            }

    def delete_session_context(self, session_id: str, *, reason: str = "session_deleted") -> dict[str, Any]:
        """删除 Session Serving 投影，但保留不可变 SessionArchive 证据。"""

        with self._mutation_fence():
            self._require_ready()
            tombstones = self.tombstone_service.enqueue_session(
                session_id,
                tenant_id=self.tenant_id,
                reason=reason,
            )
            result = self.tombstone_service.process_tombstones(tombstones, tenant_id=self.tenant_id)
            self._require_complete_cleanup(result)
            return {
                "session_id": session_id,
                "tenant_id": self.tenant_id,
                "tombstone_ids": list(tombstones),
                "processed": list(result.processed),
                "stale": list(result.stale),
                "evidence_retained": True,
            }

    def run_retention_cycle(self, *, now: datetime | None = None) -> dict[str, Any]:
        """执行一次有界的冷热分层、向量回收和派生状态清理。"""

        with self._mutation_fence():
            self._require_ready()
            manager = self._retention_manager()
            tiers = manager.apply_serving_tiers(tenant_id=self.tenant_id, now=now)
            vectors = manager.gc_vectors(tenant_id=self.tenant_id)
            stale = manager.gc_stale_projections(tenant_id=self.tenant_id)
            auxiliary = manager.gc_auxiliary_state(tenant_id=self.tenant_id, now=now)
            return {
                "tenant_id": self.tenant_id,
                "tiers": asdict(tiers),
                "vectors": asdict(vectors),
                "stale": asdict(stale),
                "auxiliary": asdict(auxiliary),
            }

    def restore_cold_context(self, record_key: str, *, now: datetime | None = None) -> dict[str, Any]:
        """把一条精确冷层记录恢复为有界热访问。"""

        with self._mutation_fence():
            self._require_ready()
            return self._retention_manager().restore_cold_record(
                record_key,
                tenant_id=self.tenant_id,
                now=now,
            ).to_dict()

    def enqueue_context_tombstones(
        self,
        uri: str,
        *,
        reason: str,
    ) -> tuple[str, ...]:
        """普通上下文退役前，耐久记录它的全部派生投影。"""

        with self._mutation_fence():
            self._require_ready()
            return self.tombstone_service.enqueue_uri(
                uri,
                tenant_id=self.tenant_id,
                reason=reason,
                require_source_retired=True,
            )

    def process_projection_tombstones(
        self,
        tombstone_ids: list[str] | tuple[str, ...],
    ) -> TombstoneRunResult:
        """精确重放一组墓碑，避免全局队列扫描造成饥饿。"""

        with self._mutation_fence():
            self._require_ready()
            return self.tombstone_service.process_tombstones(
                tombstone_ids,
                tenant_id=self.tenant_id,
            )

    def _retention_manager(self) -> CatalogRetentionManager:
        if self.retention_manager is None:
            raise RuntimeError("context lifecycle operation requires CatalogRetentionManager")
        return self.retention_manager

    @staticmethod
    def _require_complete_cleanup(result: TombstoneRunResult) -> None:
        if result.failed:
            raise RuntimeError("derived projection tombstone cleanup is retryable but incomplete")


__all__ = ["ContextLifecycleService"]
