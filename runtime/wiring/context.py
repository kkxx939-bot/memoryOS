"""Context 查询、维护和派生清理对象的装配。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import cast

from infrastructure.context.facade import ContextDB
from infrastructure.context.layers import MemoryDocumentContextOverlay
from infrastructure.context.maintenance import (
    CallbackDocumentServingMaintenance,
    CatalogDocumentProjectionVerifier,
    ContextLifecycleService,
    DerivedServingMaintenanceService,
)
from infrastructure.context.maintenance.retention import CatalogRetentionManager, RetentionPolicy
from infrastructure.context.maintenance.retention_policy import RetentionCatalogStore
from infrastructure.context.maintenance.tombstone import ProjectionTombstoneService
from runtime.config import RuntimeConfig
from runtime.container import ContextRuntime, MemoryRuntime, StoreRuntime


@dataclass(frozen=True)
class ContextMaintenance:
    """在事务提交器之前即可创建的派生层清理对象。"""

    tombstone_service: ProjectionTombstoneService
    retention_manager: CatalogRetentionManager


def wire_context_maintenance(stores: StoreRuntime, config: RuntimeConfig) -> ContextMaintenance:
    tombstone = ProjectionTombstoneService(
        stores.index,
        source_store=stores.source,
        vector_store=stores.vector,
        relation_store=stores.relation,
    )
    retention = CatalogRetentionManager(
        cast(RetentionCatalogStore, stores.index),
        vector_store=stores.vector,
        tombstone_service=tombstone,
        policy=RetentionPolicy.from_config(config.retention.to_mapping()),
    )
    return ContextMaintenance(tombstone_service=tombstone, retention_manager=retention)


def wire_context(
    stores: StoreRuntime,
    config: RuntimeConfig,
    *,
    readiness,  # noqa: ANN001
    committer,  # noqa: ANN001
    memory: MemoryRuntime,
    maintenance: ContextMaintenance,
    owner_user_ids,  # noqa: ANN001
) -> ContextRuntime:
    """连接 Context 门面、管理服务和 Markdown Memory serving 维护。"""

    serving_lock = RLock()
    facade = ContextDB(
        stores.source,
        stores.index,
        stores.relation,
        relation_committer=committer,
        readiness=readiness,
        tenant_id=config.tenant_id,
        serving_lock=serving_lock,
    )
    document_serving = CallbackDocumentServingMaintenance(
        full_scan=memory.document_store.full_scan,
        rebuild_owner=memory.projection_worker.rebuild_owner,
        verify_owner=CatalogDocumentProjectionVerifier(stores.index),
        owner_user_ids=owner_user_ids,
        max_documents_per_owner=config.memory_scan_max_files,
    )
    administration = DerivedServingMaintenanceService(
        stores.source,
        stores.index,
        stores.relation,
        tenant_id=config.tenant_id,
        document_serving=document_serving,
        retention_manager=maintenance.retention_manager,
        readiness=readiness,
        domain_overlay=facade.domain_overlay,
        index_policy=facade.index_policy,
        serving_lock=serving_lock,
    )
    lifecycle = ContextLifecycleService(
        stores.source,
        maintenance.tombstone_service,
        tenant_id=config.tenant_id,
        retention_manager=maintenance.retention_manager,
        readiness=readiness,
        serving_lock=serving_lock,
    )
    overlay = MemoryDocumentContextOverlay(
        memory.document_store,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
    )
    return ContextRuntime(
        facade=facade,
        administration_service=administration,
        lifecycle_service=lifecycle,
        memory_document_overlay=overlay,
        tombstone_service=maintenance.tombstone_service,
        retention_manager=maintenance.retention_manager,
    )


__all__ = ["ContextMaintenance", "wire_context", "wire_context_maintenance"]
