"""Context 查询、维护和派生清理对象的装配。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import cast

from infrastructure.context.facade import ContextDB
from infrastructure.context.maintenance import (
    ContextLifecycleService,
    DerivedServingMaintenanceService,
)
from infrastructure.context.maintenance.retention import CatalogRetentionManager, RetentionPolicy
from infrastructure.context.maintenance.retention_policy import RetentionCatalogStore
from infrastructure.context.maintenance.tombstone import ProjectionTombstoneService
from runtime.config import RuntimeConfig
from runtime.container import ContextRuntime, StoreRuntime


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
    maintenance: ContextMaintenance,
) -> ContextRuntime:
    """连接 Context 门面、管理服务和派生层维护。"""

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
    administration = DerivedServingMaintenanceService(
        stores.source,
        stores.index,
        stores.relation,
        tenant_id=config.tenant_id,
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
    return ContextRuntime(
        facade=facade,
        administration_service=administration,
        lifecycle_service=lifecycle,
        tombstone_service=maintenance.tombstone_service,
        retention_manager=maintenance.retention_manager,
    )


__all__ = ["ContextMaintenance", "wire_context", "wire_context_maintenance"]
