"""统一 Context 事务内核的装配。"""

from __future__ import annotations

from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from infrastructure.context.operation_target import ContextOperationTargetResolver
from infrastructure.store.operation import build_operation_control_stores
from policy.action_policy.integration.commit_registration import build_action_policy_transaction_extensions
from runtime.config import RuntimeConfig
from runtime.container import StoreRuntime, TransactionRuntime
from runtime.recovery.transaction_worker import RecoveryWorker
from transaction.commit.operation_committer import OperationCommitter
from transaction.commit.recovery import RecoveryService


def wire_transaction(
    stores: StoreRuntime,
    config: RuntimeConfig,
    *,
    tenant_root,  # noqa: ANN001
    tombstone_service,  # noqa: ANN001
) -> TransactionRuntime:
    """一次性构造完整提交器，禁止后续属性补注入。"""

    committer = OperationCommitter(
        stores.source,
        stores.index,  # pyright: ignore[reportArgumentType]
        str(config.root_path),
        build_operation_control_stores(tenant_root),
        lock_store=stores.lock,
        relation_store=stores.relation,
        target_resolver=ContextOperationTargetResolver(stores.index, stores.source),
        context_effects=InfrastructureContextOperationEffects(),
        domain_extensions=build_action_policy_transaction_extensions(),
        tenant_id=config.tenant_id,
        tombstone_service=tombstone_service,
    )
    recovery_service = RecoveryService(committer.redo, committer)
    return TransactionRuntime(
        committer=committer,
        recovery_service=recovery_service,
        recovery_worker=RecoveryWorker(recovery_service),
    )


__all__ = ["wire_transaction"]
