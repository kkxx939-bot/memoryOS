"""MemoryOS 唯一的运行时对象构建顺序。"""

from __future__ import annotations

from runtime.config import RuntimeConfig
from runtime.container import RuntimeContainer
from runtime.dependencies import RuntimeDependencies
from runtime.lifecycle import RuntimeLifecycle
from runtime.recovery.coordinator import RuntimeRecoveryCoordinator
from runtime.wiring.agent import wire_agent
from runtime.wiring.context import wire_context, wire_context_maintenance
from runtime.wiring.policy import wire_policy
from runtime.wiring.session import wire_session
from runtime.wiring.store import wire_stores
from runtime.wiring.transaction import wire_transaction


class RuntimeBuilder:
    """只创建和连接对象；调用者必须随后显式执行 ``runtime.start()``。"""

    def __init__(
        self,
        config: RuntimeConfig,
        dependencies: RuntimeDependencies | None = None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies or RuntimeDependencies()

    def build(self) -> RuntimeContainer:
        """按依赖方向创建运行时，但不执行恢复或发布 READY。"""

        base = wire_stores(self.config, self.dependencies)

        context_maintenance = wire_context_maintenance(base.stores, self.config)
        transaction = wire_transaction(
            base.stores,
            self.config,
            tenant_root=base.layout.tenant_root,
            tombstone_service=context_maintenance.tombstone_service,
        )
        session = wire_session(
            base.stores,
            transaction,
            self.config,
            tenant_root=base.layout.tenant_root,
        )
        context = wire_context(
            base.stores,
            self.config,
            readiness=base.readiness,
            committer=transaction.committer,
            maintenance=context_maintenance,
        )
        return RuntimeContainer(
            config=self.config,
            layout=base.layout,
            readiness=base.readiness,
            stores=base.stores,
            transaction=transaction,
            session=session,
            context=context,
            policy=wire_policy(base.stores, self.config, self.dependencies),
            agent=wire_agent(self.config),
            lifecycle=RuntimeLifecycle(RuntimeRecoveryCoordinator()),
        )


__all__ = ["RuntimeBuilder"]
