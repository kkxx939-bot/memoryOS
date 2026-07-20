"""ActionPolicy 在线决策和动作执行对象的装配。"""

from __future__ import annotations

from infrastructure.store.action_policy import ActionPolicyDecisionLedger
from policy.action_policy.decision.engine import PredictionEngine
from policy.action_policy.execution.executor import ActionExecutor
from policy.action_policy.execution.tool_registry import ToolRegistry
from runtime.config import RuntimeConfig
from runtime.container import PolicyRuntime, StoreRuntime
from runtime.dependencies import RuntimeDependencies


def wire_policy(
    stores: StoreRuntime,
    config: RuntimeConfig,
    dependencies: RuntimeDependencies,
) -> PolicyRuntime:
    engine = PredictionEngine(
        stores.index,
        ActionPolicyDecisionLedger(config.root_path),
        source_store=stores.source,
        relation_store=stores.relation,
        vector_store=stores.vector,
        embedding_provider=stores.embedding,
        hybrid_search=stores.hybrid_search,
    )
    return PolicyRuntime(
        engine=engine,
        executor=ActionExecutor(dependencies.tool_registry or ToolRegistry()),
    )


__all__ = ["wire_policy"]
