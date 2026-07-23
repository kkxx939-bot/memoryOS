"""按领域分组的运行时对象。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_hook.session_service import AgentSessionService
from foundation.readiness import RuntimeReadiness
from infrastructure.context.facade import ContextDB
from infrastructure.context.maintenance import ContextAdministrationService, ContextLifecycleService
from infrastructure.context.maintenance.retention import CatalogRetentionManager
from infrastructure.context.maintenance.tombstone import ProjectionTombstoneService
from infrastructure.context.reranking import Reranker
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.queue import QueueStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.contracts.vector import VectorStore
from infrastructure.store.filesystem.session_archive import SessionArchiveStore
from infrastructure.store.runtime_layout import RuntimeLayout
from LLMClient import LLMClient
from policy.action_policy.decision.engine import PredictionEngine
from policy.action_policy.execution.executor import ActionExecutor
from runtime.config import RuntimeConfig
from runtime.recovery.transaction_worker import RecoveryWorker
from runtime.session.commit_service import SessionCommitService
from transaction.commit.operation_committer import OperationCommitter
from transaction.commit.recovery import RecoveryService

if TYPE_CHECKING:
    from runtime.lifecycle import RuntimeLifecycle
    from runtime.recovery.report import RecoveryReport


@dataclass(frozen=True)
class StoreRuntime:
    """进程内共享的持久化、模型和检索基础对象。"""

    source: SourceStore
    index: IndexStore
    relation: RelationStore
    queue: QueueStore
    lock: LockStore
    vector: VectorStore | None
    embedding: EmbeddingProvider | None
    hybrid_search: HybridSearch | None
    reranker: Reranker | None
    model_client: LLMClient | None


@dataclass(frozen=True)
class TransactionRuntime:
    """普通 Context 事务提交和恢复对象。"""

    committer: OperationCommitter
    recovery_service: RecoveryService
    recovery_worker: RecoveryWorker


@dataclass(frozen=True)
class SessionRuntime:
    """会话归档和普通派生提交对象。"""

    archive_store: SessionArchiveStore
    commit_service: SessionCommitService


@dataclass(frozen=True)
class ContextRuntime:
    """统一 Context 的查询、维护和派生层清理对象。"""

    facade: ContextDB
    administration_service: ContextAdministrationService
    lifecycle_service: ContextLifecycleService
    tombstone_service: ProjectionTombstoneService
    retention_manager: CatalogRetentionManager


@dataclass(frozen=True)
class PolicyRuntime:
    """ActionPolicy 的在线决策与动作执行对象。"""

    engine: PredictionEngine
    executor: ActionExecutor


@dataclass(frozen=True)
class AgentRuntime:
    """Coding Agent 会话接入对象。"""

    session_service: AgentSessionService


@dataclass
class RuntimeContainer:
    """整个进程唯一的、按领域分组的运行时实例。"""

    config: RuntimeConfig
    layout: RuntimeLayout
    readiness: RuntimeReadiness
    stores: StoreRuntime
    transaction: TransactionRuntime
    session: SessionRuntime
    context: ContextRuntime
    policy: PolicyRuntime
    agent: AgentRuntime
    lifecycle: RuntimeLifecycle

    def start(self) -> RecoveryReport:
        """执行显式启动恢复，在成功后发布 READY。"""

        return self.lifecycle.start(self)

    def stop(self) -> None:
        """停止运行时并关闭生命周期入口。"""

        self.lifecycle.stop(self)


__all__ = [
    "AgentRuntime",
    "ContextRuntime",
    "PolicyRuntime",
    "RuntimeContainer",
    "SessionRuntime",
    "StoreRuntime",
    "TransactionRuntime",
]
