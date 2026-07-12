"""运行时里的依赖组装。"""

from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.commit_group import CommitGroupStore
from memoryos.contextdb.session.planners import ActionPolicyCommitPlanner, BehaviorCommitPlanner, MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.store import FileSystemSourceStore, IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.source_store import LockStore, QueueStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_lock_store import SQLiteLockStore
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.memory.canonical.identity import AliasRegistry
from memoryos.memory.canonical.projection import CanonicalMemoryProjector, MemoryProjectionWorker
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.pipeline.executor import ActionExecutor
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.providers.rerank import Reranker
from memoryos.runtime.config import RuntimeConfig
from memoryos.skill.tool_registry import ToolRegistry
from memoryos.workers.recovery_worker import RecoveryWorker


@dataclass
class RuntimeContainer:
    """把 SDK、接口和后台任务共用的运行组件放在一起。"""

    source_store: SourceStore
    index_store: IndexStore
    relation_store: RelationStore
    queue_store: QueueStore
    lock_store: LockStore
    vector_store: VectorStore | None
    embedding_provider: EmbeddingProvider | None
    hybrid_search: HybridSearch | None
    reranker: Reranker | None
    committer: OperationCommitter
    session_archive_store: SessionArchiveStore
    session_commit_service: SessionCommitService
    context_db: ContextDB
    engine: PredictionEngine
    executor: ActionExecutor
    memory_projection_worker: MemoryProjectionWorker
    recovery_service: RecoveryService
    recovery_worker: RecoveryWorker


def build_runtime_container(
    config: RuntimeConfig,
    *,
    index_store: IndexStore | None = None,
    source_store: SourceStore | None = None,
    relation_store: RelationStore | None = None,
    queue_store: QueueStore | None = None,
    lock_store: LockStore | None = None,
    tool_registry: ToolRegistry | None = None,
    vector_store: VectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    hybrid_search: HybridSearch | None = None,
) -> RuntimeContainer:
    """组装默认运行链路，并拒绝会直接生成数据库操作的旧提取器。"""

    if config.memory_extractor is not None and (
        not getattr(config.memory_extractor, "semantic_proposal_backend", False)
        or not getattr(config.memory_extractor, "llm_semantic_backend", False)
    ):
        raise TypeError("memory_extractor must be an LLM MemorySemanticProposal backend")
    root_path = config.root_path
    root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_path.chmod(0o700)
    source = source_store or FileSystemSourceStore(root_path, tenant_id=config.tenant_id)
    source_tenant = getattr(source, "tenant_id", config.tenant_id)
    if str(source_tenant) != config.tenant_id:
        raise ValueError("SourceStore tenant does not match RuntimeConfig tenant_id")
    tenant_root = root_path if config.tenant_id == "default" else root_path / "tenants" / config.tenant_id
    tenant_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    tenant_root.chmod(0o700)
    index_root = tenant_root / "indexes"
    index = index_store or SQLiteIndexStore(index_root / "context.sqlite3")
    relation = relation_store or SQLiteRelationStore(index_root / "relations.sqlite3")
    queue = queue_store or SQLiteQueueStore(tenant_root / "queues" / "jobs.sqlite3")
    lock = lock_store or SQLiteLockStore(root_path / "system" / "locks.sqlite3")
    configured_embedding = embedding_provider or config.embedding
    configured_vector_store = vector_store or config.vector_store
    search = hybrid_search or (
        HybridSearch(
            index, vector_store=configured_vector_store, embedding_provider=configured_embedding, source_store=source
        )
        if configured_vector_store is not None and configured_embedding is not None
        else None
    )
    committer = OperationCommitter(
        source,
        index,
        config.root,
        lock_store=lock,
        relation_store=relation,
        queue_store=queue,
        tenant_id=config.tenant_id,
    )
    session_archive_store = SessionArchiveStore(root_path, tenant_id=config.tenant_id)
    projection_store = ProjectionRecordStore(tenant_root)
    memory_projection_worker = MemoryProjectionWorker(
        CanonicalMemoryProjector(
            source,
            index,
            tenant_root,
            relation_store=relation,
            vector_store=configured_vector_store,
            embedding_provider=configured_embedding,
            record_store=projection_store,
        ),
        queue,
    )
    recovery_service = RecoveryService(committer.redo, committer)
    recovery_worker = RecoveryWorker(recovery_service)
    session_commit_service = SessionCommitService(
        session_archive_store,
        queue,
        committer=committer,
        memory_planner=MemoryCommitPlanner(
            source_store=source,
            index_store=index,
            relation_store=relation,
            hybrid_search=search,
            extractor=config.memory_extractor,
            alias_registry=AliasRegistry(config.memory_aliases),
        ),
        behavior_planner=BehaviorCommitPlanner(index_store=index, source_store=source),
        action_policy_planner=ActionPolicyCommitPlanner(index_store=index, source_store=source),
        projection_worker=memory_projection_worker,
        commit_group_store=CommitGroupStore(tenant_root),
    )
    context_db = ContextDB(
        source,
        index,
        relation,
        queue_store=queue,
        session_commit_service=session_commit_service,
        committer=committer,
        projection_store=projection_store,
    )
    engine = PredictionEngine(
        index,
        PredictionLedger(config.root),
        source_store=source,
        relation_store=relation,
        vector_store=configured_vector_store,
        embedding_provider=configured_embedding,
        hybrid_search=search,
    )
    return RuntimeContainer(
        source_store=source,
        index_store=index,
        relation_store=relation,
        queue_store=queue,
        lock_store=lock,
        vector_store=configured_vector_store,
        embedding_provider=configured_embedding,
        hybrid_search=search,
        reranker=config.reranker,
        committer=committer,
        session_archive_store=session_archive_store,
        session_commit_service=session_commit_service,
        context_db=context_db,
        engine=engine,
        executor=ActionExecutor(tool_registry),
        memory_projection_worker=memory_projection_worker,
        recovery_service=recovery_service,
        recovery_worker=recovery_worker,
    )
