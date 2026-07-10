from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
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
from memoryos.memory.canonical.identity import AliasRegistry
from memoryos.memory.canonical.projection import CanonicalMemoryProjector, MemoryProjectionWorker
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.pipeline.executor import ActionExecutor
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.providers.rerank import Reranker
from memoryos.runtime.config import RuntimeConfig
from memoryos.skill.tool_registry import ToolRegistry


@dataclass
class RuntimeContainer:
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
    root_path = config.root_path
    source = source_store or FileSystemSourceStore(root_path)
    index = index_store or SQLiteIndexStore(root_path / "indexes" / "context.sqlite3")
    relation = relation_store or SQLiteRelationStore(root_path / "indexes" / "relations.sqlite3")
    queue = queue_store or SQLiteQueueStore(root_path / "queues" / "jobs.sqlite3")
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
    )
    session_archive_store = SessionArchiveStore(root_path)
    memory_projection_worker = MemoryProjectionWorker(
        CanonicalMemoryProjector(
            source,
            index,
            root_path,
            vector_store=configured_vector_store,
            embedding_provider=configured_embedding,
        ),
        queue,
    )
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
    )
    context_db = ContextDB(
        source,
        index,
        relation,
        queue_store=queue,
        session_commit_service=session_commit_service,
        committer=committer,
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
    )
