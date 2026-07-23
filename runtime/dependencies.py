"""由宿主或测试显式注入的运行实例依赖。"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.context.reranking import Reranker
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.queue import QueueStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.contracts.vector import VectorStore
from LLMClient import LLMClient
from policy.action_policy.execution.tool_registry import ToolRegistry


@dataclass(frozen=True)
class RuntimeDependencies:
    """保存已创建的实现对象，避免把实例塞进配置对象。"""

    index_store: IndexStore | None = None
    source_store: SourceStore | None = None
    relation_store: RelationStore | None = None
    queue_store: QueueStore | None = None
    lock_store: LockStore | None = None
    tool_registry: ToolRegistry | None = None
    vector_store: VectorStore | None = None
    embedding_provider: EmbeddingProvider | None = None
    hybrid_search: HybridSearch | None = None
    reranker: Reranker | None = None
    model_client: LLMClient | None = None


__all__ = ["RuntimeDependencies"]
