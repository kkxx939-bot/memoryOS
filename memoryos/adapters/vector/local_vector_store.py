"""适配器里的本地向量存储。"""

from memoryos.contextdb.store.vector_store import InMemoryVectorStore

LocalVectorStore = InMemoryVectorStore

__all__ = ["LocalVectorStore"]
