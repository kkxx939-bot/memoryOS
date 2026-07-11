"""适配器里的Qdrant存储。"""

from memoryos.contextdb.store.vector_store import InMemoryVectorStore

QdrantStore = InMemoryVectorStore

__all__ = ["QdrantStore"]
