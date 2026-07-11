"""适配器里的Milvus存储。"""

from memoryos.contextdb.store.vector_store import InMemoryVectorStore

MilvusStore = InMemoryVectorStore

__all__ = ["MilvusStore"]
