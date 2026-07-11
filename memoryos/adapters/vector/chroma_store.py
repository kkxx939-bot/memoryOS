"""适配器里的Chroma存储。"""

from memoryos.contextdb.store.vector_store import InMemoryVectorStore

ChromaStore = InMemoryVectorStore

__all__ = ["ChromaStore"]
