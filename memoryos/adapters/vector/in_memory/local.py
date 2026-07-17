"""Historical local-vector name with explicit in-memory ownership."""

from memoryos.adapters.vector.in_memory.store import InMemoryVectorStore


class LocalVectorStore(InMemoryVectorStore):
    """Local process vector store backed by the in-memory implementation."""


__all__ = ["LocalVectorStore"]
