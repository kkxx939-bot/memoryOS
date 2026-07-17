"""Compatibility exports for the historical vector store module."""

from memoryos.adapters.vector.in_memory import InMemoryVectorStore
from memoryos.contextdb.store.vector import (
    VectorCapabilities,
    VectorHit,
    VectorStore,
    require_production_vector_capabilities,
    vector_capabilities,
    vector_row_id,
)

__all__ = [
    "InMemoryVectorStore",
    "VectorCapabilities",
    "VectorHit",
    "VectorStore",
    "require_production_vector_capabilities",
    "vector_capabilities",
    "vector_row_id",
]
