"""Canonical projection's deterministic compatibility fallback."""

from __future__ import annotations

from memoryos.core.embedding import hash_embedding


class DeterministicProjectionEmbedding:
    """Preserve historical projection vectors when no provider is injected."""

    model_name = "hashing-v1"

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = int(dimension)

    def embed(self, text: str) -> list[float]:
        return hash_embedding(text, self.dimension)


__all__ = ["DeterministicProjectionEmbedding"]
