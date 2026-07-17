"""Embedding capability consumed by ContextDB retrieval."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Structural contract required by vector-backed retrieval."""

    model_name: str
    dimension: int

    def embed(self, text: str) -> list[float]: ...


__all__ = ["EmbeddingProvider"]
