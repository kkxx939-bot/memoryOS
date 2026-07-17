"""Built-in no-op embedding provider and historical protocol export."""

from __future__ import annotations

from memoryos.contextdb.retrieval.embedding import EmbeddingProvider


class NoopEmbeddingProvider:
    model_name = "noop"
    dimension = 0

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return []


__all__ = ["EmbeddingProvider", "NoopEmbeddingProvider"]
