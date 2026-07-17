"""基于哈希的本地向量实现。"""

from __future__ import annotations

from memoryos.core.embedding import hash_embedding


class HashingEmbeddingProvider:
    model_name = "hashing-v1"

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = int(dimension)

    def embed(self, text: str) -> list[float]:
        return hash_embedding(text, self.dimension)
