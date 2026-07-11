"""基于哈希的本地向量实现。"""

from __future__ import annotations


class HashingEmbeddingProvider:
    model_name = "hashing-v1"

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = int(dimension)

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimension
        for index, char in enumerate(str(text)):
            buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
        norm = sum(value * value for value in buckets) ** 0.5 or 1.0
        return [round(value / norm, 6) for value in buckets]
