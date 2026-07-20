"""测试专用的确定性向量替身，不参与生产运行。"""

from __future__ import annotations


class DeterministicEmbeddingProvider:
    """用稳定字符桶生成向量，让召回测试不依赖外部模型。"""

    model_name = "test-deterministic"

    def __init__(self, dimension: int = 16) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimension
        for index, char in enumerate(str(text)):
            buckets[index % self.dimension] += (ord(char) % 31) / 31.0
        norm = sum(value * value for value in buckets) ** 0.5 or 1.0
        return [round(value / norm, 6) for value in buckets]


__all__ = ["DeterministicEmbeddingProvider"]
