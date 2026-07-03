from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    provider: str
    dimension: int
    content_hash: str
    token_count: int | None = None
    latency_ms: int | None = None
    normalized: bool = False
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "vector": self.vector,
            "model": self.model,
            "provider": self.provider,
            "dimension": self.dimension,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "normalized": self.normalized,
            "usage": self.usage,
        }


class EmbeddingProvider(Protocol):
    provider_name: str
    model: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for text."""
        ...

    def embed_text(self, text: str) -> EmbeddingResult: ...

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]: ...

    def health_check(self) -> dict: ...


class HashingEmbeddingProvider:
    """Deterministic local embedding for tests and offline development.

    This is not a semantic model. It only gives the indexing pipeline a stable
    vector interface until a real embedding provider is attached.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self.dimension = dimensions
        self.model = f"hashing-{dimensions}"
        self.provider_name = "local_hashing"

    def embed(self, text: str) -> list[float]:
        return self.embed_text(text).vector

    def embed_text(self, text: str) -> EmbeddingResult:
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        normalized = normalize(vector)
        return EmbeddingResult(
            vector=normalized,
            model=self.model,
            provider=self.provider_name,
            dimension=self.dimension,
            content_hash=content_hash(text),
            token_count=len(tokens),
            normalized=True,
        )

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]

    def health_check(self) -> dict:
        return {
            "status": "ok",
            "provider": self.provider_name,
            "model": self.model,
            "dimension": self.dimension,
        }

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
        return tokens


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
