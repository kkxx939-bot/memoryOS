"""Deterministic embedding math shared by local fallback implementations."""

from __future__ import annotations


def hash_embedding(text: str, dimension: int) -> list[float]:
    """Return the stable character-bucket vector used by the local hash model."""

    buckets = [0.0] * int(dimension)
    for index, char in enumerate(str(text)):
        buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
    norm = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [round(value / norm, 6) for value in buckets]


__all__ = ["hash_embedding"]
