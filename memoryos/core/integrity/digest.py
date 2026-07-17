"""Stable digests over canonical JSON bytes."""

from __future__ import annotations

import hashlib
from typing import Any

from memoryos.core.integrity.canonical_json import canonical_json


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    """Return the SHA-256 digest of exact UTF-8 text bytes."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["canonical_digest", "text_digest"]
