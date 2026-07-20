"""基于规范 JSON 字节生成稳定摘要。"""

from __future__ import annotations

import hashlib
from typing import Any

from foundation.integrity.canonical_json import canonical_json


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    """返回精确 UTF-8 文本字节的 SHA-256 摘要。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["canonical_digest", "text_digest"]
