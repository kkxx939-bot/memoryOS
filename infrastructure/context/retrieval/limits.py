"""上下文召回入口统一使用的数量边界。"""

from __future__ import annotations

from typing import Any

MAX_RETRIEVAL_LIMIT = 100


def bounded_int(value: Any, *, default: int, minimum: int, maximum: int, label: str) -> int:
    try:
        resolved = int(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not minimum <= resolved <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return resolved


__all__ = ["MAX_RETRIEVAL_LIMIT", "bounded_int"]
