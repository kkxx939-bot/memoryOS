"""核心工具里的标识。"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def require_safe_path_segment(value: object, field_name: str) -> str:
    """Return one identifier that cannot escape its intended parent path."""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be one safe non-empty path segment")
    return value


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_hash(payload: Any, length: int = 24) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
