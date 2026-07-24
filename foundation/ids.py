"""核心工具里的标识。"""

from __future__ import annotations


def require_safe_path_segment(value: object, field_name: str) -> str:
    """返回无法逃逸目标父目录的安全标识。"""

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
