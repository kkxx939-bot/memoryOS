"""跨入口复用的数据脱敏与安全投影。"""

from sanitization.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
    SanitizedContextProjection,
)
from sanitization.text import (
    ENV_SECRET_RE,
    INLINE_SECRET_RE,
    PRIVATE_KEY_RE,
    SECRET_KEY_RE,
    sanitize_changed_files,
    sanitize_error_text,
    sanitize_payload,
    sanitize_text,
    summarize_tool_result,
)

__all__ = [
    "ContextProjectionSanitizationError",
    "ContextProjectionSanitizer",
    "ENV_SECRET_RE",
    "INLINE_SECRET_RE",
    "PRIVATE_KEY_RE",
    "SECRET_KEY_RE",
    "SanitizedContextProjection",
    "sanitize_changed_files",
    "sanitize_error_text",
    "sanitize_payload",
    "sanitize_text",
    "summarize_tool_result",
]
