"""Agent 输入清理。"""

from __future__ import annotations

import re
from typing import Any

SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|password|secret|authorization)")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)
BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+")
ENV_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|PASSWORD|SECRET)[A-Z0-9_]*)(\s*=\s*)([^\s]+)"
)
INLINE_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|password|secret)(\s*[:=]\s*)([^\s,;]+)")
LOCAL_PATH_RE = re.compile(r"(?:(?:/Users|/home|/tmp|/private/tmp)/[^\s'\",;:)]*)")
NOISY_PATH_PARTS = {".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__"}
SENSITIVE_FILE_NAMES = {".env", ".gitignore", ".memoryosignore", "id_rsa", "id_ed25519", "credentials", "credentials.json"}
MAX_TEXT = 4000
MAX_LOG_LINES = 80


def sanitize_payload(value: Any, *, max_text: int = MAX_TEXT) -> Any:
    if isinstance(value, bytes):
        return "<binary>"
    if isinstance(value, str):
        return sanitize_text(value, max_text=max_text)
    if isinstance(value, list):
        return [sanitize_payload(item, max_text=max_text) for item in value if not _is_noisy_path(item)]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_RE.search(key_text):
                sanitized[key_text] = "<redacted>"
            elif _is_noisy_path(item):
                continue
            else:
                sanitized[key_text] = sanitize_payload(item, max_text=max_text)
        return sanitized
    return value


def sanitize_text(text: str, *, max_text: int = MAX_TEXT) -> str:
    if "\x00" in text:
        return "<binary>"
    redacted = PRIVATE_KEY_RE.sub("<redacted-private-key>", text)
    redacted = BEARER_RE.sub(r"\1<redacted>", redacted)
    redacted = ENV_SECRET_RE.sub(r"\1\2<redacted>", redacted)
    redacted = INLINE_SECRET_RE.sub(r"\1\2<redacted>", redacted)
    lines = redacted.splitlines()
    if len(lines) > MAX_LOG_LINES:
        head = lines[:40]
        tail = lines[-20:]
        redacted = "\n".join([*head, f"... <{len(lines) - 60} lines omitted> ...", *tail])
    if len(redacted) > max_text:
        return redacted[:max_text] + f"\n... <{len(redacted) - max_text} chars omitted> ..."
    return redacted


def sanitize_error_text(text: str, *, max_text: int = 300) -> str:
    redacted = sanitize_text(text, max_text=max_text)
    redacted = LOCAL_PATH_RE.sub("<redacted-path>", redacted)
    return redacted[:max_text]


def sanitize_changed_files(paths: list[str]) -> list[str]:
    return [path for path in (str(item) for item in paths) if not _is_noisy_path(path)]


def summarize_tool_result(tool_name: str | None, tool_input: dict[str, Any] | None, tool_output: Any, changed_files: list[str]) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "tool_input": sanitize_payload(tool_input or {}, max_text=1200),
        "tool_output": sanitize_payload(tool_output, max_text=2000),
        "changed_files": sanitize_changed_files(changed_files),
    }


def _is_noisy_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = set(value.replace("\\", "/").split("/"))
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(parts & NOISY_PATH_PARTS) or name in SENSITIVE_FILE_NAMES or name.startswith(".env.")
