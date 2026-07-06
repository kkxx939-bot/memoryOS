from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MCPErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CLIENT_ERROR = "CLIENT_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class MCPToolError:
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": _safe_message(self.message),
            "retryable": self.retryable,
            "details": _safe_details(self.details),
        }


class ToolValidationError(ValueError):
    pass


class ToolPermissionError(PermissionError):
    pass


def error_payload(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"error": MCPToolError(code, message, retryable=retryable, details=details or {}).to_dict()}


def ok_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["error"] = None
    return result


def exception_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ToolValidationError | ValueError):
        return error_payload(MCPErrorCode.VALIDATION_ERROR, str(exc), retryable=False)
    if isinstance(exc, ToolPermissionError | PermissionError):
        return error_payload(MCPErrorCode.PERMISSION_DENIED, str(exc), retryable=False)
    if isinstance(exc, FileNotFoundError | OSError):
        return error_payload(MCPErrorCode.STORAGE_ERROR, exc.__class__.__name__, retryable=True)
    return error_payload(MCPErrorCode.INTERNAL_ERROR, exc.__class__.__name__, retryable=True)


def _safe_message(message: str) -> str:
    sanitized = str(message)
    home_markers = ["/Users/", "/private/", "/var/", "/tmp/"]
    for marker in home_markers:
        if marker in sanitized:
            sanitized = sanitized.split(marker, 1)[0] + "<redacted-path>"
            break
    return sanitized[:500]


def _safe_details(details: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in details.items():
        key_text = str(key)
        if any(token in key_text.lower() for token in ("token", "secret", "password", "key")):
            safe[key_text] = "<redacted>"
        elif isinstance(value, str):
            safe[key_text] = _safe_message(value)
        else:
            safe[key_text] = value
    return safe
