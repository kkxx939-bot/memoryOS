"""供应商错误分类和有界重试计时。"""

from __future__ import annotations

import random
from collections.abc import Callable

from LLMClient.contracts import (
    ModelAuthenticationError,
    ModelClientError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTransportError,
)

_INPUT_TOO_LARGE = (
    "payload too large",
    "request entity too large",
    "context window exceeded",
    "context_length_exceeded",
    "maximum context length",
    "too many input tokens",
    "input length exceeds",
)
_CONTENT_SAFETY = (
    "content policy",
    "content_filter",
    "contentfilter",
    "moderation",
    "sensitive content",
    "内容安全",
    "敏感内容",
)
_QUOTA = (
    "quota exceeded",
    "quota_exceeded",
    "insufficient_quota",
    "account quota",
    "accountoverdue",
    "billing hard limit",
)
_RATE_LIMIT = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "toomanyrequests",
    "tpm limit",
    "rpm limit",
)
_TRANSIENT = (
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
)


def normalize_provider_error(error: Exception) -> ModelClientError:
    """把 SDK 专用异常和传输异常映射到稳定的客户端异常体系。"""

    if isinstance(error, ModelClientError):
        return error
    status = _status_code(error)
    retry_after = _retry_after(error)
    message = str(error).strip() or type(error).__name__
    lowered = message.casefold()

    if _contains(lowered, _CONTENT_SAFETY):
        return ModelContentSafetyError(message)
    if status == 413 or _contains(lowered, _INPUT_TOO_LARGE):
        return ModelInputTooLargeError(message)
    if _contains(lowered, _QUOTA):
        return ModelQuotaError(message)
    if status == 401:
        return ModelAuthenticationError(message)
    if status == 403:
        return ModelPermissionError(message)
    if status == 429 or _contains(lowered, _RATE_LIMIT):
        return ModelRateLimitError(message, retry_after_seconds=retry_after)
    if status in {408, 409, 425, 500, 502, 503, 504}:
        return ModelTransportError(message, retry_after_seconds=retry_after)
    if isinstance(error, (TimeoutError, ConnectionError, OSError)) or _contains(lowered, _TRANSIENT):
        return ModelTransportError(message, retry_after_seconds=retry_after)
    return ModelResponseError(message)


def retry_delay(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    error: ModelClientError,
    uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """返回带随机抖动的指数退避时间，并遵守有界的 Retry-After 值。"""

    if error.retry_after_seconds is not None:
        return min(max(0.0, error.retry_after_seconds), max_delay)
    delay = min(base_delay * (2**attempt), max_delay)
    return min(max_delay, delay * uniform(0.8, 1.2))


def _contains(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _status_code(error: Exception) -> int | None:
    for candidate in (error, getattr(error, "__cause__", None)):
        if candidate is None:
            continue
        value = getattr(candidate, "status_code", None)
        if value is None:
            value = getattr(candidate, "code", None)
        if isinstance(value, bool) or not isinstance(value, str | int):
            continue
        try:
            status = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= status <= 599:
            return status
    return None


def _retry_after(error: Exception) -> float | None:
    for candidate in (error, getattr(error, "__cause__", None)):
        if candidate is None:
            continue
        direct = getattr(candidate, "retry_after", None)
        if direct is not None:
            try:
                return max(0.0, float(direct))
            except (TypeError, ValueError):
                pass
        response = getattr(candidate, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        try:
            value = headers.get("Retry-After")
        except (AttributeError, TypeError):
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


__all__ = ["normalize_provider_error", "retry_delay"]
