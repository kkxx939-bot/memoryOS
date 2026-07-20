"""语义记忆提取的有限失败类型。"""

from __future__ import annotations


class MemoryExtractionError(RuntimeError):
    retryable = False
    terminal = True
    code = "MEMORY_EXTRACTION_ERROR"


class MemoryExtractionTransportError(MemoryExtractionError):
    retryable = True
    terminal = False
    code = "MEMORY_EXTRACTION_TRANSPORT"


class MemoryExtractionTimeoutError(MemoryExtractionTransportError):
    code = "MEMORY_EXTRACTION_TIMEOUT"


class MemoryExtractionRateLimitError(MemoryExtractionTransportError):
    code = "MEMORY_EXTRACTION_RATE_LIMIT"


class MemoryExtractionMalformedEnvelopeError(MemoryExtractionError, ValueError):
    retryable = True
    terminal = False
    code = "MEMORY_EXTRACTION_MALFORMED_ENVELOPE"


class MemoryExtractionCandidateValidationError(MemoryExtractionError, ValueError):
    code = "MEMORY_EXTRACTION_CANDIDATE_INVALID"


class MemoryExtractionSecurityError(MemoryExtractionError):
    code = "MEMORY_EXTRACTION_SECURITY"


class MemoryExtractionConfigurationError(MemoryExtractionError):
    code = "MEMORY_EXTRACTION_CONFIGURATION"


def classify_memory_extraction_failure(exc: BaseException) -> MemoryExtractionError:
    """把 Provider 异常映射为有限、可判断是否重试的生产契约。"""

    if isinstance(exc, MemoryExtractionError):
        return exc
    name = type(exc).__name__.replace("_", "").casefold()
    message = str(exc) or type(exc).__name__
    status = getattr(exc, "status_code", getattr(exc, "status", None))
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code == 429 or "ratelimit" in name:
        return MemoryExtractionRateLimitError(message)
    if isinstance(exc, TimeoutError) or status_code in {408, 504} or "timeout" in name:
        return MemoryExtractionTimeoutError(message)
    retryable_status = status_code in {425, 500, 502, 503}
    retryable_name = any(
        token in name for token in ("connection", "transport", "network", "unavailable", "serviceunavailable")
    )
    retryable_message = isinstance(exc, RuntimeError) and any(
        token in message.casefold() for token in ("unavailable", "temporarily", "transport", "connection", "network")
    )
    if isinstance(exc, ConnectionError | OSError) or retryable_status or retryable_name or retryable_message:
        return MemoryExtractionTransportError(message)
    return MemoryExtractionConfigurationError(
        f"memory model provider failed with unsupported error: {type(exc).__name__}"
    )
