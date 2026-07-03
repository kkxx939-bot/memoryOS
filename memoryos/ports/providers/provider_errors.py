from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderErrorInfo:
    provider: str
    error_type: str
    message: str
    retryable: bool = False
    status_code: int | None = None
    raw: dict | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "raw": self.raw,
            "metadata": self.metadata,
        }


class ProviderError(RuntimeError):
    error_type = "provider_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        raw: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.info = ProviderErrorInfo(
            provider=provider,
            error_type=self.error_type,
            message=message,
            retryable=self.retryable,
            status_code=status_code,
            raw=raw,
            metadata=metadata or {},
        )


class ProviderTimeout(ProviderError):
    error_type = "timeout"
    retryable = True


class ProviderRateLimited(ProviderError):
    error_type = "rate_limited"
    retryable = True


class ProviderUnavailable(ProviderError):
    error_type = "unavailable"
    retryable = True


class ProviderBadResponse(ProviderError):
    error_type = "bad_response"


class ProviderAuthFailed(ProviderError):
    error_type = "auth_failed"


class ProviderQuotaExceeded(ProviderError):
    error_type = "quota_exceeded"


class ProviderValidationError(ProviderError):
    error_type = "validation_error"
