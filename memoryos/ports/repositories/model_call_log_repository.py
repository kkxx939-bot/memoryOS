from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ModelCallLog:
    call_id: str
    user_id: str
    provider: str
    model: str
    operation: str
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_estimate: float = 0.0
    status: str = "ok"
    error_type: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "user_id": self.user_id,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "prompt_version": self.prompt_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_estimate": self.cost_estimate,
            "status": self.status,
            "error_type": self.error_type,
            "metadata": self.metadata,
        }


class ModelCallLogRepository(Protocol):
    def append(self, log: ModelCallLog) -> dict: ...

    def usage_summary(self, user_id: str, day: str | None = None) -> dict: ...
