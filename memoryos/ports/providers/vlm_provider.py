from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memoryos.ports.providers.chat_provider import ModelResponse


@dataclass(frozen=True)
class VLMFrame:
    uri: str
    mime_type: str = "image/jpeg"
    timestamp_ms: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VLMRequest:
    prompt: str
    frames: list[VLMFrame]
    model: str | None = None
    prompt_version: str | None = None
    metadata: dict = field(default_factory=dict)


class VLMProvider(Protocol):
    provider_name: str
    model: str

    def analyze(self, request: VLMRequest) -> ModelResponse: ...

    def health_check(self) -> dict: ...
