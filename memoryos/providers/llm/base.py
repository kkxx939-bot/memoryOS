from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.0
    prompt_version: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    provider: str
    prompt_version: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: int | None = None
    raw: dict | None = None


class ChatProvider(Protocol):
    provider_name: str
    model: str

    def complete(self, request: ChatRequest | str) -> ModelResponse | str: ...

    def health_check(self) -> dict: ...
