"""Agent-hook transport protocol and explicit composition registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from memoryos.adapters.agent_hooks.config import AgentHookConfig


class AgentHookTransport(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AgentHookCompositionError(RuntimeError):
    """Raised when a delivery entrypoint did not register its transport."""


AgentHookTransportFactory = Callable[[AgentHookConfig], AgentHookTransport]
_transport_factory: AgentHookTransportFactory | None = None


def register_agent_hook_transport_factory(factory: AgentHookTransportFactory) -> None:
    global _transport_factory
    _transport_factory = factory


def build_agent_hook_transport(config: AgentHookConfig) -> AgentHookTransport:
    if _transport_factory is None:
        raise AgentHookCompositionError(
            "agent hook transport is not configured; use the memoryos-agent-hook delivery entrypoint"
        )
    return _transport_factory(config)


__all__ = [
    "AgentHookCompositionError",
    "AgentHookTransport",
    "AgentHookTransportFactory",
    "build_agent_hook_transport",
    "register_agent_hook_transport_factory",
]
