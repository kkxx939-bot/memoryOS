"""Agent Hook 传输协议和显式装配注册。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from agent_hook.config import AgentHookConfig


class AgentHookTransport(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AgentHookCompositionError(RuntimeError):
    """交付入口未注册传输实现时抛出。"""


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
