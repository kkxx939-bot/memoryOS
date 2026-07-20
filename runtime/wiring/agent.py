"""Agent Hook 会话服务的装配。"""

from __future__ import annotations

from agent_hook.session_service import AgentSessionService
from runtime.config import RuntimeConfig
from runtime.container import AgentRuntime


def wire_agent(config: RuntimeConfig) -> AgentRuntime:
    return AgentRuntime(
        session_service=AgentSessionService(str(config.root_path)),
    )


__all__ = ["wire_agent"]
