"""把 Cursor Hook 载荷映射为统一的上下文召回与会话提交动作。"""

from __future__ import annotations

from typing import Any

from agent_hook.base import BaseAgentHookAdapter, HookResult
from agent_hook.composition import build_agent_hook_transport
from agent_hook.config import AgentHookConfig
from agent_hook.events import AgentHookEvent


class CursorHookAdapter(BaseAgentHookAdapter):
    @classmethod
    def from_env(cls) -> CursorHookAdapter:
        config = AgentHookConfig.from_env("cursor")
        return cls(config, mcp_client=build_agent_hook_transport(config))

    def handle(self, hook_name: str, payload: dict[str, Any] | None) -> HookResult:
        event = AgentHookEvent.from_payload(
            payload,
            adapter_id="cursor",
            hook_name=hook_name,
            agent_name=self.config.agent_name,
            user_id=self.config.user_id,
        )
        if hook_name == "before_prompt":
            return self.assemble_context(event)
        if hook_name == "after_turn":
            committed = self.commit_now(event)
            if self.config.flush_mode in {"stop", "immediate"}:
                flushed = self.flush()
                committed.flushed = flushed.flushed
            return committed
        if hook_name == "flush":
            return self.flush()
        return HookResult(ok=False, session_id=event.session_id, error={"code": "VALIDATION_ERROR", "message": f"unknown Cursor hook: {hook_name}"})
