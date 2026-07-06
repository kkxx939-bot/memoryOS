from __future__ import annotations

from typing import Any

from memoryos.adapters.agent_hooks.base import BaseAgentHookAdapter, HookResult
from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.events import AgentHookEvent


class ClaudeCodeHookAdapter(BaseAgentHookAdapter):
    @classmethod
    def from_env(cls) -> ClaudeCodeHookAdapter:
        return cls(AgentHookConfig.from_env("claude_code"))

    def handle(self, hook_name: str, payload: dict[str, Any] | None) -> HookResult:
        event = AgentHookEvent.from_payload(
            payload,
            adapter_id="claude_code",
            hook_name=hook_name,
            agent_name=self.config.agent_name,
            user_id=self.config.user_id,
        )
        if hook_name == "before_prompt":
            return self.assemble_context(event)
        if hook_name == "after_turn":
            committed = self.commit_now(event)
            flushed = self.flush()
            committed.flushed = flushed.flushed
            return committed
        return HookResult(ok=False, session_id=event.session_id, error={"code": "VALIDATION_ERROR", "message": f"unknown Claude Code hook: {hook_name}"})
