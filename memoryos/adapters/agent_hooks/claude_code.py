from __future__ import annotations

from typing import Any

from memoryos.adapters.agent_hooks.base import BaseAgentHookAdapter, HookResult
from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.contracts import ClaudeCodePayloadParser


class ClaudeCodeHookAdapter(BaseAgentHookAdapter):
    @classmethod
    def from_env(cls) -> ClaudeCodeHookAdapter:
        return cls(AgentHookConfig.from_env("claude_code"))

    def handle(self, hook_name: str, payload: dict[str, Any] | None) -> HookResult:
        data = {**dict(payload or {}), "hook_event_name": dict(payload or {}).get("hook_event_name", hook_name)}
        event = ClaudeCodePayloadParser().parse(data, user_id=self.config.user_id)
        hook_name = event.hook_name
        if hook_name in {"SessionStart", "UserPromptSubmit", "before_prompt"}:
            return self.assemble_context(event)
        if hook_name in {"PostToolUse", "SubagentStop"}:
            return self.enqueue_commit(event)
        if hook_name == "Stop":
            return self.checkpoint(event)
        if hook_name in {"SessionEnd", "after_turn"}:
            committed = self.commit_now(event)
            if self.config.flush_mode in {"stop", "immediate"}:
                flushed = self.flush()
                committed.flushed = flushed.flushed
            return committed
        if hook_name in {"PreCompact", "pre_compact"}:
            committed = self.commit_now(event)
            if self.config.flush_mode in {"stop", "immediate"}:
                flushed = self.flush()
                committed.flushed = flushed.flushed
            return committed
        return HookResult(ok=False, session_id=event.session_id, error={"code": "VALIDATION_ERROR", "message": f"unknown Claude Code hook: {hook_name}"})
