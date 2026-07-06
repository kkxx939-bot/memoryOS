from __future__ import annotations

from typing import Any

from memoryos.adapters.agent_hooks.base import BaseAgentHookAdapter, HookResult
from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.events import AgentHookEvent


class CodexHookAdapter(BaseAgentHookAdapter):
    @classmethod
    def from_env(cls) -> CodexHookAdapter:
        return cls(AgentHookConfig.from_env("codex"))

    def handle(self, hook_name: str, payload: dict[str, Any] | None) -> HookResult:
        event = AgentHookEvent.from_payload(
            payload,
            adapter_id="codex",
            hook_name=hook_name,
            agent_name=self.config.agent_name,
            user_id=self.config.user_id,
        )
        if hook_name == "SessionStart":
            result = self.assemble_context(event)
            result.session_id = event.session_id
            return result
        if hook_name == "UserPromptSubmit":
            return self.assemble_context(event)
        if hook_name == "PostToolUse":
            queued = self.enqueue_commit(event)
            flushed = self.flush()
            queued.flushed = flushed.flushed
            return queued
        if hook_name == "Stop":
            committed = self.commit_now(event)
            flushed = self.flush()
            committed.flushed = flushed.flushed
            return committed
        if hook_name == "PreCompact":
            assembled = self.assemble_context(event)
            committed = self.commit_now(event)
            assembled.committed = committed.committed
            assembled.queued = committed.queued
            assembled.error = assembled.error or committed.error
            return assembled
        return HookResult(ok=False, session_id=event.session_id, error={"code": "VALIDATION_ERROR", "message": f"unknown Codex hook: {hook_name}"})
