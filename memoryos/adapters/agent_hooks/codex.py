"""Codex 适配器。"""

from __future__ import annotations

from typing import Any

from memoryos.adapters.agent_hooks.base import BaseAgentHookAdapter, HookResult
from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.contracts import CodexPayloadParser


class CodexHookAdapter(BaseAgentHookAdapter):
    @classmethod
    def from_env(cls) -> CodexHookAdapter:
        return cls(AgentHookConfig.from_env("codex"))

    def handle(self, hook_name: str, payload: dict[str, Any] | None) -> HookResult:
        event = CodexPayloadParser().parse(payload, hook_name=hook_name, user_id=self.config.user_id)
        hook_name = event.hook_name
        if hook_name == "SessionStart":
            result = self.assemble_context(event)
            result.session_id = event.session_id
            return result
        if hook_name == "UserPromptSubmit":
            return self.assemble_context(event)
        if hook_name == "PostToolUse":
            queued = self.enqueue_commit(event)
            if self.config.flush_mode == "immediate":
                flushed = self.flush()
                queued.flushed = flushed.flushed
            return queued
        if hook_name == "Stop":
            committed = self.commit_now(event)
            if self.config.flush_mode in {"stop", "immediate"}:
                flushed = self.flush()
                committed.flushed = flushed.flushed
            return committed
        if hook_name == "PreCompact":
            committed = self.commit_now(event)
            assembled = self.assemble_context(event)
            committed.injection_text = assembled.injection_text
            committed.metadata.update(assembled.metadata)
            committed.error = committed.error or assembled.error
            if self.config.flush_mode in {"stop", "immediate"}:
                flushed = self.flush()
                committed.flushed = flushed.flushed
            return committed
        if hook_name == "SessionEnd":
            return self.commit_now(event)
        if hook_name == "SubagentStop":
            return self.enqueue_commit(event)
        if hook_name == "PostCompact":
            return self.checkpoint(event)
        return HookResult(ok=False, session_id=event.session_id, error={"code": "VALIDATION_ERROR", "message": f"unknown Codex hook: {hook_name}"})
