"""Agent 钩子的接口约定。"""

from __future__ import annotations

from typing import Any

from memoryos.adapters.agent_hooks.base import HookResult
from memoryos.adapters.agent_hooks.events import AgentHookEvent


class ClaudeCodePayloadParser:
    def parse(self, payload: dict[str, Any] | None, *, user_id: str | None = None) -> AgentHookEvent:
        data = dict(payload or {})
        event_name = str(data.get("hook_event_name") or data.get("event_name") or "")
        if not event_name:
            raise ValueError("Claude Code payload requires hook_event_name")
        if event_name == "SubagentStop" and data.get("last_assistant_message"):
            data["messages"] = [{"role": "assistant", "content": str(data["last_assistant_message"]), "agent_id": data.get("agent_id")}]
            data["transcript_path"] = data.get("agent_transcript_path") or data.get("transcript_path")
        return AgentHookEvent.from_payload(data, adapter_id="claude_code", hook_name=event_name, agent_name="claude_code", user_id=user_id)


class CodexPayloadParser:
    def parse(self, payload: dict[str, Any] | None, *, hook_name: str = "", user_id: str | None = None) -> AgentHookEvent:
        data = dict(payload or {})
        event_name = str(data.get("hook_event_name") or data.get("event_name") or hook_name)
        if not event_name:
            raise ValueError("Codex payload requires event_name")
        canonical = {
            "session_start": "SessionStart",
            "user_prompt_submit": "UserPromptSubmit",
            "post_tool_use": "PostToolUse",
            "pre_compact": "PreCompact",
            "post_compact": "PostCompact",
            "subagent_stop": "SubagentStop",
        }.get(event_name, event_name)
        return AgentHookEvent.from_payload(data, adapter_id="codex", hook_name=canonical, agent_name="codex", user_id=user_id)


class ClaudeCodeOutputRenderer:
    def render(self, hook_name: str, result: HookResult) -> dict[str, Any]:
        output: dict[str, Any] = {"continue": True, "suppressOutput": True}
        if result.injection_text and hook_name in {"SessionStart", "UserPromptSubmit", "PostToolUse"}:
            output["hookSpecificOutput"] = {"hookEventName": hook_name, "additionalContext": result.injection_text}
        if result.error:
            output["systemMessage"] = f"MemoryOS degraded: {result.error.get('code', 'UNKNOWN')}"
        return output


class CodexOutputRenderer(ClaudeCodeOutputRenderer):
    def render(self, hook_name: str, result: HookResult) -> dict[str, Any]:
        output: dict[str, Any] = {"continue": True, "suppressOutput": True}
        if result.injection_text and hook_name in {"SessionStart", "UserPromptSubmit", "PostToolUse", "PreCompact"}:
            output["hookSpecificOutput"] = {
                "hookEventName": hook_name,
                "additionalContext": result.injection_text,
            }
        if result.error:
            output["systemMessage"] = f"MemoryOS degraded: {result.error.get('code', 'UNKNOWN')}"
        return output
