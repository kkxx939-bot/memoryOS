"""Agent 事件定义。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from memoryos.core.clock import utc_now
from memoryos.security.workspace_identity import repository_workspace_id


class AgentEventType(str, Enum):
    SESSION_START = "SESSION_START"
    PROMPT_SUBMIT = "PROMPT_SUBMIT"
    TOOL_RESULT = "TOOL_RESULT"
    TURN_END = "TURN_END"
    PRE_COMPACT = "PRE_COMPACT"
    POST_COMPACT = "POST_COMPACT"
    SESSION_END = "SESSION_END"
    SUBAGENT_END = "SUBAGENT_END"


@dataclass(frozen=True)
class NormalizedAgentEvent:
    event_id: str
    event_type: AgentEventType
    adapter_id: str
    user_id: str
    project_id: str
    native_session_id: str
    session_key: str
    tenant_id: str = "default"
    cwd: str | None = None
    repo_root: str | None = None
    git_remote: str | None = None
    branch: str | None = None
    worktree_id: str | None = None
    prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call: dict[str, Any] | None = None
    changed_files: list[str] = field(default_factory=list)
    transcript_path: str | None = None
    transcript_cursor: str | int | None = None
    timestamp: str = field(default_factory=utc_now)
    raw_event_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHookEvent:
    event_id: str
    agent_name: str
    adapter_id: str
    hook_name: str
    session_id: str
    user_id: str | None = None
    cwd: str | None = None
    repo_root: str | None = None
    branch: str | None = None
    prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | str | None = None
    changed_files: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "default"

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        adapter_id: str,
        hook_name: str,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> AgentHookEvent:
        data = dict(payload or {})
        cwd = _str_or_none(data.get("cwd") or data.get("workspace") or os.getcwd())
        repo_root = _str_or_none(data.get("repo_root")) or _git_value(cwd, ["rev-parse", "--show-toplevel"])
        branch = _str_or_none(data.get("branch")) or _git_value(cwd, ["branch", "--show-current"])
        git_remote = _str_or_none(data.get("git_remote")) or _git_value(cwd, ["config", "--get", "remote.origin.url"])
        prompt = _str_or_none(data.get("prompt") or data.get("user_prompt") or data.get("input"))
        raw_messages = data.get("messages")
        messages: list[Any] = raw_messages if isinstance(raw_messages, list) else []
        raw_changed_files = data.get("changed_files")
        changed_files: list[Any] = raw_changed_files if isinstance(raw_changed_files, list) else []
        session_id = _str_or_none(
            data.get("session_id") or data.get("conversation_id") or os.environ.get("MEMORYOS_SESSION_ID")
        )
        if not session_id:
            session_id = (
                "agent-"
                + _stable_hash(
                    {
                        "adapter_id": adapter_id,
                        "cwd": cwd,
                        "repo_root": repo_root,
                        "branch": branch,
                        "task_hint": _task_hint(data, prompt, messages, changed_files),
                    }
                )[:16]
            )
        event_core = {
            "adapter_id": adapter_id,
            "hook_name": hook_name,
            "session_id": session_id,
            "prompt": prompt,
            "tool_name": data.get("tool_name") or data.get("name"),
            "tool_input": data.get("tool_input"),
            "tool_output": data.get("tool_output", data.get("tool_response", data.get("output"))),
            "changed_files": changed_files,
            "messages": messages,
        }
        event_id = _str_or_none(data.get("event_id")) or "hook-" + _stable_hash(event_core)[:24]
        known = {
            "event_id",
            "agent_name",
            "adapter_id",
            "hook_name",
            "session_id",
            "user_id",
            "tenant_id",
            "cwd",
            "workspace",
            "repo_root",
            "branch",
            "prompt",
            "user_prompt",
            "input",
            "messages",
            "tool_name",
            "name",
            "tool_input",
            "tool_output",
            "tool_response",
            "output",
            "changed_files",
            "timestamp",
            "metadata",
            "used_contexts",
            "used_skills",
            "tool_results",
            "scope",
            "provenance",
        }
        metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}
        metadata.update({key: value for key, value in data.items() if key not in known})
        for key in ("used_contexts", "used_skills", "tool_results", "scope", "provenance"):
            if key in data:
                metadata[key] = data[key]
        if git_remote:
            metadata["git_remote"] = git_remote
        return cls(
            event_id=event_id,
            agent_name=_str_or_none(data.get("agent_name")) or agent_name or adapter_id,
            adapter_id=adapter_id,
            hook_name=hook_name,
            session_id=session_id,
            user_id=_str_or_none(data.get("user_id")) or user_id,
            cwd=cwd,
            repo_root=repo_root,
            branch=branch,
            prompt=prompt,
            messages=[dict(item) for item in messages if isinstance(item, dict)],
            tool_name=_str_or_none(data.get("tool_name") or data.get("name")),
            tool_input=dict(data.get("tool_input", {})) if isinstance(data.get("tool_input"), dict) else None,
            tool_output=data.get("tool_output", data.get("tool_response", data.get("output"))),
            changed_files=[str(item) for item in changed_files],
            timestamp=_str_or_none(data.get("timestamp")) or utc_now(),
            metadata=metadata,
            tenant_id=_str_or_none(data.get("tenant_id")) or "default",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def query(self) -> str:
        parts = [self.prompt or "", self.repo_root or self.cwd or "", self.branch or ""]
        return "\n".join(part for part in parts if part)

    def normalize(self) -> NormalizedAgentEvent:
        project_id = str(
            self.metadata.get("project_id")
            or project_identity(self.repo_root, self.cwd, self.metadata.get("git_remote"))
        )
        user_id = self.user_id or "default"
        session_key = make_session_key(
            user_id,
            project_id,
            self.adapter_id,
            self.session_id,
            tenant_id=self.tenant_id,
        )
        event_type = EVENT_TYPE_MAP.get(self.hook_name, AgentEventType.TURN_END)
        tool_call = None
        if self.tool_name or self.tool_input is not None or self.tool_output is not None:
            tool_call = {"name": self.tool_name, "input": self.tool_input, "output": self.tool_output}
        return NormalizedAgentEvent(
            event_id=self.event_id,
            event_type=event_type,
            adapter_id=self.adapter_id,
            user_id=user_id,
            project_id=project_id,
            native_session_id=self.session_id,
            session_key=session_key,
            tenant_id=self.tenant_id,
            cwd=self.cwd,
            repo_root=self.repo_root,
            git_remote=_str_or_none(self.metadata.get("git_remote")),
            branch=self.branch,
            worktree_id=_str_or_none(self.metadata.get("worktree_id")),
            prompt=self.prompt,
            messages=list(self.messages),
            tool_call=tool_call,
            changed_files=list(self.changed_files),
            transcript_path=_str_or_none(self.metadata.get("transcript_path")),
            transcript_cursor=self.metadata.get("transcript_cursor"),
            timestamp=self.timestamp,
            raw_event_name=self.hook_name,
            metadata=dict(self.metadata),
        )


EVENT_TYPE_MAP = {
    **{event.value: event for event in AgentEventType},
    "SessionStart": AgentEventType.SESSION_START,
    "session_start": AgentEventType.SESSION_START,
    "UserPromptSubmit": AgentEventType.PROMPT_SUBMIT,
    "before_prompt": AgentEventType.PROMPT_SUBMIT,
    "user_prompt_submit": AgentEventType.PROMPT_SUBMIT,
    "PostToolUse": AgentEventType.TOOL_RESULT,
    "tool_result": AgentEventType.TOOL_RESULT,
    "post_tool_use": AgentEventType.TOOL_RESULT,
    "Stop": AgentEventType.TURN_END,
    "after_turn": AgentEventType.TURN_END,
    "PreCompact": AgentEventType.PRE_COMPACT,
    "pre_compact": AgentEventType.PRE_COMPACT,
    "PostCompact": AgentEventType.POST_COMPACT,
    "post_compact": AgentEventType.POST_COMPACT,
    "SessionEnd": AgentEventType.SESSION_END,
    "session_end": AgentEventType.SESSION_END,
    "SubagentStop": AgentEventType.SUBAGENT_END,
    "subagent_end": AgentEventType.SUBAGENT_END,
    "subagent_stop": AgentEventType.SUBAGENT_END,
}


def project_identity(repo_root: str | None, cwd: str | None, git_remote: str | None = None) -> str:
    return repository_workspace_id(repo_root=repo_root or "", cwd=cwd or "", git_remote=git_remote or "")


def make_session_key(
    user_id: str,
    project_id: str,
    adapter_id: str,
    native_session_id: str,
    tenant_id: str = "default",
) -> str:
    parts = (
        (user_id, project_id, adapter_id, native_session_id)
        if tenant_id == "default"
        else ("tenant-v1", tenant_id, user_id, project_id, adapter_id, native_session_id)
    )
    raw = "|".join(parts)
    return "session-" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_hint(
    data: dict[str, Any], prompt: str | None, messages: list[Any], changed_files: list[Any]
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "messages": messages,
        "tool_name": data.get("tool_name") or data.get("name"),
        "tool_input": data.get("tool_input"),
        "changed_files": changed_files,
    }


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _git_value(cwd: str | None, args: list[str]) -> str | None:
    if not cwd:
        return None
    try:
        output = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True, timeout=1)
    except Exception:
        return None
    value = output.stdout.strip()
    return value or None
