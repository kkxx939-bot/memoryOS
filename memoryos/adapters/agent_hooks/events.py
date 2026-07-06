from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from memoryos.core.time import utc_now


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
        session_id = _str_or_none(data.get("session_id") or data.get("conversation_id") or os.environ.get("MEMORYOS_SESSION_ID"))
        if not session_id:
            session_id = "agent-" + _stable_hash({"adapter_id": adapter_id, "cwd": cwd, "repo_root": repo_root})[:16]
        prompt = _str_or_none(data.get("prompt") or data.get("user_prompt") or data.get("input"))
        raw_messages = data.get("messages")
        messages: list[Any] = raw_messages if isinstance(raw_messages, list) else []
        raw_changed_files = data.get("changed_files")
        changed_files: list[Any] = raw_changed_files if isinstance(raw_changed_files, list) else []
        event_core = {
            "adapter_id": adapter_id,
            "hook_name": hook_name,
            "session_id": session_id,
            "timestamp": data.get("timestamp"),
            "prompt": prompt,
            "tool_name": data.get("tool_name") or data.get("name"),
        }
        event_id = _str_or_none(data.get("event_id")) or "hook-" + _stable_hash(event_core)[:24]
        known = {
            "event_id",
            "agent_name",
            "adapter_id",
            "hook_name",
            "session_id",
            "user_id",
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
            "output",
            "changed_files",
            "timestamp",
            "metadata",
            "used_contexts",
            "tool_results",
        }
        metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}
        metadata.update({key: value for key, value in data.items() if key not in known})
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
            tool_output=data.get("tool_output", data.get("output")),
            changed_files=[str(item) for item in changed_files],
            timestamp=_str_or_none(data.get("timestamp")) or utc_now(),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def query(self) -> str:
        parts = [self.prompt or "", self.repo_root or self.cwd or "", self.branch or ""]
        return "\n".join(part for part in parts if part)


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
