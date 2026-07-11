"""Agent 适配器基础接口。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.events import AgentHookEvent
from memoryos.adapters.agent_hooks.injection import MemoryOSContextRenderer
from memoryos.adapters.agent_hooks.mcp_client import AgentHookTransportClient
from memoryos.adapters.agent_hooks.queue import PendingItem, PendingQueue
from memoryos.adapters.agent_hooks.sanitizer import sanitize_error_text
from memoryos.adapters.agent_hooks.session_service import AgentSessionService
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode


@dataclass
class HookResult:
    ok: bool
    session_id: str = ""
    injection_text: str = ""
    queued: bool = False
    committed: bool = False
    flushed: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "session_id": self.session_id,
            "injection_text": self.injection_text,
            "queued": self.queued,
            "committed": self.committed,
            "flushed": self.flushed,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseAgentHookAdapter:
    def __init__(
        self,
        config: AgentHookConfig,
        *,
        mcp_client: Any | None = None,
        queue: PendingQueue | None = None,
    ) -> None:
        self.config = config
        self.mcp_client = mcp_client or AgentHookTransportClient(config)
        self.queue = queue or PendingQueue(config.queue_path)
        self.session_service = AgentSessionService(config.root)

    def assemble_context(self, event: AgentHookEvent, *, token_budget: int | None = None) -> HookResult:
        normalized = self._append_event(event)
        try:
            result = self.mcp_client.call_tool(
                "memoryos_assemble_context",
                {
                    "query": event.query() or event.session_id,
                    "user_id": event.user_id or self.config.user_id,
                    "token_budget": token_budget or self.config.token_budget,
                    "connect_metadata": _agent_hook_metadata(event.adapter_id, event),
                    "project_id": _project_id_from_event(event),
                },
            )
            if result.get("error"):
                return HookResult(ok=True, session_id=event.session_id, error=result["error"])
            injection = format_injection(result)
            self.session_service.record_recall(normalized.session_key, result)
            return HookResult(
                ok=True,
                session_id=event.session_id,
                injection_text=injection,
                metadata={"source_uris": result.get("source_uris", []), "dropped_contexts": result.get("dropped_contexts", [])},
            )
        except Exception as exc:
            return HookResult(ok=True, session_id=event.session_id, error=_hook_error(exc, "assemble_context"))

    def enqueue_commit(self, event: AgentHookEvent, *, messages: list[dict[str, Any]] | None = None) -> HookResult:
        normalized = self._append_event(event)
        return HookResult(ok=True, session_id=event.session_id, metadata={"state": "ARCHIVED", "session_key": normalized.session_key})

    def commit_now(self, event: AgentHookEvent) -> HookResult:
        normalized = self._append_event(event)
        arguments = {
            **self.session_service.commit_payload(normalized),
            "connect_metadata": _agent_hook_metadata(event.adapter_id, event),
            "async_commit": True,
        }
        try:
            result = self.mcp_client.call_tool("memoryos_commit_session", arguments)
            if result.get("error"):
                queued = self.queue.enqueue(
                    PendingItem(
                        event_id=event.event_id,
                        session_id=event.session_id,
                        adapter_id=event.adapter_id,
                        hook_name=event.hook_name,
                        payload={"tool_name": "memoryos_commit_session", "arguments": arguments},
                        last_error=str(result["error"].get("code", "")),
                    )
                )
                return HookResult(ok=True, session_id=event.session_id, queued=queued, error=result["error"])
            self.session_service.finalize(normalized.session_key)
            result_payload = result.get("result", result)
            status = str(result_payload.get("status", result.get("status", ""))).lower()
            committed = status in {"done", "committed"}
            queued = status in {"queued", "processing"}
            self.session_service.finalize(normalized.session_key, commit_state="COMMITTED" if committed else "QUEUED")
            return HookResult(ok=True, session_id=event.session_id, committed=committed, queued=queued, metadata={"project_id": normalized.project_id, "session_key": normalized.session_key, "state": status.upper()})
        except Exception as exc:
            queued = self.queue.enqueue(
                PendingItem(
                    event_id=event.event_id,
                    session_id=event.session_id,
                    adapter_id=event.adapter_id,
                    hook_name=event.hook_name,
                    payload={"tool_name": "memoryos_commit_session", "arguments": arguments},
                    last_error=exc.__class__.__name__,
                )
            )
            return HookResult(ok=True, session_id=event.session_id, queued=queued, error=_hook_error(exc, "commit_session"))

    def flush(self) -> HookResult:
        try:
            return HookResult(ok=True, flushed=self.queue.flush(self.mcp_client))
        except Exception as exc:
            return HookResult(ok=True, error=_hook_error(exc, "flush_queue"))

    def checkpoint(self, event: AgentHookEvent) -> HookResult:
        normalized = self._append_event(event)
        metadata = self.session_service.checkpoint(normalized.session_key)
        return HookResult(ok=True, session_id=event.session_id, metadata=metadata)

    def _append_event(self, event: AgentHookEvent):
        normalized = event.normalize()
        self.session_service.append_event(normalized)
        self.session_service.append_transcript(normalized)
        return normalized


def format_injection(result: dict[str, Any]) -> str:
    return MemoryOSContextRenderer().render(result)


def _agent_hook_metadata(adapter_id: str, event: AgentHookEvent | None = None) -> dict[str, Any]:
    return ConnectMetadata(
        connect_type=ConnectType.AGENT,
        adapter_id=adapter_id,
        run_mode=PipelineMode.CONTEXT_REDUCTION,
        world_domain="digital",
        source_kind="coding_agent",
    ).to_dict() | ({"project_id": _project_id_from_event(event)} if event else {})


def _project_id_from_event(event: AgentHookEvent) -> str:
    for key in ("project_id", "project"):
        value = event.metadata.get(key)
        if value:
            return str(value)
    return ""


def _hook_error(exc: Exception, operation: str) -> dict[str, Any]:
    return {
        "code": "HOOK_SOFT_FAIL",
        "message": sanitize_error_text(str(exc) or exc.__class__.__name__),
        "retryable": True,
        "request_id": str(uuid.uuid4()),
        "operation": operation,
    }
