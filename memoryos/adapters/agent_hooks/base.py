"""Agent 适配器基础接口。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.injection import MemoryOSContextRenderer
from memoryos.adapters.agent_hooks.queue import PendingItem, PendingQueue
from memoryos.adapters.agent_hooks.session_service import AgentSessionService
from memoryos.application.session.events import AgentHookEvent
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode
from memoryos.security.sanitization import sanitize_error_text


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
        if mcp_client is None:
            raise TypeError("agent hook adapters require a transport supplied by the composition root")
        self.mcp_client = mcp_client
        if queue is not None and (queue.tenant_id != config.tenant_id or queue.user_id != config.user_id):
            raise ValueError("pending hook queue principal does not match AgentHookConfig")
        self.queue = queue or PendingQueue(
            config.queue_path,
            tenant_id=config.tenant_id,
            user_id=config.user_id,
        )
        self.session_service = AgentSessionService(
            config.root,
            tenant_id=config.tenant_id,
            transcript_roots=config.transcript_roots,
            max_transcript_bytes=config.max_transcript_bytes,
        )

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
                metadata={
                    "source_uris": result.get("source_uris", []),
                    "dropped_contexts": result.get("dropped_contexts", []),
                },
            )
        except Exception as exc:
            return HookResult(ok=True, session_id=event.session_id, error=_hook_error(exc, "assemble_context"))

    def enqueue_commit(self, event: AgentHookEvent, *, messages: list[dict[str, Any]] | None = None) -> HookResult:
        normalized = self._append_event(event)
        return HookResult(
            ok=True, session_id=event.session_id, metadata={"state": "ARCHIVED", "session_key": normalized.session_key}
        )

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
                        tenant_id=self.config.tenant_id,
                        user_id=self.config.user_id,
                        last_error=str(result["error"].get("code", "")),
                    )
                )
                return HookResult(ok=True, session_id=event.session_id, queued=queued, error=result["error"])
            result_payload = result.get("result", result)
            state = str(result_payload.get("state") or "").upper()
            status = str(result_payload.get("status", result.get("status", ""))).lower()
            committed = state == "COMMITTED" or status in {"done", "done_with_pending", "committed"}
            queued = state in {"QUEUED", "PROCESSING"} or status in {"queued", "processing"}
            commit_state = (
                "COMMITTED"
                if committed
                else "DEAD_LETTER"
                if state == "DEAD_LETTER"
                else "FAILED_RETRYABLE"
                if state == "FAILED_RETRYABLE"
                else "PROCESSING"
                if state == "PROCESSING"
                else "QUEUED"
            )
            self.session_service.finalize(normalized.session_key, commit_state=commit_state)
            return HookResult(
                ok=True,
                session_id=event.session_id,
                committed=committed,
                queued=queued,
                metadata={
                    "project_id": normalized.project_id,
                    "session_key": normalized.session_key,
                    "state": commit_state,
                },
            )
        except Exception as exc:
            queued = self.queue.enqueue(
                PendingItem(
                    event_id=event.event_id,
                    session_id=event.session_id,
                    adapter_id=event.adapter_id,
                    hook_name=event.hook_name,
                    payload={"tool_name": "memoryos_commit_session", "arguments": arguments},
                    tenant_id=self.config.tenant_id,
                    user_id=self.config.user_id,
                    last_error=exc.__class__.__name__,
                )
            )
            return HookResult(
                ok=True, session_id=event.session_id, queued=queued, error=_hook_error(exc, "commit_session")
            )

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
        event.tenant_id = self.config.tenant_id
        event.user_id = self.config.user_id
        event.adapter_id = self.config.adapter_id
        event.agent_name = self.config.agent_name
        normalized = event.normalize()
        if normalized.project_id not in self.config.allowed_workspace_ids:
            raise PermissionError("agent hook workspace is not authorized by its trusted configuration")
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
