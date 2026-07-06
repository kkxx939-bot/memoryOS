from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.events import AgentHookEvent
from memoryos.adapters.agent_hooks.mcp_client import AgentHookMCPClient
from memoryos.adapters.agent_hooks.queue import PendingItem, PendingQueue
from memoryos.adapters.agent_hooks.sanitizer import sanitize_payload, summarize_tool_result
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
        self.mcp_client = mcp_client or AgentHookMCPClient(config)
        self.queue = queue or PendingQueue(config.queue_path)

    def assemble_context(self, event: AgentHookEvent, *, token_budget: int | None = None) -> HookResult:
        try:
            result = self.mcp_client.call_tool(
                "memoryos_assemble_context",
                {
                    "query": event.query() or event.session_id,
                    "user_id": event.user_id or self.config.user_id,
                    "token_budget": token_budget or self.config.token_budget,
                    "connect_metadata": _agent_hook_metadata(event.adapter_id),
                },
            )
            if result.get("error"):
                return HookResult(ok=True, session_id=event.session_id, error=result["error"])
            injection = format_injection(result)
            return HookResult(
                ok=True,
                session_id=event.session_id,
                injection_text=injection,
                metadata={"source_uris": result.get("source_uris", []), "dropped_contexts": result.get("dropped_contexts", [])},
            )
        except Exception as exc:
            return HookResult(ok=True, session_id=event.session_id, error={"code": "HOOK_SOFT_FAIL", "message": exc.__class__.__name__})

    def enqueue_commit(self, event: AgentHookEvent, *, messages: list[dict[str, Any]] | None = None) -> HookResult:
        tool_result = summarize_tool_result(event.tool_name, event.tool_input, event.tool_output, event.changed_files)
        arguments = {
            "user_id": event.user_id or self.config.user_id,
            "session_id": event.session_id,
            "messages": sanitize_payload(messages if messages is not None else event.messages),
            "tool_results": [tool_result] if event.tool_name or event.tool_output is not None else [],
            "used_contexts": sanitize_payload(event.metadata.get("used_contexts", [])),
            "connect_metadata": _agent_hook_metadata(event.adapter_id),
            "async_commit": True,
        }
        queued = self.queue.enqueue(
            PendingItem(
                event_id=event.event_id,
                session_id=event.session_id,
                adapter_id=event.adapter_id,
                hook_name=event.hook_name,
                payload={"tool_name": "memoryos_commit_session", "arguments": arguments},
            )
        )
        return HookResult(ok=True, session_id=event.session_id, queued=queued)

    def commit_now(self, event: AgentHookEvent) -> HookResult:
        arguments = {
            "user_id": event.user_id or self.config.user_id,
            "session_id": event.session_id,
            "messages": sanitize_payload(event.messages),
            "used_contexts": sanitize_payload(event.metadata.get("used_contexts", [])),
            "tool_results": sanitize_payload(event.metadata.get("tool_results", [])),
            "connect_metadata": _agent_hook_metadata(event.adapter_id),
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
            return HookResult(ok=True, session_id=event.session_id, committed=True)
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
            return HookResult(ok=True, session_id=event.session_id, queued=queued, error={"code": "HOOK_SOFT_FAIL", "message": exc.__class__.__name__})

    def flush(self) -> HookResult:
        try:
            return HookResult(ok=True, flushed=self.queue.flush(self.mcp_client))
        except Exception as exc:
            return HookResult(ok=True, error={"code": "HOOK_SOFT_FAIL", "message": exc.__class__.__name__})


def format_injection(result: dict[str, Any]) -> str:
    packed = str(result.get("packed_context", "") or "")
    if not packed:
        return ""
    uris = [str(uri) for uri in result.get("source_uris", [])]
    sources = "\n".join(f"- {uri}" for uri in uris[:20])
    return f"<memoryos_context>\n{packed}\n\nSources:\n{sources}\n</memoryos_context>"


def _agent_hook_metadata(adapter_id: str) -> dict[str, Any]:
    return ConnectMetadata(
        connect_type=ConnectType.AGENT,
        adapter_id=adapter_id,
        run_mode=PipelineMode.CONTEXT_REDUCTION,
        world_domain="digital",
        source_kind="coding_agent",
    ).to_dict()
