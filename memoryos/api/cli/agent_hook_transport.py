"""Composition-root transport for agent hook delivery."""

from __future__ import annotations

import os
from typing import Any, Protocol

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.sdk.http_client import HTTPMemoryOSClient


class AgentHookConfig(Protocol):
    """Configuration shape required by the delivery transport."""

    @property
    def root(self) -> str: ...

    @property
    def user_id(self) -> str: ...

    @property
    def tenant_id(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def agent_name(self) -> str: ...

    @property
    def token_budget(self) -> int: ...

    @property
    def queue_path(self) -> str: ...

    @property
    def allowed_workspace_ids(self) -> frozenset[str]: ...


class AgentHookTransportClient:
    def __init__(self, config: AgentHookConfig) -> None:
        mcp_config = MCPServerConfig(
            root=config.root,
            user_id=config.user_id,
            tenant_id=config.tenant_id,
            adapter_id=config.adapter_id,
            agent_name=config.agent_name,
            token_budget=config.token_budget,
            hook_queue_path=config.queue_path,
            allowed_workspace_ids=config.allowed_workspace_ids,
        )
        remote = os.environ.get("MEMORYOS_BASE_URL")
        self.server = (
            MemoryOSMCPServer(
                MemoryOSClient(config.root, tenant_id=config.tenant_id),
                config=mcp_config,
            )
            if not remote
            else None
        )
        self.remote = (
            HTTPMemoryOSClient(
                remote,
                api_token=os.environ.get("MEMORYOS_API_TOKEN"),
                account_id=os.environ.get("MEMORYOS_ACCOUNT_ID"),
                user_id=config.user_id,
                tenant_id=config.tenant_id,
            )
            if remote
            else None
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.remote:
            if name == "memoryos_search_context":
                return {
                    "results": self.remote.search_context(
                        arguments.get("query", ""), **{k: v for k, v in arguments.items() if k != "query"}
                    )
                }
            if name == "memoryos_assemble_context":
                return self.remote.assemble_context(
                    arguments.get("query", ""), **{k: v for k, v in arguments.items() if k != "query"}
                )
            if name == "memoryos_commit_session":
                return self.remote.commit_agent_session(**arguments)
            if name == "memoryos_health":
                return self.remote.health()
            if name == "memoryos_read":
                return self.remote.read(str(arguments.get("uri") or ""), layer=str(arguments.get("layer") or "L2"))
            if name == "memoryos_adopt_memory_document":
                return self.remote.adopt_memory_document(**arguments)
            if name == "memoryos_remember":
                return self.remote.remember(**arguments)
            if name == "memoryos_edit_memory_document":
                return self.remote.edit_memory_document(**arguments)
            if name == "memoryos_rename_memory_document":
                return self.remote.rename_memory_document(**arguments)
            if name == "memoryos_merge_memory_documents":
                return self.remote.merge_memory_documents(**arguments)
            if name == "memoryos_propose_memory_consolidation":
                return self.remote.propose_memory_consolidation(**arguments)
            if name == "memoryos_resume_memory_consolidation":
                return self.remote.resume_memory_consolidation(**arguments)
            if name == "memoryos_forget":
                return self.remote.forget(**arguments)
            if name == "memoryos_memory_history":
                return self.remote.list_memory_history(**arguments)
            if name == "memoryos_restore_memory_revision":
                return self.remote.restore_memory_revision(**arguments)
            if name == "memoryos_review_memory_edit":
                return self.remote.review_memory_edit(**arguments)
            if name == "memoryos_preview_memory_edit":
                return self.remote.preview_memory_edit(**arguments)
            if name == "memoryos_recall_trace":
                return self.remote.recall_trace(str(arguments.get("trace_id") or ""))
            return {
                "error": {
                    "code": "UNSUPPORTED_REMOTE_TOOL",
                    "message": f"Remote tool is not supported: {name}",
                    "retryable": False,
                }
            }
        if self.server is None:
            return {
                "error": {"code": "CLIENT_UNAVAILABLE", "message": "MemoryOS transport unavailable", "retryable": True}
            }
        return self.server.call_tool(name, arguments)


AgentHookMCPClient = AgentHookTransportClient

__all__ = ["AgentHookMCPClient", "AgentHookTransportClient"]
