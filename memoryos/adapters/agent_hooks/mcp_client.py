"""适配器里的MCP客户端。"""

from __future__ import annotations

import os
from typing import Any

from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.sdk.http_client import HTTPMemoryOSClient


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
            if name == "memoryos_remember":
                return self.remote.remember(**arguments)
            if name == "memoryos_forget":
                return self.remote.forget(**arguments)
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


# 这是保留给旧调用方的别名。它只会选择本地路由或 HTTP，并不负责 MCP 传输。
AgentHookMCPClient = AgentHookTransportClient
