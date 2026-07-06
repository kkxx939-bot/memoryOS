from __future__ import annotations

from typing import Any

from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient


class AgentHookMCPClient:
    def __init__(self, config: AgentHookConfig) -> None:
        mcp_config = MCPServerConfig(
            root=config.root,
            user_id=config.user_id,
            adapter_id=config.adapter_id,
            agent_name=config.agent_name,
            token_budget=config.token_budget,
            hook_queue_path=config.queue_path,
        )
        self.server = MemoryOSMCPServer(MemoryOSClient(config.root), config=mcp_config)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.server.call_tool(name, arguments)
