from __future__ import annotations

from typing import Any

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.tools import MCPToolRouter


class MemoryOSMCPServer:
    def __init__(self, client: Any, config: MCPServerConfig | None = None) -> None:
        self.client = client
        self.config = config or MCPServerConfig.from_env()
        self.router = MCPToolRouter(client, config=self.config)

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self.router.call(name, arguments or {})
