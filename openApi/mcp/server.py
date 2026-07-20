"""MCP 服务端的进程内公开门面。

门面公开工具清单并把调用委托给 ``MCPToolRouter``，便于 stdio、测试和嵌入式
宿主复用同一套工具行为。
"""

from __future__ import annotations

from typing import Any

from openApi.mcp.config import MCPServerConfig
from openApi.mcp.tools import MCPToolRouter


class MemoryOSMCPServer:
    """公开 MCP 工具列表并执行单次工具调用的轻量门面。"""

    def __init__(self, client: Any, config: MCPServerConfig | None = None) -> None:
        self.client = client
        self.config = config or MCPServerConfig.from_env()
        self.router = MCPToolRouter(client, config=self.config)

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """把工具名和参数交给统一安全路由。"""

        return self.router.call(name, arguments or {})
