"""基于 stdin/stdout 的 MCP JSON-RPC 传输适配器。

该进程只处理握手、工具列表和工具调用等协议消息，再把实际调用交给进程内 MCP
服务；stdout 始终保持机器可解析的 JSON 行。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from openApi.mcp.config import MCPServerConfig
from openApi.mcp.schemas import tool_definitions
from openApi.mcp.server import MemoryOSMCPServer
from openApi.sdk.http_client import HTTPMemoryOSClient
from openApi.version import __version__

if TYPE_CHECKING:
    from openApi.sdk.client import MemoryOSClient

TOOLS = tool_definitions()


def run() -> None:
    """启动官方 MCP SDK 驱动的 stdio 服务循环。"""

    config = MCPServerConfig.from_env()
    server = MemoryOSMCPServer(_build_transport_client(config), config=config)
    try:
        import anyio
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.lowlevel.server import NotificationOptions
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise RuntimeError("Install memoryos[mcp] to run the MCP stdio server") from exc

    sdk_server = Server("memoryos", version=__version__)

    @sdk_server.list_tools()
    async def list_tools() -> list[Any]:
        return [
            types.Tool(name=item["name"], description=item["description"], inputSchema=item["inputSchema"])
            for item in tool_definitions(config)
        ]

    @sdk_server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        payload = server.call_tool(name, arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, sort_keys=True))],
            isError=bool(payload.get("error")),
        )

    async def serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await sdk_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memoryos",
                    server_version=__version__,
                    capabilities=sdk_server.get_capabilities(NotificationOptions(), {}),
                ),
            )

    anyio.run(serve)


def _build_transport_client(config: MCPServerConfig) -> MemoryOSClient | HTTPMemoryOSClient:
    """根据部署环境选择进程内 SDK 或远程 HTTP SDK。"""

    base_url = os.environ.get("MEMORYOS_BASE_URL", "").strip()
    if base_url:
        return HTTPMemoryOSClient(base_url)

    # 只有进程内模式才需要加载完整运行时和具体存储实现。
    from openApi.sdk.client import MemoryOSClient

    return MemoryOSClient(
        config.root,
        user_id=config.user_id,
        adapter_id=config.adapter_id,
        model_config=config.model,
    )


def _handle_jsonrpc(server: MemoryOSMCPServer, line: str) -> dict[str, Any]:
    request_id: Any = None
    try:
        request = json.loads(line)
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        result: dict[str, Any]
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "memoryos", "version": __version__},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": tool_definitions(server.config)}
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            tool_payload = server.call_tool(str(name), arguments if isinstance(arguments, dict) else {})
            result = {
                "content": [{"type": "text", "text": json.dumps(tool_payload, ensure_ascii=False, sort_keys=True)}],
                "isError": bool(tool_payload.get("error")),
            }
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except json.JSONDecodeError:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}
    except Exception:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": "Internal error"}}


if __name__ == "__main__":
    run()
