from __future__ import annotations

import json
import os
from typing import Any

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.schemas import tool_definitions
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.sdk.http_client import HTTPMemoryOSClient

TOOLS = tool_definitions()


def main() -> None:
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

    sdk_server = Server("memoryos", version="0.1.0")

    @sdk_server.list_tools()
    async def list_tools() -> list[Any]:
        return [types.Tool(name=item["name"], description=item["description"], inputSchema=item["inputSchema"]) for item in tool_definitions(config)]

    @sdk_server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        payload = server.call_tool(name, arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, sort_keys=True))],
            isError=bool(payload.get("error")),
        )

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await sdk_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memoryos",
                    server_version="0.1.0",
                    capabilities=sdk_server.get_capabilities(NotificationOptions(), {}),
                ),
            )

    anyio.run(run)


def _build_transport_client(config: MCPServerConfig) -> MemoryOSClient | HTTPMemoryOSClient:
    base_url = os.environ.get("MEMORYOS_BASE_URL", "").strip()
    return (
        HTTPMemoryOSClient(
            base_url,
            api_token=os.environ.get("MEMORYOS_API_TOKEN"),
            account_id=os.environ.get("MEMORYOS_ACCOUNT_ID"),
            user_id=config.user_id,
        )
        if base_url
        else MemoryOSClient(config.root)
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
            result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "memoryos", "version": "0.1.0"}, "capabilities": {"tools": {}}}
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
    main()
