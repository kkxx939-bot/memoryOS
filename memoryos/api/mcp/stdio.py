from __future__ import annotations

import json
import sys
from typing import Any

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.schemas import tool_definitions
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient

TOOLS = tool_definitions()


def main() -> None:
    config = MCPServerConfig.from_env()
    server = MemoryOSMCPServer(MemoryOSClient(config.root), config=config)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = _handle_jsonrpc(server, line)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def _handle_jsonrpc(server: MemoryOSMCPServer, line: str) -> dict[str, Any]:
    try:
        request = json.loads(line)
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        result: dict[str, Any]
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "memoryos", "version": "0.1.0"}, "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
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
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)[:200] or exc.__class__.__name__}}


if __name__ == "__main__":
    main()
