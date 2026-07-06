from __future__ import annotations

import json
import sys
from typing import Any

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient

TOOLS = [
    {
        "name": "memoryos_search_context",
        "description": "Search MemoryOS context for a coding agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "user_id": {"type": "string"},
                "limit": {"type": "integer"},
                "context_type": {"type": "string"},
                "context_types": {"type": "array", "items": {"type": "string"}},
                "connect_metadata": {"type": "object"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memoryos_assemble_context",
        "description": "Assemble token-bounded MemoryOS context for prompt injection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "user_id": {"type": "string"},
                "token_budget": {"type": "integer"},
                "context_types": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
                "connect_metadata": {"type": "object"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memoryos_commit_session",
        "description": "Commit a sanitized agent session archive.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "session_id": {"type": "string"},
                "messages": {"type": "array", "items": {"type": "object"}},
                "used_contexts": {"type": "array", "items": {"type": "object"}},
                "tool_results": {"type": "array", "items": {"type": "object"}},
                "connect_metadata": {"type": "object"},
                "async_commit": {"type": "boolean"},
            },
            "required": ["session_id"],
        },
    },
    {"name": "memoryos_health", "description": "Check MemoryOS availability.", "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "memoryos_connection_schema", "description": "Describe allowed MemoryOS connection profiles.", "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {
        "name": "memoryos_predict",
        "description": "Action-capable embodied behavior prediction. Disabled by default.",
        "inputSchema": {
            "type": "object",
            "properties": {"request": {"type": "object"}, "policies": {"type": "array", "items": {"type": "object"}}, "connect_metadata": {"type": "object"}},
            "required": ["request"],
        },
    },
    {
        "name": "memoryos_process_observation",
        "description": "Action-capable embodied observation processing. Disabled by default.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request": {"type": "object"},
                "policies": {"type": "array", "items": {"type": "object"}},
                "connect_metadata": {"type": "object"},
                "archive_session": {"type": "boolean"},
                "async_commit": {"type": "boolean"},
            },
            "required": ["request"],
        },
    },
]


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
