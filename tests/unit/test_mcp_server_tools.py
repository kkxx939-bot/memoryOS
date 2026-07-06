from __future__ import annotations

import json
from typing import Any, cast

from memoryos.api.mcp import stdio
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata


class FakeMCPClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.assemble_calls: list[dict[str, Any]] = []
        self.commit_calls: list[dict[str, Any]] = []
        self.predict_calls = 0
        self.process_calls = 0
        self.fail_search = False

    def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, **kwargs})
        if self.fail_search:
            raise RuntimeError("boom /Users/gulf/secret")
        return [{"uri": "memoryos://user/u1/memories/anchors/1", "text": "MemoryOS MCP", "metadata": {}}]

    def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.assemble_calls.append({"query": query, **kwargs})
        return {
            "packed_context": "short context",
            "contexts": [{"uri": "memoryos://ctx/1"}],
            "source_uris": ["memoryos://ctx/1"],
            "dropped_contexts": [{"uri": "memoryos://ctx/2", "reason": "token_budget"}],
        }

    def commit_agent_session(self, **kwargs: Any) -> dict[str, Any]:
        self.commit_calls.append(kwargs)
        return {"status": "done", "archive_uri": "memoryos://session/s1"}

    def predict(self, request: Any, policies: Any = None) -> Any:
        self.predict_calls += 1

        class Result:
            def to_dict(self) -> dict[str, Any]:
                return {"episode_id": request.episode_id}

        return Result()

    def process_observation(self, request: Any, policies: Any = None, **kwargs: Any) -> Any:
        self.process_calls += 1

        class Result:
            def to_dict(self) -> dict[str, Any]:
                return {"archive_uri": "memoryos://session/s1"}

        return Result()


def _server(*, enable_action_tools: bool = False) -> tuple[MemoryOSMCPServer, FakeMCPClient]:
    client = FakeMCPClient()
    config = MCPServerConfig(
        root="/tmp/memory",
        user_id="u1",
        adapter_id="codex",
        agent_name="codex",
        token_budget=64,
        enable_action_tools=enable_action_tools,
    )
    return MemoryOSMCPServer(cast(MemoryOSClient, client), config=config), client


def _request(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": "u1",
        "episode_id": "s1",
        "observation": "room is hot",
        "available_actions": ["turn_on_ac"],
        "connect_metadata": metadata,
    }


def test_mcp_search_context_returns_structured_contexts_and_agent_metadata() -> None:
    server, client = _server()

    result = server.call_tool("memoryos_search_context", {"query": "MCP", "connect_metadata": {"adapter_id": "codex"}})

    assert result["error"] is None
    assert result["contexts"][0]["uri"] == "memoryos://user/u1/memories/anchors/1"
    assert result["source_uris"] == ["memoryos://user/u1/memories/anchors/1"]
    connect = client.search_calls[0]["connect_metadata"]
    assert connect["connect_type"] == "agent"
    assert connect["run_mode"] == "context_reduction"
    assert connect["source_kind"] == "coding_agent"
    assert connect["capabilities"]["can_predict_behavior"] is False


def test_mcp_assemble_context_respects_token_budget_and_dropped_contexts() -> None:
    server, client = _server()

    result = server.call_tool("memoryos_assemble_context", {"query": "MCP", "token_budget": 32})

    assert result["error"] is None
    assert result["packed_context"] == "short context"
    assert result["token_budget"] == 32
    assert result["estimated_tokens"] <= 32
    assert result["dropped_contexts"] == [{"uri": "memoryos://ctx/2", "reason": "token_budget"}]
    assert client.assemble_calls[0]["token_budget"] == 32


def test_mcp_commit_session_returns_structured_result() -> None:
    server, client = _server()

    result = server.call_tool("memoryos_commit_session", {"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})

    assert result["error"] is None
    assert result["status"] == "done"
    assert client.commit_calls[0]["session_id"] == "s1"
    assert client.commit_calls[0]["connect_metadata"]["adapter_id"] == "codex"


def test_mcp_validation_and_client_errors_are_structured() -> None:
    server, client = _server()

    missing = server.call_tool("memoryos_search_context", {})
    assert missing["error"]["code"] == "VALIDATION_ERROR"

    client.fail_search = True
    failed = server.call_tool("memoryos_search_context", {"query": "MCP"})
    assert failed["error"]["code"] == "CLIENT_ERROR"
    assert "/Users/gulf" not in failed["error"]["message"]


def test_mcp_connection_schema_and_health() -> None:
    server, _client = _server()

    schema = server.call_tool("memoryos_connection_schema", {})
    health = server.call_tool("memoryos_health", {})

    assert server.config.user_id == "u1"
    assert schema["error"] is None
    assert schema["action_tools_enabled"] is False
    assert "codex" in schema["allowed_adapter_ids"]
    assert health["client_ready"] is True
    assert server.call_tool("memoryos_health", None)["status"] == "ok"


def test_action_tools_default_closed_and_coding_agent_rejected() -> None:
    server, client = _server(enable_action_tools=False)
    coding_metadata = {"adapter_id": "codex"}

    result = server.call_tool("memoryos_predict", {"request": _request(coding_metadata)})
    observation_result = server.call_tool(
        "memoryos_process_observation",
        {"request": _request(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())},
    )

    assert result["error"]["code"] == "PERMISSION_DENIED"
    assert observation_result["error"]["code"] == "PERMISSION_DENIED"
    assert client.predict_calls == 0
    assert client.process_calls == 0


def test_action_tools_enabled_still_requires_embodied_metadata() -> None:
    server, client = _server(enable_action_tools=True)

    missing = server.call_tool("memoryos_predict", {"request": {"user_id": "u1", "episode_id": "s1", "observation": "hot", "available_actions": ["turn_on_ac"]}})
    result = server.call_tool("memoryos_predict", {"request": _request({"adapter_id": "codex"})})

    assert missing["error"]["code"] == "PERMISSION_DENIED"
    assert result["error"]["code"] == "PERMISSION_DENIED"
    assert client.predict_calls == 0


def test_action_tools_allow_reachy_mini_embodied_predict_only_with_capability() -> None:
    server, client = _server(enable_action_tools=True)
    metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()

    result = server.call_tool("memoryos_predict", {"request": _request(metadata)})

    assert result["error"] is None
    assert result["prediction"] == {"episode_id": "s1"}
    assert client.predict_calls == 1


def test_process_observation_requires_execute_capability() -> None:
    server, client = _server(enable_action_tools=True)
    metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()
    metadata["capabilities"]["can_execute_action"] = False

    denied = server.call_tool("memoryos_process_observation", {"request": _request(metadata)})
    allowed = server.call_tool("memoryos_process_observation", {"request": _request(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())})

    assert denied["error"]["code"] == "PERMISSION_DENIED"
    assert allowed["error"] is None
    assert client.process_calls == 1


def test_stdio_initialize_tools_list_and_health_call_include_stable_schemas() -> None:
    server, _client = _server()

    initialized = stdio._handle_jsonrpc(server, json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
    listed = stdio._handle_jsonrpc(server, json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    called = stdio._handle_jsonrpc(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "memoryos_health", "arguments": {}},
            }
        ),
    )

    assert initialized["result"]["serverInfo"]["name"] == "memoryos"
    tools = listed["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert {"memoryos_health", "memoryos_connection_schema"} <= names
    assert all(tool["inputSchema"]["type"] == "object" for tool in tools)
    assemble = next(tool for tool in tools if tool["name"] == "memoryos_assemble_context")
    assert assemble["inputSchema"]["type"] == "object"
    assert "query" in assemble["inputSchema"]["required"]
    health_payload = json.loads(called["result"]["content"][0]["text"])
    assert called["result"]["isError"] is False
    assert health_payload["status"] == "ok"


def test_stdio_action_tool_disabled_returns_permission_error_payload() -> None:
    server, _client = _server()
    response = stdio._handle_jsonrpc(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "memoryos_predict",
                    "arguments": {"request": _request(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())},
                },
            }
        ),
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert response["result"]["isError"] is True
    assert payload["error"]["code"] == "PERMISSION_DENIED"


def test_stdio_unknown_tool_returns_tool_error_payload() -> None:
    server, _client = _server()

    response = stdio._handle_jsonrpc(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "unknown", "arguments": {}},
            }
        ),
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert response["result"]["isError"] is True
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_stdio_malformed_json_returns_parse_error() -> None:
    server, _client = _server()

    response = stdio._handle_jsonrpc(server, "{bad json")

    assert response["error"]["code"] == -32700
    assert response["error"]["message"] == "Invalid JSON"
