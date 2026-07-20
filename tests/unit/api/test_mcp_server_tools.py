from __future__ import annotations

import json
from typing import Any, cast

from foundation.readiness import RuntimeNotReadyError, RuntimeReadinessState
from infrastructure.store.model.context.context_type import ContextType
from openApi.mcp import stdio
from openApi.mcp.config import MCPServerConfig
from openApi.mcp.server import MemoryOSMCPServer
from openApi.sdk.client import MemoryOSClient
from pre.connect import ConnectMetadata


def _document_result(*, edit_summary: str = "explicit remember") -> dict[str, Any]:
    return {
        "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "document_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "document_kind": "topic",
        "relative_path": "topics/distributed-systems.md",
        "document_revision": 1,
        "source_digest": "a" * 64,
        "changed": True,
        "edit_summary": edit_summary,
        "projection_status": "ENQUEUED",
    }


class FakeMCPClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.assemble_calls: list[dict[str, Any]] = []
        self.commit_calls: list[dict[str, Any]] = []
        self.predict_calls = 0
        self.process_calls = 0
        self.fail_search = False
        self.tenant_id = "default"
        self.adopt_calls: list[dict[str, Any]] = []
        self.remember_calls: list[dict[str, Any]] = []
        self.edit_calls: list[dict[str, Any]] = []
        self.rename_calls: list[dict[str, Any]] = []
        self.merge_calls: list[dict[str, Any]] = []
        self.consolidation_proposal_calls: list[dict[str, Any]] = []
        self.forget_calls: list[dict[str, Any]] = []
        self.history_calls: list[dict[str, Any]] = []
        self.restore_calls: list[dict[str, Any]] = []
        self.review_calls: list[dict[str, Any]] = []
        self.health_payload: dict[str, Any] = {
            "status": "ready",
            "runtime": {"state": "READY", "ready": True, "reasons": []},
        }

    def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, **kwargs})
        if self.fail_search:
            raise RuntimeError("boom /Users/gulf/secret")
        context_type = kwargs.get("context_type")
        suffix = str(context_type or "all")
        return [
            {
                "uri": f"memoryos://user/u1/memories/anchors/{suffix}",
                "text": "MemoryOS MCP",
                "metadata": {},
                "score": 1.0,
            }
        ]

    def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.assemble_calls.append({"query": query, **kwargs})
        return {
            "packed_context": "short context",
            "contexts": [{"uri": "memoryos://ctx/1"}],
            "source_uris": ["memoryos://ctx/1"],
            "dropped_contexts": [{"uri": "memoryos://ctx/2", "reason": "section_limit"}],
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

    def remember(self, **kwargs: Any) -> dict[str, Any]:
        self.remember_calls.append(kwargs)
        return _document_result()

    def adopt_memory_document(self, **kwargs: Any) -> dict[str, Any]:
        self.adopt_calls.append(kwargs)
        return _document_result(edit_summary="adopt unmanaged Markdown document")

    def edit_memory_document(self, **kwargs: Any) -> dict[str, Any]:
        self.edit_calls.append(kwargs)
        return _document_result(edit_summary="explicit full-document edit")

    def rename_memory_document(self, **kwargs: Any) -> dict[str, Any]:
        self.rename_calls.append(kwargs)
        return _document_result(edit_summary="rename and edit memory document")

    def merge_memory_documents(self, **kwargs: Any) -> dict[str, Any]:
        self.merge_calls.append(kwargs)
        return {
            "saga_id": "memsaga_" + "a" * 64,
            "status": "AWAITING_TARGET_PROJECTION",
            "target_document_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "target_projection_generation": 2,
            "target_projection_confirmed": False,
            "soft_forgotten_document_ids": [],
            "pending_document_ids": ["source-1"],
        }

    def propose_memory_consolidation(self, **kwargs: Any) -> dict[str, Any]:
        self.consolidation_proposal_calls.append(kwargs)
        document = _document_result(edit_summary="consolidation preview")
        source = kwargs["source_documents"][0]
        return {
            "proposal_id": "proposal-consolidation",
            "status": "PENDING",
            "document_uri": document["document_uri"],
            "document_id": document["document_id"],
            "document_kind": document["document_kind"],
            "relative_path": document["relative_path"],
            "source_digest": document["source_digest"],
            "proposed_source_digest": "b" * 64,
            "proposed_diff_digest": "c" * 64,
            "proposed_diff": "-target\n+target plus source",
            "edit_summary": "consolidation preview",
            "workflow_kind": "CONSOLIDATION",
            "consolidation_sources": [
                {
                    "document_uri": source["document_uri"],
                    "document_id": "source-1",
                    "relative_path": "knowledge/topics/source.md",
                    "source_digest": source["expected_digest"],
                    "size": 10,
                }
            ],
        }

    def forget(self, **kwargs: Any) -> dict[str, Any]:
        self.forget_calls.append(kwargs)
        return {**_document_result(edit_summary="soft forget"), "mode": "SOFT_FORGET", "recoverable": True}

    def list_memory_history(self, **kwargs: Any) -> dict[str, Any]:
        self.history_calls.append(kwargs)
        return {
            "document_uri": kwargs["document_uri"],
            "document_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "document_kind": "topic",
            "relative_path": "topics/distributed-systems.md",
            "revisions": [],
        }

    def restore_memory_revision(self, **kwargs: Any) -> dict[str, Any]:
        self.restore_calls.append(kwargs)
        return _document_result(edit_summary="restore retained document revision 1")

    def review_memory_edit(self, **kwargs: Any) -> dict[str, Any]:
        self.review_calls.append(kwargs)
        return {
            **_document_result(edit_summary="reviewed update"),
            "proposal_id": kwargs["proposal_id"],
            "status": "APPROVED",
            "proposed_source_digest": "b" * 64,
            "proposed_diff_digest": "c" * 64,
        }

    def health(self) -> dict[str, Any]:
        return self.health_payload


def _server(
    *,
    enable_action_tools: bool = False,
) -> tuple[MemoryOSMCPServer, FakeMCPClient]:
    client = FakeMCPClient()
    config = MCPServerConfig(
        root="/tmp/memory",
        user_id="u1",
        adapter_id="codex",
        agent_name="codex",
        enable_action_tools=enable_action_tools,
    )
    return MemoryOSMCPServer(cast(MemoryOSClient, client), config=config), client


def test_mcp_markdown_memory_commands_share_the_document_contract() -> None:
    server, client = _server()
    invalid = server.call_tool(
        "memoryos_remember",
        {
            "content": "PostgreSQL",
            "title": "unsupported field",
        },
    )
    assert invalid["error"]["code"] == "VALIDATION_ERROR"
    assert client.remember_calls == []

    remembered = server.call_tool(
        "memoryos_remember",
        {
            "content": "PostgreSQL",
            "target_hint": "topic:primary storage backend",
        },
    )
    assert remembered["error"] is None
    assert remembered["document_uri"].startswith("memoryos://user/u1/memory/documents/")
    assert client.remember_calls[0]["target_hint"] == "topic:primary storage backend"
    adopted = server.call_tool(
        "memoryos_adopt_memory_document",
        {
            "relative_path": "knowledge/topics/external.md",
            "expected_raw_sha256": "a" * 64,
        },
    )
    assert adopted["error"] is None
    assert client.adopt_calls == [
        {
            "relative_path": "knowledge/topics/external.md",
            "expected_raw_sha256": "a" * 64,
        }
    ]
    reviewed = server.call_tool(
        "memoryos_review_memory_edit",
        {
            "proposal_id": "proposal-1",
            "decision": "APPROVE",
        },
    )
    assert reviewed["error"] is None
    assert reviewed["status"] == "APPROVED"
    assert client.review_calls[0] == {"proposal_id": "proposal-1", "decision": "APPROVE"}
    rename_request = {
        "document_uri": remembered["document_uri"],
        "new_relative_path": "knowledge/topics/postgresql-renamed.md",
        "expected_digest": "a" * 64,
        "edit": "PostgreSQL renamed and edited",
    }
    renamed = server.call_tool("memoryos_rename_memory_document", rename_request)
    assert renamed["error"] is None
    assert renamed["edit_summary"] == "rename and edit memory document"
    assert client.rename_calls == [rename_request]
    proposal_request = {
        "target_document_uri": remembered["document_uri"],
        "merged_edit": "PostgreSQL plus exact source",
        "expected_target_digest": "a" * 64,
        "source_documents": [
            {
                "document_uri": remembered["document_uri"] + "-source",
                "expected_digest": "a" * 64,
            }
        ],
    }
    merged = server.call_tool("memoryos_merge_memory_documents", proposal_request)
    assert merged["error"] is None
    assert merged["status"] == "AWAITING_TARGET_PROJECTION"
    assert client.merge_calls == [proposal_request]
    proposed = server.call_tool(
        "memoryos_propose_memory_consolidation",
        proposal_request,
    )
    assert proposed["error"] is None
    assert proposed["status"] == "PENDING"
    assert proposed["workflow_kind"] == "CONSOLIDATION"
    assert client.consolidation_proposal_calls == [proposal_request]


def test_mcp_hard_erase_is_available_to_the_local_user() -> None:
    server, client = _server()
    result = server.call_tool(
        "memoryos_forget",
        {
            "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "mode": "HARD_ERASE",
            "expected_digest": "a" * 64,
        },
    )

    assert result["error"] is None
    assert client.forget_calls[0]["mode"] == "HARD_ERASE"


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
    assert result["contexts"][0]["uri"] == "memoryos://user/u1/memories/anchors/all"
    assert result["source_uris"] == ["memoryos://user/u1/memories/anchors/all"]
    connect = client.search_calls[0]["connect_metadata"]
    response_connect = result["metadata"]["connect"]
    assert connect == {"adapter_id": "codex"}
    assert response_connect["connect_type"] == "agent"
    assert response_connect["run_mode"] == "context_reduction"
    assert response_connect["source_kind"] == "coding_agent"
    assert response_connect["capabilities"]["can_predict_behavior"] is False


def test_mcp_search_context_source_kind_filter_is_explicit_only() -> None:
    server, client = _server()

    server.call_tool("memoryos_search_context", {"query": "MCP"})
    server.call_tool(
        "memoryos_search_context", {"query": "MCP", "connect_metadata": {"adapter_id": "codex", "source_kind": "chat"}}
    )
    server.call_tool("memoryos_assemble_context", {"query": "MCP"})
    server.call_tool(
        "memoryos_assemble_context",
        {"query": "MCP", "connect_metadata": {"adapter_id": "codex", "source_kind": "terminal"}},
    )

    assert client.search_calls[0]["connect_metadata"] == {"adapter_id": "codex"}
    assert client.search_calls[1]["connect_metadata"] == {"adapter_id": "codex", "source_kind": "chat"}
    assert client.assemble_calls[0]["connect_metadata"] == {"adapter_id": "codex"}
    assert client.assemble_calls[1]["connect_metadata"] == {"adapter_id": "codex", "source_kind": "terminal"}


def test_mcp_search_context_supports_multiple_context_types() -> None:
    server, client = _server()

    result = server.call_tool("memoryos_search_context", {"query": "MCP", "context_types": ["memory", "session"]})

    assert result["error"] is None
    assert len(client.search_calls) == 1
    assert client.search_calls[0]["context_type"] is None
    assert client.search_calls[0]["options"].context_types == (ContextType.MEMORY, ContextType.SESSION)


def test_mcp_search_context_passes_generic_workspace_scope_arguments() -> None:
    server, client = _server()

    result = server.call_tool(
        "memoryos_search_context",
        {
            "query": "shared context",
            "search_scope": "workspace_shared",
            "project_id": "memoryOS",
            "retrieval_views": ["project:memoryOS:shared"],
        },
    )

    assert result["error"] is None
    call = client.search_calls[0]
    assert call["search_scope"] == "workspace_shared"
    assert call["project_id"] == "memoryOS"
    assert call["retrieval_views"] == ["project:memoryOS:shared"]


def test_mcp_search_context_rejects_unknown_adapter_id() -> None:
    server, client = _server()

    result = server.call_tool(
        "memoryos_search_context", {"query": "MCP", "connect_metadata": {"adapter_id": "openclaw"}}
    )

    assert result["error"]["code"] == "VALIDATION_ERROR"
    assert client.search_calls == []


def test_mcp_config_allows_custom_adapter_from_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("MEMORYOS_ADAPTER_ID", "local_agent")
    monkeypatch.setenv("MEMORYOS_ALLOWED_ADAPTER_IDS", "team_agent")

    config = MCPServerConfig.from_env()

    assert "local_agent" in config.allowed_adapter_ids
    assert "team_agent" in config.allowed_adapter_ids
    assert config.adapter_id == "local_agent"


def test_mcp_commit_session_defaults_to_fast_queued_commit() -> None:
    server, client = _server()

    server.call_tool("memoryos_commit_session", {"session_id": "s1"})

    assert client.commit_calls[0]["async_commit"] is False


def test_mcp_agent_metadata_response_remains_context_reduction() -> None:
    server, client = _server()

    result = server.call_tool("memoryos_search_context", {"query": "MCP", "connect_metadata": {"adapter_id": "codex"}})
    connect = result["metadata"]["connect"]
    assert connect["connect_type"] == "agent"
    assert connect["run_mode"] == "context_reduction"
    assert connect["source_kind"] == "coding_agent"
    assert connect["capabilities"]["can_predict_behavior"] is False
    assert client.search_calls[0]["connect_metadata"] == {"adapter_id": "codex"}


def test_mcp_assemble_context_returns_selected_and_dropped_contexts() -> None:
    server, client = _server()

    result = server.call_tool("memoryos_assemble_context", {"query": "MCP"})

    assert result["error"] is None
    assert result["packed_context"] == "short context"
    assert result["dropped_contexts"] == [{"uri": "memoryos://ctx/2", "reason": "section_limit"}]
    assert "token_budget" not in client.assemble_calls[0]


def test_mcp_optional_int_rejects_bool_and_accepts_numbers() -> None:
    server, client = _server()

    bad_limit = server.call_tool("memoryos_search_context", {"query": "MCP", "limit": True})
    good_limit = server.call_tool("memoryos_search_context", {"query": "MCP", "limit": "2"})

    assert bad_limit["error"]["code"] == "VALIDATION_ERROR"
    assert good_limit["error"] is None
    assert client.search_calls[-1]["limit"] == 2


def test_mcp_commit_session_returns_structured_result() -> None:
    server, client = _server()

    result = server.call_tool(
        "memoryos_commit_session", {"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]}
    )

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


def test_mcp_health_and_memory_calls_do_not_report_ready_before_runtime_recovery() -> None:
    server, client = _server()
    client.health_payload = {"runtime": {"state": "NOT_READY", "ready": False, "reasons": ["receipt mismatch"]}}

    health = server.call_tool("memoryos_health", {})

    assert health["status"] == "not_ready"
    assert health["storage_ready"] is False
    assert health["contextdb_ready"] is False
    assert health["client_ready"] is False

    client.search_context = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeNotReadyError(RuntimeReadinessState.NOT_READY, ("receipt mismatch",))
    )
    blocked = server.call_tool("memoryos_search_context", {"query": "memory"})
    assert blocked["error"]["code"] == "NOT_READY"
    assert blocked["error"]["retryable"] is True


def test_mcp_health_requires_an_explicit_runtime_readiness_proof() -> None:
    server, client = _server()
    client.health = lambda: {}  # type: ignore[method-assign]

    health = server.call_tool("memoryos_health", {})

    assert health["status"] == "not_ready"
    assert health["storage_ready"] is False
    assert health["contextdb_ready"] is False
    assert health["client_ready"] is False


def test_mcp_health_propagates_degraded_derived_state_even_when_runtime_is_ready() -> None:
    server, client = _server()
    client.health_payload = {
        "status": "degraded",
        "runtime": {"state": "READY", "ready": True, "reasons": []},
        "queue": {"dead_letter": 1},
    }

    health = server.call_tool("memoryos_health", {})

    assert health["status"] == "degraded"
    assert health["storage_ready"] is False
    assert health["contextdb_ready"] is False
    assert health["client_ready"] is False


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

    missing = server.call_tool(
        "memoryos_predict",
        {"request": {"user_id": "u1", "episode_id": "s1", "observation": "hot", "available_actions": ["turn_on_ac"]}},
    )
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
    allowed = server.call_tool(
        "memoryos_process_observation",
        {"request": _request(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())},
    )

    assert denied["error"]["code"] == "PERMISSION_DENIED"
    assert allowed["error"] is None
    assert client.process_calls == 1


def test_action_tools_reject_string_false_capabilities() -> None:
    server, client = _server(enable_action_tools=True)
    predict_metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()
    predict_metadata["capabilities"]["can_predict_behavior"] = "false"
    process_metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()
    process_metadata["capabilities"]["can_execute_action"] = "false"

    predict = server.call_tool("memoryos_predict", {"request": _request(predict_metadata)})
    process = server.call_tool("memoryos_process_observation", {"request": _request(process_metadata)})

    assert predict["error"]["code"] == "VALIDATION_ERROR"
    assert process["error"]["code"] == "VALIDATION_ERROR"
    assert client.predict_calls == 0
    assert client.process_calls == 0


def test_action_tool_request_payload_schema_errors_are_validation_errors() -> None:
    server, client = _server(enable_action_tools=True)
    metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()

    unknown = server.call_tool(
        "memoryos_predict", {"request": {**_request(metadata), "unknown": "/Users/gulf token=abc"}}
    )
    missing = server.call_tool(
        "memoryos_predict",
        {
            "request": {
                "episode_id": "s1",
                "observation": "hot",
                "available_actions": ["turn_on_ac"],
                "connect_metadata": metadata,
            }
        },
    )
    bad_policy = server.call_tool("memoryos_predict", {"request": _request(metadata), "policies": ["not-object"]})

    assert unknown["error"]["code"] == "VALIDATION_ERROR"
    assert missing["error"]["code"] == "VALIDATION_ERROR"
    assert bad_policy["error"]["code"] == "VALIDATION_ERROR"
    assert "/Users/gulf" not in unknown["error"]["message"]
    assert "abc" not in unknown["error"]["message"]
    assert client.predict_calls == 0


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
                    "arguments": {
                        "request": _request(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())
                    },
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


def test_stdio_internal_exception_is_redacted(monkeypatch) -> None:  # noqa: ANN001
    server, _client = _server()

    def boom(name: str, arguments: dict | None = None) -> dict:  # noqa: ARG001
        raise RuntimeError("boom /Users/gulf password=secret Bearer sk-test")

    monkeypatch.setattr(server, "call_tool", boom)
    response = stdio._handle_jsonrpc(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "memoryos_health", "arguments": {}},
            }
        ),
    )

    assert response["id"] == 9
    assert response["error"]["code"] == -32603
    assert response["error"]["message"] == "Internal error"
    assert "/Users" not in json.dumps(response)
    assert "password" not in json.dumps(response)
    assert "Bearer" not in json.dumps(response)
