from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from memoryos.api.http.app import _bound_payload, handle
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.schemas import TOOL_INPUT_SCHEMAS
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.retrieval_contract import retrieval_options_json_schema
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.sdk.http_client import HTTPMemoryOSClient
from memoryos.api.trusted_context import READ_CONTEXT, TrustedRequestContext
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.orchestrator import RetrievalMetrics, UnifiedRetrievalResult
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent


class _RecordingTransportClient:
    tenant_id = "default"

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.assemble_calls: list[dict[str, Any]] = []
        self.last_recall_trace_id = "trace-1"

    def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, **kwargs})
        return [{"uri": "memoryos://context/1", "source_uri": "memoryos://source/1"}]

    def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.assemble_calls.append({"query": query, **kwargs})
        options = kwargs.get("options")
        return {
            "packed_context": "bounded context",
            "contexts": [{"uri": "memoryos://context/1"}],
            "source_uris": ["memoryos://source/1"],
            "dropped_contexts": [],
            "total_budget": options.token_budget if isinstance(options, RetrievalOptions) else 2000,
            "query_plan": {"semantic_query": query},
            "metrics": {"structured_candidates": 1},
        }


def _caller() -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=frozenset({READ_CONTEXT}),
        allowed_workspace_ids=frozenset({"project-a"}),
    )


def test_http_and_mcp_publish_the_same_structured_options_schema() -> None:
    expected = retrieval_options_json_schema()

    assert TOOL_INPUT_SCHEMAS["memoryos_search_context"]["properties"]["options"] == expected
    assert TOOL_INPUT_SCHEMAS["memoryos_assemble_context"]["properties"]["options"] == expected
    assert TOOL_INPUT_SCHEMAS["memoryos_search"] is TOOL_INPUT_SCHEMAS["memoryos_search_context"]
    assert TOOL_INPUT_SCHEMAS["memoryos_assemble"] is TOOL_INPUT_SCHEMAS["memoryos_assemble_context"]


def test_http_handle_parses_options_for_search_and_assemble() -> None:
    client = _RecordingTransportClient()
    payload = {
        "query": "tool result",
        "options": {
            "context_types": ["session", "resource"],
            "target_paths": ["timeline/2026/07/14"],
            "query_intent": "OPEN_RECALL",
            "candidate_limit": 40,
            "final_limit": 7,
            "token_budget": 512,
        },
    }

    searched = handle("POST /context/search", cast(MemoryOSClient, client), payload)
    assembled = handle("POST /context/assemble", cast(MemoryOSClient, client), payload)

    assert searched["results"]
    assert assembled["packed_context"] == "bounded context"
    for call in (client.search_calls[0], client.assemble_calls[0]):
        options = call["options"]
        assert isinstance(options, RetrievalOptions)
        assert options.context_types == (ContextType.SESSION, ContextType.RESOURCE)
        assert options.target_paths == ("timeline/2026/07/14",)
        assert options.query_intent is RetrievalQueryIntent.OPEN_RECALL


@pytest.mark.parametrize(
    "options",
    [
        {"tenant_id": "other"},
        {"owner_user_id": "u2"},
        {"workspace_ids": ["project-b"]},
        {"adapter_id": "other-agent"},
    ],
)
def test_http_rejects_nested_options_that_expand_trusted_scope(options: dict[str, Any]) -> None:
    with pytest.raises(PermissionError):
        _bound_payload({"query": "private", "options": options}, _caller())


@pytest.mark.parametrize(
    "scope_key",
    [
        "memoryos:principal:victim",
        "memoryos:team:administrators",
        "memoryos:workspace:project-b",
    ],
)
def test_http_rejects_metadata_scope_keys_outside_trusted_grants(scope_key: str) -> None:
    with pytest.raises(PermissionError, match="exceed trusted caller grants"):
        _bound_payload(
            {
                "query": "private",
                "options": {
                    "workspace_ids": ["project-a"],
                    "metadata_filters": {"applicability_scope_keys": [scope_key]},
                },
            },
            _caller(),
        )


def test_mcp_uses_one_call_and_passes_structured_options() -> None:
    client = _RecordingTransportClient()
    server = MemoryOSMCPServer(
        cast(MemoryOSClient, client),
        config=MCPServerConfig(
            root="/tmp/memoryos-api-contract",
            user_id="u1",
            tenant_id="default",
            adapter_id="codex",
            actor_id="codex",
            allowed_workspace_ids=frozenset({"project-a"}),
        ),
    )

    result = server.call_tool(
        "memoryos_search_context",
        {
            "query": "desktop file",
            "options": {
                "tenant_id": "default",
                "owner_user_id": "u1",
                "workspace_ids": ["project-a"],
                "context_types": ["session", "resource"],
                "target_paths": ["resources/desktop"],
                "query_intent": "OPEN_RECALL",
                "candidate_limit": 30,
                "final_limit": 5,
            },
        },
    )

    assert result["error"] is None
    assert result["trace_id"] == "trace-1"
    assert len(client.search_calls) == 1
    options = client.search_calls[0]["options"]
    assert options.context_types == (ContextType.SESSION, ContextType.RESOURCE)
    assert options.target_paths == ("resources/desktop",)


def test_mcp_short_names_are_same_handler_compatibility_aliases() -> None:
    client = _RecordingTransportClient()
    server = MemoryOSMCPServer(
        cast(MemoryOSClient, client),
        config=MCPServerConfig(
            root="/tmp/memoryos-api-contract-aliases",
            user_id="u1",
            tenant_id="default",
            adapter_id="codex",
            actor_id="codex",
            allowed_workspace_ids=frozenset({"project-a"}),
        ),
    )

    searched = server.call_tool("memoryos_search", {"query": "desktop file"})
    assembled = server.call_tool("memoryos_assemble", {"query": "desktop file", "token_budget": 64})

    assert searched["error"] is None
    assert assembled["error"] is None
    assert len(client.search_calls) == 1
    assert len(client.assemble_calls) == 1


def test_mcp_rejects_metadata_scope_key_escalation_before_client_call() -> None:
    client = _RecordingTransportClient()
    server = MemoryOSMCPServer(
        cast(MemoryOSClient, client),
        config=MCPServerConfig(
            root="/tmp/memoryos-api-contract",
            user_id="u1",
            tenant_id="default",
            adapter_id="codex",
            actor_id="codex",
            allowed_workspace_ids=frozenset({"project-a"}),
        ),
    )

    result = server.call_tool(
        "memoryos_search_context",
        {
            "query": "private",
            "options": {
                "workspace_ids": ["project-a"],
                "metadata_filters": {"applicability_scope_keys": ["memoryos:team:administrators"]},
            },
        },
    )

    assert result["error"]["code"] == "PERMISSION_DENIED"
    assert client.search_calls == []


def test_remote_sdk_serializes_retrieval_options() -> None:
    class RecordingHTTPClient(HTTPMemoryOSClient):
        def __init__(self) -> None:
            super().__init__("http://memoryos.test", retries=0)
            self.payloads: list[dict[str, Any]] = []

        def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            self.payloads.append({"method": method, "path": path, "payload": dict(payload or {})})
            return {"results": [], "contexts": []}

    client = RecordingHTTPClient()
    options = RetrievalOptions(
        context_types=(ContextType.SESSION,),
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        candidate_limit=25,
        final_limit=5,
    )

    client.search_context("history", options=options)
    client.assemble_context("history", options=options)

    assert [entry["path"] for entry in client.payloads] == ["/v1/context/search", "/v1/context/assemble"]
    assert all(entry["payload"]["options"] == options.to_dict() for entry in client.payloads)


def test_local_sdk_builds_a_plan_and_calls_the_unified_orchestrator(tmp_path: Path, monkeypatch: Any) -> None:
    client = MemoryOSClient(str(tmp_path))
    captured: list[Any] = []

    class Orchestrator:
        def execute(self, plan: Any) -> UnifiedRetrievalResult:
            captured.append(plan)
            return UnifiedRetrievalResult(
                plan=plan,
                contexts=(
                    {
                        "uri": "memoryos://context/1",
                        "source_uri": "memoryos://source/1",
                        "content": "desktop report.txt",
                        "selected_layer": "L1",
                    },
                ),
                dropped_contexts=(),
                load_plan=(),
                metrics=RetrievalMetrics(selected_count=1),
                total_budget=plan.token_budget,
                used_tokens=4,
                remaining_tokens=plan.token_budget - 4,
            )

    monkeypatch.setattr(client, "_retrieval_orchestrator", lambda: Orchestrator())
    results = client.search_context(
        "report",
        options=RetrievalOptions(
            owner_user_id="u1",
            target_paths=("resources/desktop",),
            context_types=(ContextType.SESSION,),
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            candidate_limit=20,
            final_limit=4,
        ),
    )

    assert results[0]["source_uri"] == "memoryos://source/1"
    assert captured[0].semantic_query == "report"
    assert captured[0].target_paths == ("resources/desktop",)
    assert captured[0].owner_user_id == "u1"
    assert client.last_recall_trace_id


def test_archive_search_is_a_unified_retrieval_wrapper(tmp_path: Path, monkeypatch: Any) -> None:
    client = MemoryOSClient(str(tmp_path))
    calls: list[dict[str, Any]] = []

    def search(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"query": query, **kwargs})
        return [
            {
                "uri": "memoryos://context/session-1/tool-1",
                "source_uri": "memoryos://user/u1/sessions/history/session-1",
                "content": "report.txt",
                "metadata": {"session_id": "session-1"},
            }
        ]

    monkeypatch.setattr(client, "search_context", search)
    monkeypatch.setattr(
        client,
        "archive_read",
        lambda archive_uri, **kwargs: {
            "archive": {"archive_uri": archive_uri, "session_id": "session-1"},
            "messages": [{"role": "tool", "content": "report.txt"}],
            "tool_results": [],
        },
    )
    results = client.archive_search(
        "report",
        user_id="u1",
        tenant_id="default",
        timezone_name="Asia/Singapore",
    )

    assert len(calls) == 1
    assert calls[0]["options"].context_types == (ContextType.SESSION, ContextType.MEMORY)
    assert calls[0]["options"].query_intent is RetrievalQueryIntent.OPEN_RECALL
    assert calls[0]["options"].timezone == "Asia/Singapore"
    assert calls[0]["options"].metadata_filters == {}
    assert results == [
        {
            "uri": "memoryos://context/session-1/tool-1",
            "source_uri": "memoryos://user/u1/sessions/history/session-1",
            "content": "report.txt",
            "metadata": {"session_id": "session-1"},
            "archive_uri": "memoryos://user/u1/sessions/history/session-1",
            "session_id": "session-1",
            "preview": "report.txt",
        }
    ]


def test_archive_search_sanitizes_secret_and_absolute_path_in_preview(tmp_path: Path, monkeypatch: Any) -> None:
    client = MemoryOSClient(str(tmp_path))
    monkeypatch.setattr(
        client,
        "search_context",
        lambda query, **kwargs: [
            {
                "uri": "memoryos://context/session-1/tool-1",
                "source_uri": "memoryos://user/u1/sessions/history/session-1",
                "content": "budget.xlsx API_KEY=<redacted> <home>/Desktop/budget.xlsx",
                "metadata": {"session_id": "session-1"},
            }
        ],
    )
    monkeypatch.setattr(
        client,
        "archive_read",
        lambda archive_uri, **kwargs: {
            "archive": {"archive_uri": archive_uri, "session_id": "session-1"},
            "messages": [],
            "tool_results": [
                {
                    "content": "budget.xlsx API_KEY=top-secret /Users/u1/Desktop/budget.xlsx",
                }
            ],
        },
    )

    preview = client.archive_search("budget.xlsx", user_id="u1", tenant_id="default")[0]["preview"]

    assert "budget.xlsx" in preview
    assert "top-secret" not in preview
    assert "/Users/u1" not in preview
    assert "<redacted>" in preview


def test_archive_search_does_not_repeat_lexical_matching_over_full_archive(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    monkeypatch.setattr(
        client,
        "search_context",
        lambda query, **kwargs: [
            {
                "uri": "memoryos://context/session-1/semantic-1",
                "source_uri": "memoryos://user/u1/sessions/history/session-1",
                "content": "sanitized semantic catalog match",
                "metadata": {"session_id": "session-1"},
            }
        ],
    )
    monkeypatch.setattr(
        client,
        "archive_read",
        lambda archive_uri, **kwargs: {
            "archive": {"archive_uri": archive_uri, "session_id": "session-1"},
            "messages": [{"role": "user", "content": "raw evidence does not repeat the query token"}],
            "tool_results": [],
        },
    )

    results = client.archive_search("different semantic query", user_id="u1", tenant_id="default")

    assert len(results) == 1
    assert results[0]["preview"] == "sanitized semantic catalog match"
