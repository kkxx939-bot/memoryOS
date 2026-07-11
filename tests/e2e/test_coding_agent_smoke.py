from __future__ import annotations

import asyncio
import json
from typing import Any

from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.tools import MCPToolRouter
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend


async def _request(
    app: MemoryOSASGI,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    query_string: bytes = b"",
) -> dict[str, Any]:
    sent: list[dict[str, Any]] = []
    body = json.dumps(payload or {}).encode()
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {"type": "http", "method": method, "path": path, "headers": [], "query_string": query_string},
        receive,
        send,
    )
    assert sent[0]["status"] == 200
    return json.loads(sent[1]["body"])


def test_agent_session_to_memory_http_mcp_smoke(tmp_path) -> None:  # noqa: ANN001
    provider = FakeMemoryModelProvider(
        response=json.dumps(
            {
                "candidates": [
                    {
                        "proposal_id": "p-sqlite-queue",
                        "memory_type": "project_decision",
                        "identity_fields": {"decision_topic": "local queue"},
                        "value_fields": {"canonical_value": "SQLite"},
                        "semantic": {
                            "speech_act": "confirmation",
                            "commitment": "confirmed",
                            "temporal_scope": "current",
                            "relation_to_existing": "unrelated",
                        },
                        "epistemic_status": "EXPLICIT",
                        "suggested_scope_refs": [{"namespace": "memoryos", "kind": "workspace", "id": "project-a"}],
                        "evidence_refs": [{"event_id": "event-1"}],
                        "field_evidence_refs": {
                            "identity.decision_topic": [{"event_id": "event-1"}],
                            "value.canonical_value": [{"event_id": "event-1"}],
                            "semantic.speech_act": [{"event_id": "event-1"}],
                            "semantic.commitment": [{"event_id": "event-1"}],
                            "semantic.temporal_scope": [{"event_id": "event-1"}],
                            "semantic.relation_to_existing": [{"event_id": "event-1"}],
                            "transition": [{"event_id": "event-1"}],
                        },
                        "confidence": 0.95,
                        "source_role": "user",
                    }
                ]
            }
        )
    )
    client = MemoryOSClient(
        str(tmp_path),
        mode="server",
        memory_extractor=LLMMemoryExtractorBackend(provider),
    )
    app = MemoryOSASGI(client)

    archived = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/sessions/events",
            {
                "event_id": "event-1",
                "event_type": "PROMPT_SUBMIT",
                "adapter_id": "claude_code",
                "user_id": "u1",
                "project_id": "project-a",
                "session_id": "native-1",
                "prompt": "We decided to use SQLite for the local queue.",
            },
        )
    )
    finalized = asyncio.run(
        _request(app, "POST", f"/v1/sessions/{archived['session_key']}/finalize", {"async_commit": True})
    )
    recalled = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/context/assemble",
            {
                "query": "SQLite queue",
                "user_id": "u1",
                "project_id": "project-a",
                "search_scope": "project_decisions",
                "token_budget": 200,
            },
        )
    )
    archive_search = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/archives/search",
            {"query": "SQLite", "user_id": "u1"},
        )
    )
    archive_read = asyncio.run(
        _request(
            app,
            "GET",
            "/v1/archives/read",
            query_string=f"archive_uri={finalized['archive_uri']}".encode(),
        )
    )
    router = MCPToolRouter(
        client,
        MCPServerConfig(root=str(tmp_path), user_id="u1", adapter_id="codex"),
    )
    mcp_search = router.call(
        "memoryos_search_context",
        {
            "query": "SQLite queue",
            "project_id": "project-a",
            "search_scope": "project_decisions",
        },
    )
    health = router.call("memoryos_health", {})
    direct = client.search_context(
        "SQLite queue",
        user_id="u1",
        project_id="project-a",
        search_scope="project_decisions",
    )

    assert finalized["done"] is True
    assert direct, [obj.to_dict() for obj in client.source_store.list_objects()]
    assert "SQLite" in recalled["packed_context"], recalled
    assert recalled["trace_id"]
    assert archive_search["results"]
    assert archive_read["archive"]["archive_uri"] == finalized["archive_uri"]
    assert mcp_search["error"] is None and mcp_search["results"]
    assert health["error"] is None and health["http_server"] == "ready"
