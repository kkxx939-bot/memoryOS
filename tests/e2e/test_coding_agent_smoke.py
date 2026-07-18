from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode

from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.tools import MCPToolRouter
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    READ_CONTEXT,
    TrustedRequestContext,
)


async def _request(
    app: MemoryOSASGI,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    query_string: bytes = b"",
) -> dict[str, Any]:
    sent: list[dict[str, Any]] = []
    incoming = iter(
        [{"type": "http.request", "body": json.dumps(payload or {}).encode(), "more_body": False}]
    )

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"authorization", b"Bearer test-token")],
            "query_string": query_string,
        },
        receive,
        send,
    )
    assert sent[0]["status"] == 200, sent
    return json.loads(sent[1]["body"])


def test_markdown_memory_http_sdk_mcp_smoke(tmp_path) -> None:  # noqa: ANN001
    capabilities = frozenset({READ_CONTEXT, AUTHORITATIVE_REMEMBER, AUTHORITATIVE_FORGET})
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="user",
        actor_id="u1",
        capabilities=capabilities,
    )
    client = MemoryOSClient(str(tmp_path), mode="server")
    app = MemoryOSASGI(
        client,
        api_token="test-token",
        trusted_context=caller,
    )

    remembered = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/memories/remember",
            {
                "content": "Use SQLite for the local queue.",
                "target_hint": "topic:local queue",
            },
        )
    )
    edited = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/memories/edit",
            {
                "document_uri": remembered["document_uri"],
                "edit": "Use SQLite WAL mode for the local queue.",
                "expected_digest": remembered["source_digest"],
            },
        )
    )
    history = asyncio.run(
        _request(
            app,
            "GET",
            "/v1/memories/history",
            query_string=urlencode({"document_uri": remembered["document_uri"]}).encode(),
        )
    )

    router = MCPToolRouter(
        client,
        MCPServerConfig(
            root=str(tmp_path),
            user_id="u1",
            actor_kind="user",
            actor_id="u1",
            capabilities=capabilities,
        ),
    )
    forgotten = router.call(
        "memoryos_forget",
        {
            "document_uri": remembered["document_uri"],
            "mode": "SOFT_FORGET",
            "expected_digest": edited["source_digest"],
        },
    )
    health = router.call("memoryos_health", {})

    assert edited["document_id"] == remembered["document_id"]
    assert edited["document_revision"] > remembered["document_revision"]
    assert history["document_uri"] == remembered["document_uri"]
    assert len(history["revisions"]) >= 2
    assert forgotten["error"] is None
    assert forgotten["mode"] == "SOFT_FORGET"
    assert forgotten["recoverable"] is True
    assert health["error"] is None
