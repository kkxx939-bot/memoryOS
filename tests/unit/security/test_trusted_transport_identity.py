from __future__ import annotations

import asyncio
import io
import json
import multiprocessing
import sys
import threading
import types
import urllib.error
import uuid
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from memoryos.adapters.agent_hooks.base import BaseAgentHookAdapter
from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.events import AgentHookEvent
from memoryos.adapters.agent_hooks.queue import PendingItem, PendingQueue
from memoryos.adapters.agent_hooks.session_service import AgentSessionService
from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.http.app import main as http_main
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.schemas import tool_definitions
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient, _scope_keys
from memoryos.api.sdk.http_client import HTTPMemoryOSClient, RemoteMemoryOSError
from memoryos.api.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    DEFAULT_AGENT_CAPABILITIES,
    READ_CONTEXT,
    TrustedRequestContext,
)
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.runtime.readiness import RuntimeReadinessState


def _caller(
    *,
    actor_kind: str = "agent",
    actor_id: str = "codex",
    capabilities: frozenset[str] = DEFAULT_AGENT_CAPABILITIES,
) -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind=actor_kind,
        actor_id=actor_id,
        capabilities=capabilities,
        allowed_workspace_ids=frozenset({"trusted-project"}),
    )


def _visible_metadata(*, workspace_id: str = "trusted-project", **extra: Any) -> dict[str, Any]:
    return {
        "scope": {
            "subject": {"namespace": "memoryos", "kind": "principal", "id": "u1"},
            "applicability": {"all_of": [{"namespace": "memoryos", "kind": "workspace", "id": workspace_id}]},
            "visibility": {
                "tenant_id": "default",
                "allowed_principal_ids": ["u1"],
            },
            "authority": {},
            "origin_refs": [],
        },
        **extra,
    }


def _live_event(event_id: str, *, transcript_path: str | None = None):  # noqa: ANN202
    return AgentHookEvent(
        event_id=event_id,
        agent_name="codex",
        adapter_id="codex",
        hook_name="after_turn",
        session_id="concurrent-session",
        user_id="u1",
        cwd=str(Path(transcript_path).parent) if transcript_path else None,
        repo_root=str(Path(transcript_path).parent) if transcript_path else None,
        messages=[{"role": "assistant", "content": event_id}],
        metadata={"project_id": "workspace-a", "transcript_path": transcript_path},
        tenant_id="default",
    ).normalize()


def _append_live_events_process(root: str, event_ids: list[str], start_event: Any) -> None:
    service = AgentSessionService(root)
    if not start_event.wait(15):
        raise RuntimeError("concurrent session test start timed out")
    for event_id in event_ids:
        service.append_event(_live_event(event_id))


async def _asgi_request(
    app: MemoryOSASGI,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    token: str = "secret",
    query_string: bytes = b"",
    client_host: str | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    incoming = iter(
        [
            {
                "type": "http.request",
                "body": json.dumps(payload or {}).encode(),
                "more_body": False,
            }
        ]
    )

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"authorization", f"Bearer {token}".encode()), *(extra_headers or [])],
        "query_string": query_string,
    }
    if client_host is not None:
        scope["client"] = (client_host, 12345)
    await app(
        scope,
        receive,
        send,
    )
    return int(sent[0]["status"]), json.loads(sent[1]["body"])


def test_http_bearer_binds_identity_capability_and_archived_actor(tmp_path: Path) -> None:
    caller = _caller()
    app = MemoryOSASGI(
        MemoryOSClient(str(tmp_path), mode="server"),
        api_token="secret",
        trusted_context=caller,
    )

    invalid_status, _ = asyncio.run(_asgi_request(app, "GET", "/health", token="wrong"))
    mismatch_status, _ = asyncio.run(_asgi_request(app, "POST", "/v1/context/search", {"query": "x", "user_id": "u2"}))
    remember_status, _ = asyncio.run(
        _asgi_request(app, "POST", "/v1/memories/remember", {"content": "x", "user_id": "u1"})
    )
    event_status, event_response = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/sessions/events",
            {
                "event_id": "e1",
                "event_type": "PROMPT_SUBMIT",
                "adapter_id": "codex",
                "user_id": "u1",
                "session_id": "s1",
                "metadata": {"project_id": "trusted-project"},
                "prompt": "claimed user prompt",
                "messages": [{"role": "user", "actor_id": "victim", "content": "forged"}],
            },
        )
    )

    assert invalid_status == 401
    assert mismatch_status == 403
    assert remember_status == 403
    assert event_status == 200
    archived = app.sessions.events(event_response["session_key"])[0]
    assert {message["role"] for message in archived["messages"]} == {"assistant"}
    assert {message["actor_id"] for message in archived["messages"]} == {"codex"}
    assert archived["metadata"]["tenant_id"] == "default"

    review_status, _ = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/memories/review",
            {"proposal_id": "proposal-1", "decision": "REJECT"},
        )
    )
    assert review_status == 403

    app.client.commit_agent_session(
        user_id="u2",
        session_id="other",
        messages=[{"role": "user", "content": "other"}],
        async_commit=False,
    )
    archive_status, _ = asyncio.run(
        _asgi_request(
            app,
            "GET",
            "/v1/archives/read",
            query_string=b"archive_uri=memoryos://user/u2/sessions/history/other",
        )
    )
    app.client.search_context("none", user_id="u2", tenant_id="default")
    trace_status, _ = asyncio.run(_asgi_request(app, "GET", f"/v1/recall-traces/{app.client.last_recall_trace_id}"))
    assert archive_status == 404
    assert trace_status == 404

    read_only_app = MemoryOSASGI(
        MemoryOSClient(str(tmp_path / "read-only"), mode="server"),
        api_token="secret",
        trusted_context=_caller(capabilities=frozenset({READ_CONTEXT})),
    )
    event_denied, _ = asyncio.run(
        _asgi_request(
            read_only_app,
            "POST",
            "/v1/sessions/events",
            {"adapter_id": "codex", "user_id": "u1", "session_id": "s2"},
        )
    )
    assert event_denied == 403


def test_http_without_token_requires_explicit_loopback_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = _caller()
    closed = MemoryOSASGI(MemoryOSClient(str(tmp_path / "closed")), trusted_context=caller)
    local = MemoryOSASGI(
        MemoryOSClient(str(tmp_path / "local")),
        trusted_context=caller,
        allow_unauthenticated_local=True,
    )

    closed_status, _ = asyncio.run(_asgi_request(closed, "GET", "/health", client_host="127.0.0.1"))
    missing_client_status, _ = asyncio.run(_asgi_request(local, "GET", "/health"))
    remote_status, _ = asyncio.run(
        _asgi_request(
            local,
            "GET",
            "/health",
            client_host="203.0.113.5",
            extra_headers=[(b"x-forwarded-for", b"127.0.0.1")],
        )
    )
    ipv4_status, _ = asyncio.run(_asgi_request(local, "GET", "/health", client_host="127.0.0.1"))
    ipv6_status, _ = asyncio.run(_asgi_request(local, "GET", "/health", client_host="::1"))

    assert (closed_status, missing_client_status, remote_status) == (401, 401, 401)
    assert (ipv4_status, ipv6_status) == (200, 200)

    monkeypatch.delenv("MEMORYOS_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="required when binding HTTP to a non-loopback host"):
        http_main(["--host", "0.0.0.0", "--root", str(tmp_path / "server")])

    started: dict[str, Any] = {}
    fake_uvicorn = types.ModuleType("uvicorn")

    def run(app: MemoryOSASGI, *, host: str, port: int) -> None:
        started.update({"app": app, "host": host, "port": port})

    fake_uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    http_main(["--host", "127.0.0.1", "--port", "9876", "--root", str(tmp_path / "loopback-server")])
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 9876
    assert started["app"].allow_unauthenticated_local is True


def test_http_query_routes_preserve_trusted_caller_at_sdk_boundary(tmp_path: Path) -> None:
    class RecordingClient(MemoryOSClient):
        def __init__(self, root: str) -> None:
            super().__init__(root, mode="server")
            self.query_callers: list[TrustedRequestContext | None] = []

        def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            self.query_callers.append(kwargs.get("caller"))
            return []

        def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
            self.query_callers.append(kwargs.get("caller"))
            return {"contexts": [], "source_uris": [], "packed_context": ""}

    client = RecordingClient(str(tmp_path))
    caller = _caller()
    app = MemoryOSASGI(client, api_token="secret", trusted_context=caller)

    for path in ("/v1/context/search", "/v1/context/assemble"):
        status, _ = asyncio.run(
            _asgi_request(
                app,
                "POST",
                path,
                {"query": "trusted", "project_id": "trusted-project"},
            )
        )
        assert status == 200

    assert client.query_callers == [caller, caller]

    spoofed_status, _ = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/context/search",
            {
                "query": "private",
                "project_id": "trusted-project",
                "connect_metadata": {"adapter_id": "cursor"},
            },
        )
    )
    assert spoofed_status == 403


@pytest.mark.parametrize(("actor_kind", "actor_id"), [("user", "u1"), ("service", "service-a")])
def test_http_user_and_service_connect_identity_is_statically_bound(
    tmp_path: Path,
    actor_kind: str,
    actor_id: str,
) -> None:
    class RecordingClient(MemoryOSClient):
        def __init__(self, root: str) -> None:
            super().__init__(root, mode="server")
            self.connect_metadata: dict[str, Any] = {}

        def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            self.connect_metadata = dict(kwargs.get("connect_metadata") or {})
            return []

    client = RecordingClient(str(tmp_path / actor_kind))
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind=actor_kind,
        actor_id=actor_id,
        capabilities=frozenset({READ_CONTEXT}),
        allowed_workspace_ids=frozenset({"trusted-project"}),
    )
    app = MemoryOSASGI(client, api_token="secret", trusted_context=caller)

    rejected_status, _ = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/context/search",
            {
                "query": "private",
                "project_id": "trusted-project",
                "connect_metadata": {"adapter_id": "forged-adapter"},
            },
        )
    )
    accepted_status, _ = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/context/search",
            {
                "query": "private",
                "project_id": "trusted-project",
                "connect_metadata": {
                    "adapter_id": actor_id,
                    "connect_type": "embodied",
                    "run_mode": "action_capable",
                    "world_domain": "physical",
                    "source_kind": "robot",
                    "capabilities": {"can_execute_action": True},
                },
            },
        )
    )

    assert rejected_status == 403
    assert accepted_status == 200
    assert client.connect_metadata["adapter_id"] == actor_id
    assert client.connect_metadata["connect_type"] == "agent"
    assert client.connect_metadata["run_mode"] == "context_reduction"
    assert client.connect_metadata["world_domain"] == "digital"
    assert client.connect_metadata["source_kind"] == "coding_agent"
    assert client.connect_metadata["capabilities"]["can_execute_action"] is False


def test_http_trusted_user_remember_binds_authenticated_document_owner(tmp_path: Path) -> None:
    class RecordingClient(MemoryOSClient):
        def __init__(self, root: str) -> None:
            super().__init__(root, mode="server")
            self.remember_call: dict[str, Any] = {}

        def remember(self, **kwargs: Any) -> dict[str, Any]:
            self.remember_call = dict(kwargs)
            return {
                "document_uri": "memoryos://user/u1/memory/documents/document-1",
                "document_id": "document-1",
                "document_kind": "topic",
                "relative_path": "topics/trusted.md",
                "document_revision": 1,
                "source_digest": "a" * 64,
                "changed": True,
                "edit_summary": "explicit remember",
                "projection_status": "ENQUEUED",
            }

    client = RecordingClient(str(tmp_path))
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="user",
        actor_id="u1",
        capabilities=frozenset({AUTHORITATIVE_REMEMBER}),
        allowed_workspace_ids=frozenset({"trusted-project"}),
    )
    app = MemoryOSASGI(client, api_token="secret", trusted_context=caller)

    status, _ = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/memories/remember",
            {
                "content": "trusted",
                "target_hint": "topic:trusted",
            },
        )
    )

    assert status == 200
    assert client.remember_call["caller"] == caller
    assert client.remember_call["tenant_id"] == "default"
    assert client.remember_call["target_hint"] == "topic:trusted"


def test_http_remember_rejects_removed_fields_before_client_call(tmp_path: Path) -> None:
    class RecordingClient(MemoryOSClient):
        def __init__(self, root: str) -> None:
            super().__init__(root, mode="server")
            self.calls = 0

        def remember(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return {"status": "COMMITTED"}

    client = RecordingClient(str(tmp_path))
    caller = _caller(
        actor_kind="user",
        actor_id="u1",
        capabilities=frozenset({AUTHORITATIVE_REMEMBER}),
    )
    app = MemoryOSASGI(client, api_token="secret", trusted_context=caller)

    status, body = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/memories/remember",
            {
                "content": "PostgreSQL",
                "title": "primary storage backend",
            },
        )
    )

    assert status == 400
    assert body["error"]["code"] == "BAD_REQUEST"
    assert client.calls == 0


def test_http_memory_api_returns_503_until_runtime_ready(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), mode="server")
    client.readiness.transition(
        RuntimeReadinessState.NOT_READY,
        reasons=("receipt history integrity failed",),
    )
    app = MemoryOSASGI(client, api_token="secret", trusted_context=_caller())

    status, body = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/context/search",
            {
                "query": "memory",
                "user_id": "u1",
                "tenant_id": "default",
                "project_id": "trusted-project",
            },
        )
    )
    before_session_files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    event_status, event_body = asyncio.run(
        _asgi_request(
            app,
            "POST",
            "/v1/sessions/events",
            {
                "event_id": "not-ready-event",
                "event_type": "PROMPT_SUBMIT",
                "adapter_id": "codex",
                "user_id": "u1",
                "session_id": "not-ready-session",
                "metadata": {"project_id": "trusted-project"},
                "prompt": "must not be archived while recovery is incomplete",
            },
        )
    )
    health_status, health = asyncio.run(_asgi_request(app, "GET", "/health"))

    assert status == 503
    assert body["error"]["code"] == "NOT_READY"
    assert event_status == 503
    assert event_body["error"]["code"] == "NOT_READY"
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()} == before_session_files
    assert health_status == 200
    assert health["status"] == "not_ready"
    assert health["source_store"] == "not_ready"
    assert health["index_store"] == "not_ready"
    assert health["queue_store"] == "not_ready"
    assert health["runtime"]["state"] == "NOT_READY"
    assert health["runtime"]["ready"] is False


def test_mcp_agent_cannot_select_another_allowed_adapter_identity() -> None:
    client = _MCPClient()
    server = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root="/tmp/memory",
            user_id="u1",
            adapter_id="codex",
            allowed_workspace_ids=frozenset({"trusted-project"}),
        ),
    )

    result = server.call_tool(
        "memoryos_search_context",
        {
            "query": "private",
            "project_id": "trusted-project",
            "search_scope": "agent_private",
            "connect_metadata": {"adapter_id": "cursor"},
        },
    )

    assert result["error"]["code"] == "VALIDATION_ERROR"
    assert client.calls == []


class _MCPClient:
    def __init__(self, *, forget_error: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.forget_error = forget_error

    def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("search", {"query": query, **kwargs}))
        return []

    def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("assemble", {"query": query, **kwargs}))
        return {}

    def commit_agent_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("commit", kwargs))
        return {"status": "accepted"}

    def forget(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("forget", kwargs))
        if self.forget_error:
            return {"error": {"code": "HTTP_ERROR", "message": "HTTP 409"}}
        return {"document_uri": kwargs["document_uri"], "mode": "SOFT_FORGET"}


@pytest.mark.parametrize(("actor_kind", "actor_id"), [("user", "u1"), ("service", "service-a")])
def test_mcp_user_and_service_cannot_select_another_allowed_adapter(
    actor_kind: str,
    actor_id: str,
) -> None:
    client = _MCPClient()
    server = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root="/tmp/memory",
            user_id="u1",
            actor_kind=actor_kind,
            actor_id=actor_id,
            adapter_id="codex",
            capabilities=frozenset({READ_CONTEXT}),
            allowed_workspace_ids=frozenset({"trusted-project"}),
        ),
    )

    rejected = server.call_tool(
        "memoryos_search_context",
        {
            "query": "private",
            "project_id": "trusted-project",
            "search_scope": "agent_private",
            "connect_metadata": {"adapter_id": "cursor"},
        },
    )
    accepted = server.call_tool(
        "memoryos_search_context",
        {
            "query": "private",
            "project_id": "trusted-project",
            "search_scope": "agent_private",
            "connect_metadata": {
                "adapter_id": "codex",
                "connect_type": "embodied",
                "capabilities": {"can_execute_action": True},
            },
        },
    )

    assert rejected["error"]["code"] == "VALIDATION_ERROR"
    assert accepted["error"] is None
    assert accepted["metadata"]["connect"]["adapter_id"] == "codex"
    assert accepted["metadata"]["connect"]["connect_type"] == "agent"
    assert accepted["metadata"]["connect"]["capabilities"]["can_execute_action"] is False
    assert len(client.calls) == 1


def test_mcp_defaults_hide_and_reject_authoritative_tools_and_identity_override() -> None:
    client = _MCPClient()
    config = MCPServerConfig(root="/tmp/memory", user_id="u1", tenant_id="default", adapter_id="codex")
    server = MemoryOSMCPServer(client, config=config)
    names = {item["name"] for item in tool_definitions(config)}

    assert "memoryos_remember" not in names
    assert "memoryos_forget" not in names
    assert "memoryos_merge_memory_documents" not in names
    assert server.call_tool("memoryos_remember", {"content": "x"})["error"]["code"] == "PERMISSION_DENIED"
    assert (
        server.call_tool("memoryos_commit_session", {"session_id": "s1", "user_id": "u2"})["error"]["code"]
        == "PERMISSION_DENIED"
    )
    assert (
        server.call_tool(
            "memoryos_archive_read",
            {"archive_uri": "memoryos://user/u2/sessions/history/s1"},
        )["error"]["code"]
        == "STORAGE_ERROR"
    )
    assert client.calls == []


def test_mcp_trusted_user_capability_is_static_and_preserves_remote_error() -> None:
    capabilities = DEFAULT_AGENT_CAPABILITIES | frozenset({AUTHORITATIVE_REMEMBER, AUTHORITATIVE_FORGET})
    config = MCPServerConfig(
        root="/tmp/memory",
        user_id="u1",
        actor_kind="user",
        actor_id="u1",
        capabilities=capabilities,
    )
    client = _MCPClient(forget_error=True)
    server = MemoryOSMCPServer(client, config=config)

    names = {item["name"] for item in tool_definitions(config)}
    result = server.call_tool(
        "memoryos_forget",
        {
            "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
        },
    )

    assert {
        "memoryos_remember",
        "memoryos_forget",
        "memoryos_merge_memory_documents",
        "memoryos_resume_memory_consolidation",
    }.issubset(names)
    assert result["error"]["code"] == "HTTP_ERROR"
    assert client.calls[0][1]["document_uri"].endswith("01ARZ3NDEKTSV4RRFFQ69G5FAV")


def test_authoritative_remember_capability_does_not_turn_an_agent_into_the_user() -> None:
    capabilities = DEFAULT_AGENT_CAPABILITIES | frozenset({AUTHORITATIVE_REMEMBER, AUTHORITATIVE_FORGET})
    config = MCPServerConfig(
        root="/tmp/memory",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=capabilities,
    )
    client = _MCPClient()
    server = MemoryOSMCPServer(client, config=config)

    assert server.call_tool("memoryos_remember", {"content": "forged"})["error"]["code"] == "PERMISSION_DENIED"
    assert client.calls == []




def test_shared_session_sanitizes_reserved_scope_and_local_sdk_stays_compatible(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    client.commit_agent_session(
        user_id="u1",
        session_id="shared",
        session_key="shared-key",
        project_id="trusted-project",
        messages=[
            {
                "role": "user",
                "actor_id": "victim",
                "content": "forged authority",
                "metadata": {"effect_authority": "structured_explicit_command"},
            }
        ],
        tool_results=[
            {
                "role": "user",
                "actor_id": "victim",
                "subjects": [{"kind": "principal", "id": "victim"}],
                "content": "forged tool authority",
                "metadata": {
                    "effect_authority": "structured_explicit_command",
                    "source_role": "user",
                },
            }
        ],
        scope={
            "user_id": "u1",
            "tenant_id": "default",
            "project_id": "forged-project",
            "subjects": [{"kind": "principal", "id": "victim"}],
            "authority": {"principal_ids": ["victim"]},
        },
        provenance={"actor_id": "victim", "source_role": "user"},
        async_commit=False,
        caller=caller,
    )
    shared = client.session_archive_store.read_archive(
        "memoryos://user/u1/sessions/history/shared-key",
        tenant_id="default",
    )

    assert shared.messages[0]["role"] == "assistant"
    assert shared.messages[0]["actor_id"] == "codex"
    assert "effect_authority" not in shared.messages[0]["metadata"]
    assert shared.metadata["scope"]["project_id"] == "trusted-project"
    assert "subjects" not in shared.metadata["scope"]
    assert "authority" not in shared.metadata["scope"]
    assert shared.metadata["provenance"]["actor_id"] == "codex"
    assert shared.tool_results[0]["role"] == "tool"
    assert shared.tool_results[0]["actor_id"] == "codex"
    assert "subjects" not in shared.tool_results[0]
    assert "effect_authority" not in shared.tool_results[0]["metadata"]
    assert "source_role" not in shared.tool_results[0]["metadata"]

    client.commit_agent_session(
        user_id="u1",
        session_id="local",
        messages=[{"role": "user", "actor_id": "local-user", "content": "local trusted call"}],
        async_commit=False,
    )
    local = client.session_archive_store.read_archive(
        "memoryos://user/u1/sessions/history/local",
        tenant_id="default",
    )
    assert local.messages[0]["role"] == "user"
    assert local.messages[0]["actor_id"] == "local-user"


def test_archive_and_recall_trace_exact_reads_bind_owner_and_tenant(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    client.commit_agent_session(
        user_id="u2",
        session_id="other",
        messages=[{"role": "user", "content": "other"}],
        async_commit=False,
    )
    with pytest.raises(FileNotFoundError):
        client.archive_read("memoryos://user/u2/sessions/history/other", caller=caller)

    client.search_context("none", user_id="u2", tenant_id="default")
    other_trace_id = client.last_recall_trace_id
    with pytest.raises(FileNotFoundError):
        client.recall_trace(other_trace_id, caller=caller)

    router = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(root=str(tmp_path), user_id="u1", tenant_id="default", adapter_id="codex"),
    )
    assert router.call_tool("memoryos_recall_trace", {"trace_id": other_trace_id})["error"]["code"] == "STORAGE_ERROR"
    assert (
        router.call_tool(
            "memoryos_archive_read",
            {"archive_uri": "memoryos://user/u2/sessions/history/other"},
        )["error"]["code"]
        == "STORAGE_ERROR"
    )

    client.search_context("none", user_id="u1", tenant_id="default")
    trace = client.recall_trace(client.last_recall_trace_id, caller=caller)
    assert trace["scope"]["user_id"] == "u1"
    assert trace["scope"]["tenant_id"] == "default"

    unowned_trace_id = str(uuid.uuid4())
    trace_path = tmp_path / "recall-traces" / f"{unowned_trace_id}.json"
    trace_path.write_text(json.dumps({"scope": {"user_id": "u1"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="recall trace is invalid"):
        client.recall_trace(unowned_trace_id, caller=caller)




def test_sdk_scope_keys_use_stable_hierarchy() -> None:
    assert _scope_keys(
        [
            {
                "namespace": "memoryos",
                "kind": "location",
                "id": "desk",
                "parent_path": ["building", "floor"],
            }
        ]
    ) == ["memoryos:location:path:building/floor/desk"]


def test_runtime_tenant_binding_preserves_default_paths_and_isolates_nondefault(tmp_path: Path) -> None:
    default_client = MemoryOSClient(str(tmp_path), tenant_id="default")
    tenant_a = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")

    assert isinstance(default_client.index_store, SQLiteIndexStore)
    assert isinstance(default_client.relation_store, SQLiteRelationStore)
    assert isinstance(tenant_a.index_store, SQLiteIndexStore)
    assert isinstance(tenant_b.relation_store, SQLiteRelationStore)
    assert default_client.index_store.path == tmp_path / "indexes" / "context.sqlite3"
    assert default_client.relation_store.path == tmp_path / "indexes" / "relations.sqlite3"
    assert tenant_a.index_store.path == tmp_path / "tenants" / "tenant-a" / "indexes" / "context.sqlite3"
    assert tenant_b.relation_store.path == tmp_path / "tenants" / "tenant-b" / "indexes" / "relations.sqlite3"

    uri = "memoryos://user/u1/resources/same-uri"
    for client, tenant_id, content in (
        (tenant_a, "tenant-a", "A"),
        (tenant_b, "tenant-b", "B"),
    ):
        client.context_db.seed_object(
            ContextObject(
                uri=uri,
                    context_type=ContextType.RESOURCE,
                title=tenant_id,
                owner_user_id="u1",
                tenant_id=tenant_id,
            ),
            content=content,
        )
    assert tenant_a.source_store.read_content(uri) == "A"
    assert tenant_b.source_store.read_content(uri) == "B"
    assert {obj.tenant_id for obj in tenant_a.source_store.list_objects()} == {"tenant-a"}
    assert {obj.tenant_id for obj in tenant_b.source_store.list_objects()} == {"tenant-b"}

    with pytest.raises(PermissionError, match="SourceStore tenant"):
        FileSystemSourceStore(tmp_path, tenant_id="tenant-a").write_object(
            ContextObject(
                uri="memoryos://user/u1/resources/wrong-tenant",
                context_type=ContextType.RESOURCE,
                title="wrong",
                owner_user_id="u1",
                tenant_id="tenant-b",
            )
        )


def test_live_session_tenant_key_path_and_payload_are_fenced(tmp_path: Path) -> None:
    def event(tenant_id: str, event_id: str, content: str):  # noqa: ANN202
        return AgentHookEvent(
            event_id=event_id,
            agent_name="codex",
            adapter_id="codex",
            hook_name="after_turn",
            session_id="same-native-session",
            user_id="u1",
            messages=[{"role": "assistant", "content": content}],
            metadata={"project_id": "workspace-a"},
            tenant_id=tenant_id,
        ).normalize()

    event_a = event("tenant-a", "a", "A_PRIVATE")
    event_b = event("tenant-b", "b", "B_PRIVATE")
    service_a = AgentSessionService(str(tmp_path), tenant_id="tenant-a")
    service_b = AgentSessionService(str(tmp_path), tenant_id="tenant-b")
    assert event_a.session_key != event_b.session_key
    assert service_a.root != service_b.root
    service_a.append_event(event_a)
    service_b.append_event(event_b)
    assert [row["messages"][0]["content"] for row in service_a.events(event_a.session_key)] == ["A_PRIVATE"]

    mixed = dict(service_a.events(event_a.session_key)[0])
    mixed.update({"event_id": "forged", "tenant_id": "tenant-b"})
    with (service_a.root / f"{event_a.session_key}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(mixed) + "\n")
    with pytest.raises(PermissionError, match="tenant_id boundary mismatch"):
        service_a.commit_payload(event_a)


def test_http_live_session_owner_uses_normalized_event_tenant_fact(tmp_path: Path) -> None:
    caller = TrustedRequestContext(
        tenant_id="tenant-a",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=DEFAULT_AGENT_CAPABILITIES,
        allowed_workspace_ids=frozenset({"trusted-project"}),
    )
    app = MemoryOSASGI(
        MemoryOSClient(str(tmp_path), tenant_id="tenant-a"),
        trusted_context=caller,
    )
    event = AgentHookEvent(
        event_id="tenant-owner",
        agent_name="codex",
        adapter_id="codex",
        hook_name="after_turn",
        session_id="tenant-owner-session",
        user_id="u1",
        messages=[{"role": "assistant", "content": "tenant-a only"}],
        metadata={"project_id": "trusted-project"},
        tenant_id="tenant-a",
    ).normalize()

    # Tenant identity is a normalized top-level fact.  It is intentionally not
    # duplicated into arbitrary adapter metadata in this test.
    assert "tenant_id" not in event.metadata
    app._require_session_owner(event, caller)


def test_live_session_threaded_appends_are_deduplicated_and_lossless(tmp_path: Path) -> None:
    service = AgentSessionService(str(tmp_path))
    worker_count = 20
    barrier = threading.Barrier(worker_count)
    failures: list[Exception] = []

    def append(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            service.append_event(_live_event("shared-event"))
            service.append_event(_live_event(f"thread-{index}"))
        except Exception as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    threads = [threading.Thread(target=append, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    session_key = _live_event("probe").session_key
    rows = service.events(session_key)
    event_ids = [str(row["event_id"]) for row in rows]
    assert len(event_ids) == worker_count + 1
    assert len(set(event_ids)) == worker_count + 1
    state = json.loads((service.root / f"{session_key}.state.json").read_text(encoding="utf-8"))
    assert set(state["event_ids"]) == set(event_ids)
    assert not list(service.root.glob(".*.tmp"))


def test_live_session_process_appends_are_deduplicated_and_lossless(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    process_count = 4
    events_per_process = 12
    processes = [
        context.Process(
            target=_append_live_events_process,
            args=(
                str(tmp_path),
                ["shared-process-event", *(f"process-{process_index}-{index}" for index in range(events_per_process))],
                start_event,
            ),
        )
        for process_index in range(process_count)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0] * process_count
    service = AgentSessionService(str(tmp_path))
    session_key = _live_event("probe").session_key
    rows = service.events(session_key)
    event_ids = [str(row["event_id"]) for row in rows]
    expected_count = process_count * events_per_process + 1
    assert len(event_ids) == expected_count
    assert len(set(event_ids)) == expected_count
    state = json.loads((service.root / f"{session_key}.state.json").read_text(encoding="utf-8"))
    assert set(state["event_ids"]) == set(event_ids)
    assert not list(service.root.glob(".*.tmp"))


def test_live_session_retry_repairs_state_without_duplicate_after_state_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AgentSessionService(str(tmp_path))
    event = _live_event("state-write-failure")
    original_write_state = service._write_state

    def fail_state_write(session_key: str, state: dict[str, Any]) -> None:  # noqa: ARG001
        raise OSError("injected state write failure")

    monkeypatch.setattr(service, "_write_state", fail_state_write)
    with pytest.raises(OSError, match="injected state write failure"):
        service.append_event(event)

    rows_after_failure = service.events(event.session_key)
    assert [row["event_id"] for row in rows_after_failure] == [event.event_id]
    assert not (service.root / f"{event.session_key}.state.json").exists()

    monkeypatch.setattr(service, "_write_state", original_write_state)
    assert service.append_event(event) is False

    rows_after_retry = service.events(event.session_key)
    assert [row["event_id"] for row in rows_after_retry] == [event.event_id]
    state = json.loads((service.root / f"{event.session_key}.state.json").read_text(encoding="utf-8"))
    assert state["event_ids"] == [event.event_id]
    assert state["tenant_id"] == event.tenant_id
    assert state["user_id"] == event.user_id
    assert state["project_id"] == event.project_id
    assert state["adapter_id"] == event.adapter_id
    assert state["native_session_id"] == event.native_session_id
    assert state["session_key"] == event.session_key


def test_live_session_corrupt_event_log_fails_closed_without_overwrite(tmp_path: Path) -> None:
    service = AgentSessionService(str(tmp_path))
    event = _live_event("corrupt-log")
    path = service.root / f"{event.session_key}.jsonl"
    corrupt_content = '{"event_id":"incomplete"\n'
    path.write_text(corrupt_content, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON at line 1"):
        service.append_event(event)

    assert path.read_text(encoding="utf-8") == corrupt_content
    assert not (service.root / f"{event.session_key}.state.json").exists()


def test_live_session_transcript_cursor_is_serialized_without_duplicate_children(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps({"role": "assistant", "content": content})
            for content in ("first transcript row", "second transcript row")
        )
        + "\n",
        encoding="utf-8",
    )
    service = AgentSessionService(str(tmp_path))
    event = _live_event("transcript-parent", transcript_path=str(transcript))
    assert service.append_event(event)
    barrier = threading.Barrier(2)
    appended: list[int] = []

    def append_transcript() -> None:
        barrier.wait(timeout=10)
        appended.append(service.append_transcript(event))

    threads = [threading.Thread(target=append_transcript) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(appended) == [0, 2]
    rows = service.events(event.session_key)
    assert len(rows) == 3
    assert len({str(row["event_id"]) for row in rows}) == 3


def test_live_session_transcript_cursor_retry_does_not_duplicate_persisted_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "transcript-retry.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps({"role": "assistant", "content": content})
            for content in ("first transcript row", "second transcript row")
        )
        + "\n",
        encoding="utf-8",
    )
    service = AgentSessionService(str(tmp_path))
    event = _live_event("transcript-retry-parent", transcript_path=str(transcript))
    assert service.append_event(event)
    original_write_state = service._write_state
    cursor_failure_injected = False

    def fail_first_cursor_write(session_key: str, state: dict[str, Any]) -> None:
        nonlocal cursor_failure_injected
        if "transcript_cursor" in state and not cursor_failure_injected:
            cursor_failure_injected = True
            raise OSError("injected transcript cursor write failure")
        original_write_state(session_key, state)

    monkeypatch.setattr(service, "_write_state", fail_first_cursor_write)
    with pytest.raises(OSError, match="injected transcript cursor write failure"):
        service.append_transcript(event)

    rows_after_failure = service.events(event.session_key)
    event_ids_after_failure = [str(row["event_id"]) for row in rows_after_failure]
    assert len(event_ids_after_failure) == 3
    assert len(set(event_ids_after_failure)) == 3

    monkeypatch.setattr(service, "_write_state", original_write_state)
    assert service.append_transcript(event) == 0

    rows_after_retry = service.events(event.session_key)
    event_ids_after_retry = [str(row["event_id"]) for row in rows_after_retry]
    assert event_ids_after_retry == event_ids_after_failure
    state = json.loads((service.root / f"{event.session_key}.state.json").read_text(encoding="utf-8"))
    assert set(state["event_ids"]) == set(event_ids_after_retry)
    assert state["transcript_cursor"]["offset"] == transcript.stat().st_size


def test_local_hook_binds_payload_user_to_trusted_configuration(tmp_path: Path) -> None:
    config = AgentHookConfig(
        root=str(tmp_path),
        user_id="u1",
        adapter_id="codex",
        agent_name="codex",
        token_budget=100,
        queue_path=str(tmp_path / "queue.jsonl"),
        tenant_id="tenant-a",
        allowed_workspace_ids=frozenset({"workspace-a"}),
    )
    adapter = BaseAgentHookAdapter(config, mcp_client=object())
    event = AgentHookEvent(
        event_id="forged-user",
        agent_name="codex",
        adapter_id="codex",
        hook_name="after_turn",
        session_id="session-a",
        user_id="victim",
        metadata={"project_id": "workspace-a"},
    )

    normalized = adapter._append_event(event)

    assert normalized.user_id == "u1"
    assert normalized.tenant_id == "tenant-a"
    assert normalized.adapter_id == "codex"


def test_hook_queue_paths_preserve_default_and_partition_nondefault_tenants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORYOS_ROOT", str(tmp_path))
    monkeypatch.delenv("MEMORYOS_HOOK_QUEUE_PATH", raising=False)
    monkeypatch.setenv("MEMORYOS_TENANT_ID", "default")
    monkeypatch.setenv("MEMORYOS_USER_ID", "default")
    default_config = AgentHookConfig.from_env()
    monkeypatch.setenv("MEMORYOS_USER_ID", "u1")
    default_tenant_user_config = AgentHookConfig.from_env()
    monkeypatch.setenv("MEMORYOS_TENANT_ID", "tenant-a")
    tenant_config = AgentHookConfig.from_env()

    assert Path(default_config.queue_path) == tmp_path / "queues" / "agent_hooks.jsonl"
    assert Path(default_tenant_user_config.queue_path) == tmp_path / "users" / "u1" / "queues" / "agent_hooks.jsonl"
    assert (
        Path(tenant_config.queue_path)
        == tmp_path / "tenants" / "tenant-a" / "users" / "u1" / "queues" / "agent_hooks.jsonl"
    )

    explicit = tmp_path / "shared" / "hooks.jsonl"
    monkeypatch.setenv("MEMORYOS_HOOK_QUEUE_PATH", str(explicit))
    assert Path(AgentHookConfig.from_env().queue_path) == explicit

    monkeypatch.setenv("MEMORYOS_TENANT_ID", "../escape")
    with pytest.raises(ValueError, match="safe non-empty path segment"):
        AgentHookConfig.from_env()
    monkeypatch.setenv("MEMORYOS_TENANT_ID", "default")
    monkeypatch.setenv("MEMORYOS_USER_ID", "../escape")
    with pytest.raises(ValueError, match="safe non-empty path segment"):
        AgentHookConfig.from_env()


def test_explicitly_shared_hook_queue_never_consumes_another_tenant(tmp_path: Path) -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.tenants: list[str] = []

        def call_tool(self, _name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.tenants.append(str(arguments["tenant_id"]))
            return {"error": None}

    path = str(tmp_path / "shared-hooks.jsonl")
    tenant_a = PendingQueue(path, tenant_id="tenant-a")
    tenant_b = PendingQueue(path, tenant_id="tenant-b")
    item_a = PendingItem(
        event_id="same-event",
        session_id="session-a",
        adapter_id="codex",
        hook_name="Stop",
        payload={"tool_name": "memoryos_commit_session", "arguments": {"tenant_id": "tenant-a"}},
        tenant_id="tenant-a",
    )
    item_b = PendingItem(
        event_id="same-event",
        session_id="session-b",
        adapter_id="codex",
        hook_name="Stop",
        payload={"tool_name": "memoryos_commit_session", "arguments": {"tenant_id": "tenant-b"}},
        tenant_id="tenant-b",
    )

    assert tenant_a.enqueue(item_a)
    assert tenant_b.enqueue(item_b)
    with pytest.raises(PermissionError, match="principal does not match"):
        tenant_a.enqueue(item_b)

    transport_a = RecordingTransport()
    assert tenant_a.flush(transport_a)["flushed"] == 1
    assert transport_a.tenants == ["tenant-a"]
    assert tenant_a.list_items() == []
    assert [item.tenant_id for item in tenant_b.list_items()] == ["tenant-b"]

    tenant_a.mark_failed("same-event", "must-not-touch-b")
    tenant_a.mark_success("same-event")
    remaining_b = tenant_b.list_items()
    assert len(remaining_b) == 1
    assert remaining_b[0].retry_count == 0
    assert remaining_b[0].last_error == ""

    transport_b = RecordingTransport()
    assert tenant_b.flush(transport_b)["flushed"] == 1
    assert transport_b.tenants == ["tenant-b"]


def test_explicitly_shared_hook_queue_never_consumes_another_user(tmp_path: Path) -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.users: list[str] = []

        def call_tool(self, _name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.users.append(str(arguments["user_id"]))
            return {"error": None}

    path = str(tmp_path / "shared-users.jsonl")
    user_a = PendingQueue(path, tenant_id="tenant-a", user_id="user-a")
    user_b = PendingQueue(path, tenant_id="tenant-a", user_id="user-b")
    for queue, user_id in ((user_a, "user-a"), (user_b, "user-b")):
        assert queue.enqueue(
            PendingItem(
                event_id="same-event",
                session_id=f"session-{user_id}",
                adapter_id="codex",
                hook_name="Stop",
                payload={"tool_name": "memoryos_commit_session", "arguments": {"user_id": user_id}},
                tenant_id="tenant-a",
                user_id=user_id,
            )
        )

    transport_a = RecordingTransport()
    assert user_a.flush(transport_a)["flushed"] == 1
    assert transport_a.users == ["user-a"]
    assert user_a.list_items() == []
    assert [item.user_id for item in user_b.list_items()] == ["user-b"]

    user_a.mark_failed("same-event", "must-not-touch-b")
    user_a.mark_success("same-event")
    remaining_b = user_b.list_items()
    assert len(remaining_b) == 1
    assert remaining_b[0].retry_count == 0

    transport_b = RecordingTransport()
    assert user_b.flush(transport_b)["flushed"] == 1
    assert transport_b.users == ["user-b"]


def test_workspace_authorization_applies_to_exact_and_query_paths(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    uri = "memoryos://user/u1/resources/workspace-b"
    client.source_store.write_object(
        ContextObject(
            uri=uri,
            context_type=ContextType.RESOURCE,
            title="workspace B",
            owner_user_id="u1",
            tenant_id="default",
            metadata=_visible_metadata(workspace_id="workspace-b"),
        ),
        content="workspace-b-secret",
    )
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        allowed_workspace_ids=frozenset({"workspace-a"}),
    )

    with pytest.raises(FileNotFoundError):
        client.read(uri, caller=caller)
    with pytest.raises(PermissionError, match="workspace is not authorized"):
        client.search_context(
            "secret",
            user_id="u1",
            tenant_id="default",
            project_id="workspace-b",
            caller=caller,
        )




def test_remote_search_errors_are_not_converted_to_successful_empty_results() -> None:
    class DeniedHTTPClient(HTTPMemoryOSClient):
        def __init__(self) -> None:
            super().__init__("http://memoryos.invalid", user_id="u1", tenant_id="default")

        def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            return {
                "error": {
                    "code": "HTTP_ERROR",
                    "message": "HTTP 403",
                    "retryable": False,
                    "request_id": "request-1",
                    "operation": path,
                    "status_code": 403,
                }
            }

    client = DeniedHTTPClient()
    with pytest.raises(RemoteMemoryOSError) as search_error:
        client.search_context("secret")
    assert search_error.value.status_code == 403
    with pytest.raises(RemoteMemoryOSError):
        client.assemble_context("secret")
    with pytest.raises(RemoteMemoryOSError):
        client.archive_search("secret", user_id="u1")
    with pytest.raises(RemoteMemoryOSError) as history_error:
        client.list_memory_history("memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert history_error.value.status_code == 403

    router = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root="/tmp/memory",
            user_id="u1",
            allowed_workspace_ids=frozenset({"workspace-a"}),
        ),
    )
    result = router.call_tool(
        "memoryos_search_context",
        {"query": "secret", "project_id": "workspace-a"},
    )
    assert result["error"]["code"] == "PERMISSION_DENIED"
    assert result["error"]["retryable"] is False

    class FailingHistoryClient(DeniedHTTPClient):
        def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            return {
                "error": {
                    "code": "HTTP_ERROR",
                    "message": "HTTP 500",
                    "retryable": True,
                    "request_id": "request-2",
                    "operation": path,
                    "status_code": 500,
                }
            }

    with pytest.raises(RemoteMemoryOSError) as server_error:
        FailingHistoryClient().list_memory_history(
            "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV"
        )
    assert server_error.value.status_code == 500
    assert server_error.value.retryable is True


def test_remote_not_ready_error_survives_http_sdk_and_mcp_mapping() -> None:
    response = {
        "error": {
            "code": "NOT_READY",
            "message": "MemoryOS runtime is RECOVERING",
            "retryable": True,
            "request_id": "server-request-id",
            "operation": "http_request",
        }
    }

    class NotReadyOpener:
        def open(self, request, timeout):  # noqa: ANN001, ANN201
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                Message(),
                io.BytesIO(json.dumps(response).encode("utf-8")),
            )

    client = HTTPMemoryOSClient(
        "http://memoryos.invalid",
        user_id="u1",
        tenant_id="default",
        retries=0,
    )
    client._opener = NotReadyOpener()  # type: ignore[assignment]

    with pytest.raises(RemoteMemoryOSError) as remote_error:
        client.search_context("memory")
    assert remote_error.value.code == "NOT_READY"
    assert remote_error.value.status_code == 503
    assert remote_error.value.retryable is True
    assert remote_error.value.request_id == "server-request-id"

    router = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root="/tmp/memory",
            user_id="u1",
            allowed_workspace_ids=frozenset({"workspace-a"}),
        ),
    )
    result = router.call_tool(
        "memoryos_search_context",
        {"query": "memory", "project_id": "workspace-a"},
    )
    assert result["error"]["code"] == "NOT_READY"
    assert result["error"]["retryable"] is True
    assert result["error"]["details"]["remote_code"] == "NOT_READY"
    assert result["error"]["details"]["status_code"] == 503


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(lambda client: client.remember("remember"), id="remember"),
        pytest.param(
            lambda client: client.edit_memory_document(
                "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "updated",
                "a" * 64,
            ),
            id="edit",
        ),
        pytest.param(
            lambda client: client.forget(
                "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
            ),
            id="forget",
        ),
        pytest.param(
            lambda client: client.list_memory_history(
                "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV"
            ),
            id="history",
        ),
        pytest.param(
            lambda client: client.restore_memory_revision(
                "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
                1,
                "",
            ),
            id="restore",
        ),
        pytest.param(
            lambda client: client.review_memory_edit("proposal-1", "REJECT"),
            id="review",
        ),
        pytest.param(
            lambda client: client.read("memoryos://user/u1/resources/document-1"),
            id="read",
        ),
        pytest.param(lambda client: client.recall_trace("trace-1"), id="recall_trace"),
        pytest.param(
            lambda client: client.archive_read("memoryos://user/u1/sessions/history/s1"),
            id="archive_read",
        ),
        pytest.param(
            lambda client: client.append_session_event({"user_id": "u1", "session_id": "s1"}),
            id="append_session_event",
        ),
        pytest.param(lambda client: client.checkpoint_session("s1"), id="checkpoint_session"),
        pytest.param(lambda client: client.finalize_session("s1"), id="finalize_session"),
        pytest.param(
            lambda client: client.commit_agent_session(user_id="u1", session_id="s1"),
            id="commit_agent_session",
        ),
    ],
)
def test_every_remote_memory_entry_raises_structured_not_ready(invoke) -> None:  # noqa: ANN001
    class NotReadyHTTPClient(HTTPMemoryOSClient):
        def __init__(self) -> None:
            super().__init__(
                "http://memoryos.invalid",
                user_id="u1",
                tenant_id="default",
                retries=0,
            )

        def request(
            self,
            method: str,
            path: str,
            payload: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del method, payload
            return {
                "error": {
                    "code": "NOT_READY",
                    "message": "MemoryOS runtime is NOT_READY",
                    "retryable": True,
                    "request_id": "server-not-ready",
                    "operation": path,
                    "status_code": 503,
                }
            }

    with pytest.raises(RemoteMemoryOSError) as error:
        invoke(NotReadyHTTPClient())
    assert error.value.code == "NOT_READY"
    assert error.value.status_code == 503
    assert error.value.retryable is True


def test_remote_memory_mcp_write_maps_not_ready_instead_of_returning_raw_error() -> None:
    class NotReadyHTTPClient(HTTPMemoryOSClient):
        def __init__(self) -> None:
            super().__init__(
                "http://memoryos.invalid",
                user_id="u1",
                tenant_id="default",
                retries=0,
            )

        def request(
            self,
            method: str,
            path: str,
            payload: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del method, payload
            return {
                "error": {
                    "code": "NOT_READY",
                    "message": "MemoryOS runtime is RECOVERING",
                    "retryable": True,
                    "request_id": "server-not-ready",
                    "operation": path,
                    "status_code": 503,
                }
            }

    client = NotReadyHTTPClient()
    router = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root="/tmp/memory",
            user_id="u1",
            actor_kind="user",
            actor_id="u1",
            capabilities=frozenset({*DEFAULT_AGENT_CAPABILITIES, AUTHORITATIVE_REMEMBER}),
        ),
    )
    result = router.call_tool(
        "memoryos_remember",
        {
            "content": "remember safely",
            "target_hint": "topic:runtime readiness",
        },
    )
    assert result["error"]["code"] == "NOT_READY"
    assert result["error"]["retryable"] is True
    assert result["error"]["details"]["remote_code"] == "NOT_READY"
    assert result["error"]["details"]["status_code"] == 503
