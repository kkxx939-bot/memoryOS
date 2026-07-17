from __future__ import annotations

from pathlib import Path
from typing import Any

from memoryos.api.http.app import handle
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import DEFAULT_AGENT_CAPABILITIES, TrustedRequestContext
from memoryos.connect import ConnectMetadata


def _caller() -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=DEFAULT_AGENT_CAPABILITIES,
        allowed_workspace_ids=frozenset({"project-x"}),
    )


def _payload() -> dict[str, Any]:
    return {
        "user_id": "u1",
        "session_id": "native-session",
        "session_key": "shared-session-key",
        "project_id": "project-x",
        "messages": [{"id": "m1", "role": "assistant", "content": "evidence"}],
        "used_contexts": [
            {"uri": "memoryos://user/u1/memories/context-1", "provenance": {"rank": 1}}
        ],
        "used_skills": [{"uri": "memoryos://skills/testing/pytest", "version": "1"}],
        "tool_results": [{"tool_name": "shell", "content": "ok"}],
        "scope": {"tenant_id": "default", "user_id": "u1", "purpose": "test"},
        "provenance": {"transport_evidence": "equivalent"},
        "connect_metadata": ConnectMetadata.default_agent("codex").to_dict(),
        "async_commit": False,
    }


def test_local_http_and_mcp_preserve_equivalent_session_evidence(tmp_path: Path) -> None:
    caller = _caller()
    local = MemoryOSClient(str(tmp_path / "local"))
    http = MemoryOSClient(str(tmp_path / "http"), mode="server")
    mcp = MemoryOSClient(str(tmp_path / "mcp"))
    payload = _payload()

    local.commit_agent_session(**payload, caller=caller)
    http_result = handle("POST /sessions/commit", http, payload, caller=caller)
    mcp_result = MemoryOSMCPServer(
        mcp,
        config=MCPServerConfig(
            root=str(tmp_path / "mcp"),
            tenant_id="default",
            user_id="u1",
            adapter_id="codex",
            actor_kind="agent",
            actor_id="codex",
            capabilities=DEFAULT_AGENT_CAPABILITIES,
            allowed_workspace_ids=frozenset({"project-x"}),
        ),
    ).call_tool("memoryos_commit_session", payload)

    assert http_result["status"] == "queued"
    assert http_result["state"] == "QUEUED"
    assert mcp_result["error"] is None
    assert mcp_result["status"] == "queued"
    uri = "memoryos://user/u1/sessions/history/shared-session-key"
    archives = [
        client.session_archive_store.read_archive(uri)
        for client in (local, http, mcp)
    ]
    expected = archives[0]
    for archive in archives[1:]:
        assert archive.messages == expected.messages
        assert archive.used_contexts == expected.used_contexts
        assert archive.used_skills == expected.used_skills
        assert archive.tool_results == expected.tool_results
        assert archive.metadata["scope"] == expected.metadata["scope"]
        assert archive.metadata["provenance"] == expected.metadata["provenance"]


def test_mcp_does_not_wrap_downstream_error_as_accepted() -> None:
    class FailedRemoteClient:
        tenant_id = "default"

        def commit_agent_session(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "error": {
                    "code": "REMOTE_COMMIT_FAILED",
                    "message": "downstream rejected commit",
                    "retryable": True,
                }
            }

    result = MemoryOSMCPServer(
        FailedRemoteClient(),
        config=MCPServerConfig(
            root="/tmp/memoryos-test",
            tenant_id="default",
            user_id="u1",
            adapter_id="codex",
        ),
    ).call_tool(
        "memoryos_commit_session",
        {"user_id": "u1", "session_id": "s1"},
    )

    assert result["error"]["code"] == "REMOTE_COMMIT_FAILED"
    assert result.get("status") != "accepted"
