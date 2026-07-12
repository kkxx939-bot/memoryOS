from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any

from memoryos.adapters.agent_hooks.base import HookResult
from memoryos.adapters.agent_hooks.contracts import (
    ClaudeCodeOutputRenderer,
    ClaudeCodePayloadParser,
    CodexPayloadParser,
)
from memoryos.adapters.agent_hooks.events import AgentEventType, AgentHookEvent, make_session_key, project_identity
from memoryos.adapters.agent_hooks.session_service import AgentSessionService
from memoryos.adapters.agent_hooks.transcript import (
    ClaudeCodeTranscriptReader,
    CodexTranscriptReader,
    GenericJsonlTranscriptReader,
)
from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.stdio import _build_transport_client
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.sdk.http_client import HTTPMemoryOSClient


def test_project_identity_and_session_key_are_stable_and_isolated() -> None:
    ssh = project_identity("/a/repo", "/a/repo", "git@github.com:Example/Repo.git")
    https = project_identity("/b/repo", "/b/repo", "https://github.com/example/repo.git")
    assert ssh == https
    assert make_session_key("u", ssh, "codex", "same") != make_session_key("u", ssh, "claude_code", "same")


def test_transcript_reader_preserves_cursor_on_parse_failure(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_text('{"role":"user","content":"one"}\n', encoding="utf-8")
    reader = GenericJsonlTranscriptReader()
    first = reader.read_since(str(path), None)
    path.write_text(path.read_text() + "invalid\n", encoding="utf-8")
    second = reader.read_since(str(path), first.cursor)
    assert first.messages == [{"role": "user", "content": "one"}]
    assert second.parse_failed is True
    assert second.cursor == first.cursor


def test_transcript_reader_is_incremental_bounded_and_recovers_truncation(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    first_line = '{"role":"user","content":"one"}\n'
    second_line = '{"role":"assistant","content":"two"}\n'
    path.write_text(first_line + second_line, encoding="utf-8")
    reader = GenericJsonlTranscriptReader(max_bytes=len(first_line) + 5)
    first = reader.read_since(str(path), None)
    second = reader.read_since(str(path), first.cursor)
    assert [item["content"] for item in first.messages] == ["one"]
    assert [item["content"] for item in second.messages] == ["two"]
    path.write_text('{"role":"user","content":"new"}\n', encoding="utf-8")
    rotated = reader.read_since(str(path), second.cursor)
    assert rotated.truncated is True
    assert [item["content"] for item in rotated.messages] == ["new"]


def test_platform_transcript_readers_normalize_native_envelopes(tmp_path: Path) -> None:
    claude_path = tmp_path / "claude.jsonl"
    claude_path.write_text(
        json.dumps({"uuid": "c1", "message": {"role": "assistant", "content": "claude"}}) + "\n",
        encoding="utf-8",
    )
    codex_path = tmp_path / "codex.jsonl"
    codex_path.write_text(
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": "codex"}}) + "\n",
        encoding="utf-8",
    )
    assert ClaudeCodeTranscriptReader().read_since(str(claude_path), None).messages == [
        {"id": "c1", "role": "assistant", "content": "claude"}
    ]
    assert CodexTranscriptReader().read_since(str(codex_path), None).messages == [
        {"id": "", "role": "assistant", "content": "codex"}
    ]


def test_session_journal_dedupes_and_builds_append_only_commit(tmp_path: Path) -> None:
    service = AgentSessionService(str(tmp_path))
    prompt = AgentHookEvent.from_payload({"event_id": "p1", "session_id": "n1", "prompt": "hello", "project_id": "p"}, adapter_id="codex", hook_name="UserPromptSubmit", user_id="u").normalize()
    tool = AgentHookEvent.from_payload({"event_id": "t1", "session_id": "n1", "tool_name": "shell", "tool_output": "ok", "project_id": "p"}, adapter_id="codex", hook_name="PostToolUse", user_id="u").normalize()
    assert service.append_event(prompt) is True
    assert service.append_event(prompt) is False
    assert service.append_event(tool) is True
    payload = service.commit_payload(tool)
    assert [message["content"] for message in payload["messages"]] == ["hello"]
    assert len(payload["tool_results"]) == 1
    assert service.finalize(tool.session_key)["status"] == "COMMITTED"
    assert service.finalize(tool.session_key)["status"] == "COMMITTED"


def test_native_payload_parsers_and_claude_renderer() -> None:
    claude_fixture = json.loads(Path("integrations/claude-code/fixtures/user_prompt_submit.json").read_text())
    codex_fixture = json.loads(Path("integrations/codex/fixtures/user_prompt_submit.json").read_text())
    claude = ClaudeCodePayloadParser().parse(claude_fixture, user_id="u")
    codex = CodexPayloadParser().parse(codex_fixture, user_id="u")
    assert claude.normalize().event_type == AgentEventType.PROMPT_SUBMIT
    assert codex.normalize().event_type == AgentEventType.PROMPT_SUBMIT
    rendered = ClaudeCodeOutputRenderer().render("UserPromptSubmit", HookResult(ok=True, injection_text="ctx"))
    assert rendered["hookSpecificOutput"]["additionalContext"] == "ctx"


def test_missing_transcript_soft_fails_without_losing_event(tmp_path: Path) -> None:
    service = AgentSessionService(str(tmp_path))
    event = AgentHookEvent.from_payload(
        {
            "event_id": "p1",
            "session_id": "n1",
            "prompt": "hello",
            "project_id": "p",
            "transcript_path": str(tmp_path / "not-created.jsonl"),
        },
        adapter_id="claude_code",
        hook_name="UserPromptSubmit",
        user_id="u",
    ).normalize()
    assert service.append_event(event) is True
    assert service.append_transcript(event) == 0
    assert len(service.events(event.session_key)) == 1


def test_asgi_health_and_body_limit(tmp_path: Path) -> None:
    app = MemoryOSASGI(MemoryOSClient(str(tmp_path)), api_token="secret", max_body_bytes=10)

    async def invoke() -> tuple[int, dict]:
        sent = []
        messages = iter([{"type": "http.request", "body": b"{}", "more_body": False}])
        async def receive():  # noqa: ANN202
            return next(messages)
        async def send(message):  # noqa: ANN001, ANN202
            sent.append(message)
        await app({"type": "http", "method": "GET", "path": "/health", "headers": [(b"authorization", b"Bearer secret")]}, receive, send)
        return sent[0]["status"], json.loads(sent[1]["body"])

    status, body = asyncio.run(invoke())
    assert status == 200
    assert body["source_store"] == "ready"


def test_http_client_remote_commit_archives_then_finalizes() -> None:
    class FakeHTTPClient(HTTPMemoryOSClient):
        def __init__(self) -> None:
            super().__init__("http://memoryos.invalid")
            self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

        def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            self.calls.append((method, path, payload))
            if path == "/v1/sessions/events":
                return {"status": "ARCHIVED", "session_key": "stable"}
            return {"status": "committed", "done": True}

    client = FakeHTTPClient()
    result = client.commit_agent_session(session_id="native", async_commit=True)
    assert result["done"] is True
    assert [path for _, path, _ in client.calls] == [
        "/v1/sessions/events",
        "/v1/sessions/stable/finalize",
    ]


def test_http_client_exposes_remote_memory_health_and_trace_routes() -> None:
    class FakeHTTPClient(HTTPMemoryOSClient):
        def __init__(self) -> None:
            super().__init__("http://memoryos.invalid")
            self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

        def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            self.calls.append((method, path, payload))
            return {"status": "ok"}

    client = FakeHTTPClient()
    client.health()
    client.remember(user_id="u1", content="remember")
    client.forget(user_id="u1", uri="memoryos://user/u1/memories/x")
    client.list_pending(user_id="u1", lifecycle_states=["PENDING"])
    client.review_pending(
        user_id="u1",
        pending_uri="memoryos://user/u1/memories/pending/p1",
        decision="REJECT",
    )
    client.read("memoryos://user/u1/memories/x", layer="L1")
    client.recall_trace("trace/id")
    client.checkpoint_session("session-1")
    client.archive_search("needle", user_id="u1", limit=5)
    client.archive_read("memoryos://user/u1/sessions/history/s1")

    assert [path for _, path, _ in client.calls] == [
        "/health",
        "/v1/memories/remember",
        "/v1/memories/forget",
        "/v1/memories/pending?user_id=u1&tenant_id=default&lifecycle_state=PENDING",
        "/v1/memories/pending/review",
        "/v1/context/read?uri=memoryos%3A%2F%2Fuser%2Fu1%2Fmemories%2Fx&layer=L1",
        "/v1/recall-traces/trace%2Fid",
        "/v1/sessions/session-1/checkpoint",
        "/v1/archives/search",
        "/v1/archives/read?archive_uri=memoryos%3A%2F%2Fuser%2Fu1%2Fsessions%2Fhistory%2Fs1",
    ]


def test_mcp_stdio_selects_http_transport_for_remote_mode(tmp_path: Path, monkeypatch: Any) -> None:
    config = MCPServerConfig(root=str(tmp_path), user_id="u1", adapter_id="cursor")
    monkeypatch.setenv("MEMORYOS_BASE_URL", "https://memory.example")
    monkeypatch.setenv("MEMORYOS_API_TOKEN", "test-token")

    client = _build_transport_client(config)

    assert isinstance(client, HTTPMemoryOSClient)
    assert client.base_url == "https://memory.example"
    assert client.user_id == "u1"


def test_http_client_unavailable_returns_structured_retryable_error(monkeypatch: Any) -> None:
    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    client = HTTPMemoryOSClient("http://127.0.0.1:1", retries=0, connect_timeout=0.05, read_timeout=0.05)
    error = client.request("GET", "/health")["error"]
    assert error["code"] == "REMOTE_UNAVAILABLE"
    assert error["retryable"] is True
    assert error["request_id"]
    assert error["operation"] == "/health"


def test_token_budget_degrades_l2_to_smaller_layer(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        user_id="u1",
        project_id="p1",
        memory_type="project_decision",
        title="Needle decision",
        content="needle " + ("implementation detail " * 200),
    )
    result = client.assemble_context(
        "needle",
        user_id="u1",
        project_id="p1",
        search_scope="project_decisions",
        token_budget=40,
    )
    assert result["contexts"]
    assert result["contexts"][0]["selected_layer"] in {"L0", "excerpt"}


def test_claude_installer_is_idempotent_and_uninstalls(tmp_path: Path) -> None:
    script = Path("integrations/claude-code/install.py").resolve()
    settings = tmp_path / "settings.json"
    command = [sys.executable, str(script), "--settings", str(settings)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
    installed = json.loads(settings.read_text(encoding="utf-8"))
    assert "memoryos" in installed["mcpServers"]
    assert all(len(entries) == 1 for entries in installed["hooks"].values())
    subprocess.run([*command, "--uninstall"], check=True, capture_output=True, text=True)
    removed = json.loads(settings.read_text(encoding="utf-8"))
    assert "memoryos" not in removed["mcpServers"]
    assert not removed["hooks"]


def test_codex_installer_is_idempotent_supports_dry_run_and_uninstalls(tmp_path: Path) -> None:
    script = Path("integrations/codex/install.py").resolve()
    codex_home = tmp_path / "codex-home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_codex = bin_dir / "codex"
    fake_codex.write_text("#!/bin/sh\n[ \"$2\" = get ] && exit 1\nexit 0\n", encoding="utf-8")
    fake_codex.chmod(0o755)
    env = {**os.environ, "CODEX_HOME": str(codex_home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    command = [sys.executable, str(script)]
    subprocess.run([*command, "--dry-run"], env=env, check=True, capture_output=True, text=True)
    assert not (codex_home / "hooks.json").exists()
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    installed = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    assert all(len(entries) == 1 for entries in installed["hooks"].values())
    subprocess.run([*command, "--uninstall"], env=env, check=True, capture_output=True, text=True)
    removed = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    assert not removed["hooks"]
