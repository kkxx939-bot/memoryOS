from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any

from memoryos.adapters.agent_hooks import cli
from memoryos.adapters.agent_hooks.claude_code import ClaudeCodeHookAdapter
from memoryos.adapters.agent_hooks.codex import CodexHookAdapter
from memoryos.adapters.agent_hooks.config import AgentHookConfig
from memoryos.adapters.agent_hooks.cursor import CursorHookAdapter
from memoryos.adapters.agent_hooks.events import AgentHookEvent
from memoryos.adapters.agent_hooks.mcp_client import AgentHookMCPClient
from memoryos.adapters.agent_hooks.queue import PendingItem, PendingQueue
from memoryos.adapters.agent_hooks.sanitizer import sanitize_changed_files, sanitize_payload, summarize_tool_result


class FakeHookMCPClient:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_commit = fail_commit

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "memoryos_assemble_context":
            return {
                "packed_context": "remember the local MCP plan",
                "source_uris": ["memoryos://ctx/1"],
                "dropped_contexts": [],
                "error": None,
            }
        if name == "memoryos_commit_session" and self.fail_commit:
            return {"error": {"code": "CLIENT_ERROR", "message": "temporary", "retryable": True, "details": {}}}
        return {"status": "done", "error": None}


def _config(tmp_path: Path, adapter_id: str = "codex") -> AgentHookConfig:
    return AgentHookConfig(
        root=str(tmp_path / "memory"),
        user_id="u1",
        adapter_id=adapter_id,
        agent_name=adapter_id,
        token_budget=128,
        queue_path=str(tmp_path / "queue.jsonl"),
    )


def test_codex_session_start_generates_session_id(tmp_path: Path) -> None:
    adapter = CodexHookAdapter(_config(tmp_path), mcp_client=FakeHookMCPClient())

    result = adapter.handle("SessionStart", {"cwd": str(tmp_path)}).to_dict()

    assert result["ok"] is True
    assert result["session_id"].startswith("agent-")


def test_codex_user_prompt_submit_injects_bounded_context(tmp_path: Path) -> None:
    adapter = CodexHookAdapter(_config(tmp_path), mcp_client=FakeHookMCPClient())

    result = adapter.handle("UserPromptSubmit", {"session_id": "s1", "prompt": "what changed?"}).to_dict()

    assert "<memoryos_context>" in result["injection_text"]
    assert "memoryos://ctx/1" in result["injection_text"]


def test_codex_post_tool_use_enqueues_without_immediate_flush(tmp_path: Path) -> None:
    fake = FakeHookMCPClient(fail_commit=True)
    adapter = CodexHookAdapter(_config(tmp_path), mcp_client=fake)

    result = adapter.handle(
        "PostToolUse",
        {"event_id": "e1", "session_id": "s1", "tool_name": "shell", "tool_output": "ok", "changed_files": ["memoryos/api/mcp/server.py"]},
    ).to_dict()

    assert result["queued"] is True
    assert result["flushed"] == {}
    assert fake.calls == []
    assert len(PendingQueue(_config(tmp_path).queue_path).list_items()) == 1


def test_codex_stop_commits_and_flushes_queue(tmp_path: Path) -> None:
    fake = FakeHookMCPClient()
    config = _config(tmp_path)
    queue = PendingQueue(config.queue_path)
    queue.enqueue(
        PendingItem(
            event_id="e1",
            session_id="s1",
            adapter_id="codex",
            hook_name="PostToolUse",
            payload={"tool_name": "memoryos_commit_session", "arguments": {"session_id": "s1"}},
        )
    )
    adapter = CodexHookAdapter(config, mcp_client=fake, queue=queue)

    result = adapter.handle("Stop", {"event_id": "stop1", "session_id": "s1", "messages": [{"role": "user", "content": "done"}]}).to_dict()

    assert result["committed"] is True
    assert result["flushed"]["flushed"] == 1
    assert queue.list_items() == []
    assert fake.calls[0][1]["async_commit"] is False


def test_codex_precompact_assembles_and_commits(tmp_path: Path) -> None:
    fake = FakeHookMCPClient()
    adapter = CodexHookAdapter(_config(tmp_path), mcp_client=fake)

    result = adapter.handle("PreCompact", {"session_id": "s1", "prompt": "compact this"}).to_dict()

    assert "remember the local MCP plan" in result["injection_text"]
    assert result["committed"] is True


def test_claude_code_before_prompt_and_after_turn(tmp_path: Path) -> None:
    adapter = ClaudeCodeHookAdapter(_config(tmp_path, "claude_code"), mcp_client=FakeHookMCPClient())

    before = adapter.handle("before_prompt", {"session_id": "s1", "input": "hello"}).to_dict()
    after = adapter.handle("after_turn", {"session_id": "s1", "messages": [{"role": "assistant", "content": "ok"}], "unknown": "kept"}).to_dict()

    assert before["injection_text"]
    assert after["committed"] is True


def test_cursor_before_prompt_after_turn_and_flush(tmp_path: Path) -> None:
    adapter = CursorHookAdapter(_config(tmp_path, "cursor"), mcp_client=FakeHookMCPClient())

    before = adapter.handle("before_prompt", {"session_id": "s1", "prompt": "hello"}).to_dict()
    after = adapter.handle("after_turn", {"session_id": "s1"}).to_dict()
    flushed = adapter.handle("flush", {"session_id": "s1"}).to_dict()

    assert before["injection_text"]
    assert after["committed"] is True
    assert flushed["flushed"]["remaining"] == 0


def test_hooks_only_call_context_or_commit_tools_with_agent_metadata(tmp_path: Path) -> None:
    adapters: list[tuple[Any, str, dict[str, Any]]] = [
        (CodexHookAdapter(_config(tmp_path, "codex"), mcp_client=FakeHookMCPClient()), "UserPromptSubmit", {"session_id": "s1", "prompt": "hello"}),
        (ClaudeCodeHookAdapter(_config(tmp_path, "claude_code"), mcp_client=FakeHookMCPClient()), "before_prompt", {"session_id": "s2", "input": "hello"}),
        (CursorHookAdapter(_config(tmp_path, "cursor"), mcp_client=FakeHookMCPClient()), "before_prompt", {"session_id": "s3", "prompt": "hello"}),
    ]

    for adapter, hook_name, payload in adapters:
        result = adapter.handle(hook_name, payload).to_dict()
        calls = adapter.mcp_client.calls
        assert result["ok"] is True
        assert {name for name, _args in calls} <= {
            "memoryos_assemble_context",
            "memoryos_search_context",
            "memoryos_commit_session",
            "memoryos_health",
        }
        assert "memoryos_predict" not in {name for name, _args in calls}
        assert "memoryos_process_observation" not in {name for name, _args in calls}
        metadata = calls[0][1]["connect_metadata"]
        assert metadata["connect_type"] == "agent"
        assert metadata["run_mode"] == "context_reduction"
        assert metadata["world_domain"] == "digital"
        assert metadata["source_kind"] == "coding_agent"
        assert metadata["capabilities"]["can_predict_behavior"] is False
        assert metadata["capabilities"]["can_execute_action"] is False


def test_agent_hook_mcp_client_real_initializes_and_calls_health(tmp_path: Path) -> None:
    client = AgentHookMCPClient(_config(tmp_path))

    result = client.call_tool("memoryos_health", {})

    assert result["error"] is None
    assert result["status"] == "ok"
    assert result["metadata"]["adapter_id"] == "codex"


def test_stop_and_precompact_commit_session_do_not_trigger_action_tools(tmp_path: Path) -> None:
    fake = FakeHookMCPClient()
    adapter = CodexHookAdapter(_config(tmp_path), mcp_client=fake)

    adapter.handle("Stop", {"event_id": "stop", "session_id": "s1", "messages": [{"content": "done"}]})
    adapter.handle("PreCompact", {"event_id": "compact", "session_id": "s1", "prompt": "compact"})

    tool_names = [name for name, _args in fake.calls]
    assert "memoryos_predict" not in tool_names
    assert "memoryos_process_observation" not in tool_names
    assert "memoryos_commit_session" in tool_names


def test_pending_queue_idempotency_retry_success_and_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "queue.jsonl"
    queue = PendingQueue(str(path))
    item = PendingItem(
        event_id="e1",
        session_id="s1",
        adapter_id="codex",
        hook_name="Stop",
        payload={"tool_name": "memoryos_commit_session", "arguments": {"session_id": "s1"}},
    )

    assert queue.enqueue(item) is True
    assert queue.enqueue(item) is False
    failed = queue.flush(FakeHookMCPClient(fail_commit=True))
    succeeded = queue.flush(FakeHookMCPClient())
    path.write_text("{bad json\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    queue.enqueue(PendingItem(event_id="e2", session_id="s2", adapter_id="codex", hook_name="Stop", payload={}))

    assert failed["failed"] == 1
    assert succeeded["flushed"] == 1
    assert [item.event_id for item in queue.list_items()] == ["e2"]


def test_pending_queue_concurrent_enqueue_keeps_all_unique_events(tmp_path: Path) -> None:
    queue = PendingQueue(str(tmp_path / "queue.jsonl"))

    def enqueue(index: int) -> None:
        queue.enqueue(
            PendingItem(
                event_id=f"e{index}",
                session_id="s1",
                adapter_id="codex",
                hook_name="PostToolUse",
                payload={"tool_name": "memoryos_commit_session", "arguments": {"session_id": "s1"}},
            )
        )

    threads = [threading.Thread(target=enqueue, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {item.event_id for item in queue.list_items()} == {f"e{index}" for index in range(20)}


def test_agent_hook_event_id_is_stable_for_retry_payloads(tmp_path: Path) -> None:
    payload = {"session_id": "s1", "prompt": "same prompt", "cwd": str(tmp_path)}

    first = AgentHookEvent.from_payload(payload, adapter_id="codex", hook_name="UserPromptSubmit")
    second = AgentHookEvent.from_payload(payload, adapter_id="codex", hook_name="UserPromptSubmit")
    with_timestamp = AgentHookEvent.from_payload(
        {**payload, "timestamp": "2026-07-06T00:00:00Z"},
        adapter_id="codex",
        hook_name="UserPromptSubmit",
    )
    same_timestamp = AgentHookEvent.from_payload(
        {**payload, "timestamp": "2026-07-06T00:00:00Z"},
        adapter_id="codex",
        hook_name="UserPromptSubmit",
    )

    assert first.event_id == second.event_id
    assert with_timestamp.event_id == same_timestamp.event_id


def test_agent_hook_event_id_changes_for_different_tool_payloads(tmp_path: Path) -> None:
    base = {"session_id": "s1", "cwd": str(tmp_path), "tool_name": "shell"}

    first = AgentHookEvent.from_payload({**base, "tool_input": {"cmd": "ls"}, "tool_output": "one"}, adapter_id="codex", hook_name="PostToolUse")
    same = AgentHookEvent.from_payload({**base, "tool_input": {"cmd": "ls"}, "tool_output": "one"}, adapter_id="codex", hook_name="PostToolUse")
    different_input = AgentHookEvent.from_payload({**base, "tool_input": {"cmd": "pwd"}, "tool_output": "one"}, adapter_id="codex", hook_name="PostToolUse")
    different_output = AgentHookEvent.from_payload({**base, "tool_input": {"cmd": "ls"}, "tool_output": "two"}, adapter_id="codex", hook_name="PostToolUse")
    different_files = AgentHookEvent.from_payload(
        {**base, "tool_input": {"cmd": "ls"}, "tool_output": "one", "changed_files": ["memoryos/a.py"]},
        adapter_id="codex",
        hook_name="PostToolUse",
    )
    explicit = AgentHookEvent.from_payload(
        {**base, "event_id": "external-event", "tool_input": {"cmd": "pwd"}, "tool_output": "two"},
        adapter_id="codex",
        hook_name="PostToolUse",
    )

    assert first.event_id == same.event_id
    assert different_input.event_id != first.event_id
    assert different_output.event_id != first.event_id
    assert different_files.event_id != first.event_id
    assert explicit.event_id == "external-event"


def test_fallback_session_id_includes_prompt_hint(tmp_path: Path) -> None:
    first = AgentHookEvent.from_payload({"cwd": str(tmp_path), "prompt": "task one"}, adapter_id="codex", hook_name="UserPromptSubmit")
    second = AgentHookEvent.from_payload({"cwd": str(tmp_path), "prompt": "task two"}, adapter_id="codex", hook_name="UserPromptSubmit")
    repeat = AgentHookEvent.from_payload({"cwd": str(tmp_path), "prompt": "task one"}, adapter_id="codex", hook_name="UserPromptSubmit")

    assert first.session_id == repeat.session_id
    assert first.session_id != second.session_id


def test_pending_queue_refuses_action_capable_tools(tmp_path: Path) -> None:
    queue = PendingQueue(str(tmp_path / "queue.jsonl"))
    queue.enqueue(
        PendingItem(
            event_id="e1",
            session_id="s1",
            adapter_id="codex",
            hook_name="Stop",
            payload={"tool_name": "memoryos_predict", "arguments": {"request": {}}},
        )
    )
    fake = FakeHookMCPClient()

    result = queue.flush(fake)

    assert result["failed"] == 1
    assert result["dead_lettered"] == 1
    assert fake.calls == []
    assert queue.list_items() == []
    assert "DISALLOWED_HOOK_TOOL" in queue.dead_letter_path.read_text(encoding="utf-8")


def test_pending_queue_dead_letters_after_retry_limit(tmp_path: Path) -> None:
    queue = PendingQueue(str(tmp_path / "queue.jsonl"), max_retries=2)
    queue.enqueue(
        PendingItem(
            event_id="e1",
            session_id="s1",
            adapter_id="codex",
            hook_name="Stop",
            payload={"tool_name": "memoryos_commit_session", "arguments": {"session_id": "s1"}},
        )
    )

    first = queue.flush(FakeHookMCPClient(fail_commit=True))
    second = queue.flush(FakeHookMCPClient(fail_commit=True))

    assert first["remaining"] == 1
    assert second["remaining"] == 0
    assert second["dead_lettered"] == 1
    assert "CLIENT_ERROR" in queue.dead_letter_path.read_text(encoding="utf-8")


def test_sanitizer_redacts_secret_truncates_logs_and_filters_paths() -> None:
    private_key = "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----"
    payload = {
        "api_key": "abc",
        "output": "\n".join(f"line {i}" for i in range(120)),
        "private": private_key,
        "paths": ["src/app.py", "node_modules/pkg/index.js", ".git/config"],
    }
    sanitized = sanitize_payload(payload, max_text=200)
    output = "\n".join(
        [
            "Authorization: Bearer sk-test",
            "OPENAI_API_KEY=sk-env",
            "api_key: raw",
            "password=secret",
            "ordinary context remains",
        ]
    )
    summary = summarize_tool_result("shell", {"password": "pw"}, output, ["dist/out.js", "memoryos/api/mcp/server.py"])

    assert sanitized["api_key"] == "<redacted>"
    assert "<redacted-private-key>" in sanitized["private"]
    assert "omitted" in sanitized["output"]
    assert sanitized["paths"] == ["src/app.py"]
    assert summary["tool_input"]["password"] == "<redacted>"
    assert "Bearer <redacted>" in summary["tool_output"]
    assert "OPENAI_API_KEY=<redacted>" in summary["tool_output"]
    assert "api_key: <redacted>" in summary["tool_output"]
    assert "password=<redacted>" in summary["tool_output"]
    assert "ordinary context remains" in summary["tool_output"]
    assert summary["changed_files"] == ["memoryos/api/mcp/server.py"]
    assert sanitize_changed_files([".venv/bin/python", "a.py"]) == ["a.py"]


def test_hook_fail_safe_when_mcp_unavailable(tmp_path: Path) -> None:
    class BrokenMCP:
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("down")

    adapter = CodexHookAdapter(_config(tmp_path), mcp_client=BrokenMCP())

    before = adapter.handle("UserPromptSubmit", {"session_id": "s1", "prompt": "hello"}).to_dict()
    stop = adapter.handle("Stop", {"event_id": "stop", "session_id": "s1"}).to_dict()

    assert before["ok"] is True
    assert before["error"]["code"] == "HOOK_SOFT_FAIL"
    assert stop["ok"] is True
    assert stop["queued"] is True


def test_cli_supports_stdin_json_payload_file_and_text_output(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    class FakeCursor:
        @classmethod
        def from_env(cls) -> FakeCursor:
            return cls()

        def handle(self, hook: str, payload: dict[str, Any]) -> Any:
            class Result:
                def to_dict(self) -> dict[str, Any]:
                    return {"ok": True, "injection_text": f"ctx:{payload.get('prompt', '')}"}

            return Result()

    monkeypatch.setattr(cli, "CursorHookAdapter", FakeCursor)
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps({"prompt": "from-file"}), encoding="utf-8")
    assert cli.main(["cursor", "before_prompt", "--payload-file", str(payload_file), "--format", "text"]) == 0
    assert "ctx:from-file" in capsys.readouterr().out

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "from-stdin"})))
    assert cli.main(["cursor", "before_prompt"]) == 0
    assert json.loads(capsys.readouterr().out)["injection_text"] == "ctx:from-stdin"
