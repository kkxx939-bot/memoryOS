"""Codex、Claude Code、Cursor 等 Agent Hook 的命令行桥接入口。

入口把 stdin 或文件载荷交给对应适配器规范化，再通过统一交付客户端调用
MemoryOS；它不直接解释或持久化 Agent 事件。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agent_hook.claude_code import ClaudeCodeHookAdapter
from agent_hook.codex import CodexHookAdapter
from agent_hook.composition import (
    build_agent_hook_transport,
    register_agent_hook_transport_factory,
)
from agent_hook.config import AgentHookConfig
from agent_hook.contracts import ClaudeCodeOutputRenderer, CodexOutputRenderer
from agent_hook.cursor import CursorHookAdapter
from agent_hook.queue import PendingQueue
from openApi.cli.agent_hook_transport import AgentHookTransportClient


def run(argv: list[str] | None = None) -> int:
    """规范化一次 Agent Hook 输入并按指定格式输出交付结果。"""

    # 适配器依赖统一的传输工厂，不直接选择本地 MCP 或远程 HTTP。
    register_agent_hook_transport_factory(AgentHookTransportClient)
    parser = argparse.ArgumentParser(description="MemoryOS agent hook bridge")
    parser.add_argument("adapter", choices=["codex", "claude_code", "cursor", "openclaw", "opencode", "flush"])
    parser.add_argument("hook", nargs="?")
    parser.add_argument("--payload-file")
    parser.add_argument("--format", choices=["json", "text", "native"], default="json")
    args = parser.parse_args(argv)
    if args.adapter == "flush":
        # flush 只重放已持久化的待发送事件，不重新解释原始 Hook 载荷。
        config = AgentHookConfig.from_env("codex")
        result = PendingQueue(
            config.queue_path,
            user_id=config.user_id,
        ).flush(build_agent_hook_transport(config))
        _emit({"ok": True, "flushed": result}, args.format)
        return 0
    payload = _read_payload(args.payload_file)
    if not args.hook:
        _emit({"ok": False, "error": {"code": "VALIDATION_ERROR", "message": "hook is required"}}, "json")
        return 2
    adapter = _adapter(args.adapter)
    hook_result = adapter.handle(args.hook, payload)
    result = hook_result.to_dict()
    if args.format == "native":
        if args.adapter == "claude_code":
            result = ClaudeCodeOutputRenderer().render(args.hook, hook_result)
        elif args.adapter == "codex":
            result = CodexOutputRenderer().render(args.hook, hook_result)
        _emit(result, "json")
        return 0 if hook_result.ok else 2
    _emit(result, args.format)
    return 0 if result.get("ok", False) else 2


def _adapter(name: str) -> Any:
    if name == "codex":
        return CodexHookAdapter.from_env()
    if name == "claude_code":
        return ClaudeCodeHookAdapter.from_env()
    if name == "cursor":
        return CursorHookAdapter.from_env()
    if name in {"openclaw", "opencode"}:
        return CursorHookAdapter.from_env()
    raise ValueError(name)


def _read_payload(path: str | None) -> dict[str, Any]:
    if path:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _emit(result: dict[str, Any], output_format: str) -> None:
    if output_format == "text":
        sys.stdout.write(str(result.get("injection_text", "")))
        if result.get("injection_text"):
            sys.stdout.write("\n")
        return
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(run())
