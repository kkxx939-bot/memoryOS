"""MemoryOS 运维、诊断、Worker 与预测命令协议。

CLI 只负责解析参数、创建公开 SDK 客户端、调用对应能力并输出安全结果；具体
业务规则由 Runtime 组合的 Context、Behavior 和 ActionPolicy 服务执行。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from config import MemoryOSConfig
from foundation.identity import LocalUserContext
from foundation.readiness import RuntimeNotReadyError
from LLMClient.config import ModelConfig
from openApi.version import __version__
from policy.action_policy.decision.request import PredictionRequest
from pre.connect import ConnectMetadata, ConnectType, PipelineMode


def run(argv: list[str] | None = None) -> int:
    """执行一个 CLI 命令并返回适合进程退出码的整数结果。"""

    common_config = MemoryOSConfig.from_env()
    # 命令定义集中在入口，确保 shell 参数不会绕过公开 SDK 直接触达内部存储。
    parser = argparse.ArgumentParser(description="MemoryOS Predictive Context Database")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    sub.add_parser("inspect-architecture")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--root", default=common_config.root)
    worker = sub.add_parser("worker")
    worker.add_argument(
        "kind",
        choices=[
            "recovery",
            "session-commit",
            "semantic",
            "embedding",
            "maintenance",
            "all",
        ],
    )
    worker.add_argument("--root", default=common_config.root)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--poll-interval", type=float, default=1.0)
    worker.add_argument("--batch-size", type=int, default=10)
    worker.add_argument("--lease-seconds", type=int, default=60)
    worker.add_argument("--max-retries", type=int, default=3)
    predict = sub.add_parser("predict")
    predict.add_argument("--root", default=common_config.root)
    predict.add_argument("--user", required=True)
    predict.add_argument("--episode", required=True)
    predict.add_argument("--observation", required=True)
    predict.add_argument("--policies-json", default="[]")
    predict.add_argument("--connect-metadata-json")
    predict.add_argument("--connect-metadata-file")
    context_search = sub.add_parser("context-search")
    _add_context_query_arguments(context_search, default_limit=10, default_root=common_config.root)
    context_search.add_argument("--context-type")
    context_assemble = sub.add_parser("context-assemble")
    _add_context_query_arguments(context_assemble, default_limit=20, default_root=common_config.root)
    context_assemble.add_argument("--context-type", action="append", dest="context_types")
    context_read = sub.add_parser("context-read")
    _add_context_identity_arguments(context_read, default_root=common_config.root)
    context_read.add_argument("--uri", required=True)
    context_read.add_argument("--layer", choices=["L0", "L1", "L2"], default="L2")
    recall_trace = sub.add_parser("recall-trace")
    _add_context_identity_arguments(recall_trace, default_root=common_config.root)
    recall_trace.add_argument("--trace-id", required=True)
    archive_search = sub.add_parser("archive-search")
    _add_context_query_arguments(archive_search, default_limit=20, default_root=common_config.root)
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "inspect-architecture":
        print(
            json.dumps(
                {
                    "product": "MemoryOS",
                    "positioning": "Predictive Context Database for AI Agents",
                    "production_entrypoint": "MemoryOSClient.process_observation",
                    "planes": ["ContextDB", "Behavior", "ActionPolicy", "Operation Plane"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "doctor":
        root = Path(args.root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            writable = root.is_dir() and root.stat().st_mode != 0
            client = _client(str(root))
            report = {
                "root": str(root),
                "writable": writable,
                **client.health(),
                "mcp_sdk": "ready" if __import__("importlib.util").util.find_spec("mcp") else "unavailable",
                "http_runtime": "ready" if __import__("importlib.util").util.find_spec("uvicorn") else "unavailable",
                "agent_integrations": {
                    "codex": (Path.home() / ".codex" / "hooks.json").exists(),
                    "claude_code": (Path.home() / ".claude" / "settings.json").exists(),
                },
            }
            print(json.dumps(report, ensure_ascii=False))
            return 0
        except Exception as exc:
            _print_cli_error(exc)
            return 2
    if args.command == "worker":
        # Worker 是可选运维能力，延迟导入可保持 version/doctor 等轻量命令快速启动。
        from runtime.worker.runner import WorkerRunner

        try:
            client = _client(args.root)
            result = WorkerRunner(
                client,
                poll_interval=args.poll_interval,
                batch_size=args.batch_size,
                lease_seconds=args.lease_seconds,
                max_retries=args.max_retries,
            ).run(args.kind, once=args.once)
        except RuntimeNotReadyError as exc:
            error = {
                "kind": args.kind,
                "status": "not_ready",
                "runtime_state": exc.state.value,
                "reasons": [_safe_cli_error_message(reason) for reason in exc.reasons],
            }
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
            return 2
        if result.get("status") == "not_ready":
            runtime = dict(result.get("runtime", {}) or {})
            error = {
                "kind": args.kind,
                "status": "not_ready",
                "runtime_state": str(runtime.get("state") or "NOT_READY"),
                "reasons": [_safe_cli_error_message(str(reason)) for reason in runtime.get("reasons", [])],
            }
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"kind": args.kind, "status": "completed", "result": result}, ensure_ascii=False))
        return 0
    if args.command == "predict":
        try:
            connect_metadata = _load_predict_connect_metadata(args)
            policies = _load_policies(args.policies_json)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, PermissionError) as exc:
            _print_cli_error(exc)
            return 2
        request = PredictionRequest(
            user_id=args.user,
            episode_id=args.episode,
            observation=args.observation,
            available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
            connect_metadata=connect_metadata,
        )
        try:
            print(
                json.dumps(
                    _client(args.root).predict(request, policies).to_dict(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except PermissionError as exc:
            _print_cli_error(exc)
            return 2
        return 0
    if args.command in {
        "context-search",
        "context-assemble",
        "context-read",
        "recall-trace",
        "archive-search",
    }:
        try:
            client = _client(args.root)
            caller = _context_caller(args)
            if args.command == "context-search":
                results = client.search_context(
                    args.query,
                    user_id=args.user,
                    context_type=args.context_type,
                    limit=_context_limit(args.limit),
                    project_id=args.project,
                    caller=caller,
                )
                payload = {
                    "results": results,
                    "trace_id": str(client.last_recall_trace_id or ""),
                }
            elif args.command == "context-assemble":
                payload = client.assemble_context(
                    args.query,
                    user_id=args.user,
                    context_types=args.context_types,
                    limit=_context_limit(args.limit),
                    project_id=args.project,
                    caller=caller,
                )
            elif args.command == "context-read":
                payload = client.read(
                    args.uri,
                    layer=args.layer,
                    caller=caller,
                )
            elif args.command == "recall-trace":
                payload = client.recall_trace(args.trace_id, caller=caller)
            else:
                payload = {
                    "results": client.archive_search(
                        args.query,
                        user_id=args.user,
                        limit=_context_limit(args.limit),
                        caller=caller,
                        project_id=args.project,
                    )
                }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        except (OSError, RuntimeError, TypeError, ValueError, PermissionError) as exc:
            _print_cli_error(exc)
            return 2
    return 2


def _add_context_identity_arguments(parser: argparse.ArgumentParser, *, default_root: str) -> None:
    """为本地 Context 命令声明用户和工作区参数。"""

    parser.add_argument("--root", default=default_root)
    parser.add_argument("--user", required=True)
    parser.add_argument("--project", default="")


def _add_context_query_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_limit: int,
    default_root: str,
) -> None:
    """为搜索类命令补充统一查询参数。"""

    _add_context_identity_arguments(parser, default_root=default_root)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=default_limit)


def _context_caller(args: argparse.Namespace) -> LocalUserContext:
    """根据本地 CLI 参数构造单用户运行上下文。"""

    project = str(getattr(args, "project", "") or "").strip()
    return LocalUserContext(
        user_id=str(args.user),
        adapter_id="memoryos_cli",
        workspace_id=project,
    )


def _context_limit(value: int) -> int:
    """限制 CLI 查询条目数量，避免绕过公开检索上限。"""

    limit = int(value)
    if limit < 1 or limit > 200:
        raise ValueError("context limit must be between 1 and 200")
    return limit


def _load_predict_connect_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.connect_metadata_json and args.connect_metadata_file:
        raise ValueError("provide only one of --connect-metadata-json or --connect-metadata-file")
    if args.connect_metadata_file:
        try:
            text = Path(args.connect_metadata_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError("failed to read connect metadata file") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("connect metadata file must contain valid JSON") from exc
    elif args.connect_metadata_json:
        try:
            payload = json.loads(args.connect_metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("connect metadata JSON must be valid JSON") from exc
    else:
        raise PermissionError("predict requires explicit embodied/action_capable connect metadata")
    if not isinstance(payload, dict):
        raise ValueError("connect metadata must be a JSON object")
    metadata = ConnectMetadata.from_dict(payload)
    if (
        metadata.connect_type != ConnectType.EMBODIED
        or metadata.run_mode != PipelineMode.ACTION_CAPABLE
        or not metadata.capabilities.can_predict_behavior
    ):
        raise PermissionError("predict requires embodied/action_capable metadata with can_predict_behavior=True")
    return metadata.to_dict()


def _client(root: str) -> Any:
    """在真正需要运行时容器时才加载 SDK。"""

    from openApi.sdk.client import MemoryOSClient

    return MemoryOSClient(root, model_config=ModelConfig.from_env())


def _load_policies(raw: str) -> list[Any] | None:
    try:
        policies_payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("policies JSON must be valid JSON") from exc
    if not isinstance(policies_payload, list):
        raise ValueError("policies JSON must be an array")
    if not policies_payload:
        return None
    from policy.action_policy.model.action_policy import ActionPolicy

    policies = []
    for item in policies_payload:
        if not isinstance(item, dict):
            raise ValueError("policies JSON entries must be objects")
        try:
            policies.append(ActionPolicy(**item))
        except TypeError as exc:
            raise ValueError("policies JSON entries must be valid policy objects") from exc
    return policies


SECRET_RE = re.compile(r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+")
PATH_RE = re.compile(r"(?:(?:/Users|/private|/var|/tmp)/[^\s'\",;:)]*)")


def _print_cli_error(exc: Exception) -> None:
    print(_safe_cli_error_message(str(exc)), file=sys.stderr)


def _safe_cli_error_message(message: str) -> str:
    sanitized = SECRET_RE.sub("<redacted>", message)
    sanitized = PATH_RE.sub("<redacted-path>", sanitized)
    return sanitized[:500]
