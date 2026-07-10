from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import memoryos
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode
from memoryos.prediction.model.prediction_request import PredictionRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MemoryOS Predictive Context Database")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    sub.add_parser("inspect-architecture")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--root", default="./memory-root")
    worker = sub.add_parser("worker")
    worker.add_argument(
        "kind",
        choices=["session-commit", "memory-proposal", "memory-projection", "maintenance", "all"],
    )
    worker.add_argument("--root", default="./memory-root")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--poll-interval", type=float, default=1.0)
    worker.add_argument("--batch-size", type=int, default=10)
    worker.add_argument("--lease-seconds", type=int, default=60)
    worker.add_argument("--max-retries", type=int, default=3)
    predict = sub.add_parser("predict")
    predict.add_argument("--root", default="./memory-root")
    predict.add_argument("--user", required=True)
    predict.add_argument("--episode", required=True)
    predict.add_argument("--observation", required=True)
    predict.add_argument("--policies-json", default="[]")
    predict.add_argument("--connect-metadata-json")
    predict.add_argument("--connect-metadata-file")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(memoryos.__version__)
        return 0
    if args.command == "inspect-architecture":
        print(
            json.dumps(
                {
                    "product": "MemoryOS",
                    "positioning": "Predictive Context Database for AI Agents",
                    "production_entrypoint": "MemoryOSClient.process_observation",
                    "planes": ["ContextDB", "Memory", "Behavior", "ActionPolicy", "Prediction", "Operation Plane"],
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
            client = MemoryOSClient(str(root))
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
        from memoryos.workers.runner import WorkerRunner

        client = MemoryOSClient(args.root)
        result = WorkerRunner(
            client,
            poll_interval=args.poll_interval,
            batch_size=args.batch_size,
            lease_seconds=args.lease_seconds,
            max_retries=args.max_retries,
        ).run(args.kind, once=args.once)
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
            print(json.dumps(MemoryOSClient(args.root).predict(request, policies).to_dict(), ensure_ascii=False, indent=2))
        except PermissionError as exc:
            _print_cli_error(exc)
            return 2
        return 0
    return 2


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


def _load_policies(raw: str) -> list[Any] | None:
    try:
        policies_payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("policies JSON must be valid JSON") from exc
    if not isinstance(policies_payload, list):
        raise ValueError("policies JSON must be an array")
    if not policies_payload:
        return None
    from memoryos.action_policy.model.action_policy import ActionPolicy

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


if __name__ == "__main__":
    raise SystemExit(main())
