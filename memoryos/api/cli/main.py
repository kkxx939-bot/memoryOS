from __future__ import annotations

import argparse
import json
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
    if args.command == "predict":
        try:
            connect_metadata = _load_predict_connect_metadata(args)
            policies = _load_policies(args.policies_json)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, PermissionError) as exc:
            print(str(exc), file=sys.stderr)
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
            print(str(exc), file=sys.stderr)
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

    try:
        return [ActionPolicy(**item) for item in policies_payload]
    except TypeError as exc:
        raise ValueError("policies JSON entries must be objects") from exc


if __name__ == "__main__":
    raise SystemExit(main())
