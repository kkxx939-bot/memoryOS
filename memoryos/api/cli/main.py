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
        except (ValueError, PermissionError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        request = PredictionRequest(
            user_id=args.user,
            episode_id=args.episode,
            observation=args.observation,
            available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
            connect_metadata=connect_metadata,
        )
        policies_payload = json.loads(args.policies_json)
        policies = None
        if policies_payload:
            from memoryos.action_policy.model.action_policy import ActionPolicy

            policies = [ActionPolicy(**item) for item in policies_payload]
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
        payload = json.loads(Path(args.connect_metadata_file).read_text(encoding="utf-8"))
    elif args.connect_metadata_json:
        payload = json.loads(args.connect_metadata_json)
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


if __name__ == "__main__":
    raise SystemExit(main())
