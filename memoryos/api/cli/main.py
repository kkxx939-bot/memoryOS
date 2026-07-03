from __future__ import annotations

import argparse
import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.prediction.model.prediction_request import PredictionRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryOS Predictive Context Database")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--root", default="./memory-root")
    predict.add_argument("--user", required=True)
    predict.add_argument("--episode", required=True)
    predict.add_argument("--observation", required=True)
    predict.add_argument("--policies-json", default="[]")
    args = parser.parse_args()
    if args.command == "predict":
        request = PredictionRequest(
            user_id=args.user,
            episode_id=args.episode,
            observation=args.observation,
            available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
        )
        policies = [ActionPolicy(**item) for item in json.loads(args.policies_json)]
        print(json.dumps(MemoryOSClient(args.root).predict(request, policies).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
