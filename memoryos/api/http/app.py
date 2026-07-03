from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.prediction.model.prediction_request import PredictionRequest


def handle(route: str, client: MemoryOSClient, payload: dict) -> dict:
    if route == "POST /predict":
        request = PredictionRequest(**payload["request"])
        policies = [ActionPolicy(**item) for item in payload.get("policies", [])]
        return client.predict(request, policies).to_dict()
    raise KeyError(f"Unknown route: {route}")
