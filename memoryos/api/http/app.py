from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.prediction.model.prediction_request import PredictionRequest


def handle(route: str, client: MemoryOSClient, payload: dict) -> dict:
    if route == "POST /predict":
        request = PredictionRequest(**_required_dict(payload, "request", route))
        policies = [ActionPolicy(**item) for item in payload.get("policies", [])]
        return client.predict(request, policies).to_dict()
    if route == "POST /context/search":
        query = _required_str(payload, "query", route)
        return {
            "results": client.search_context(
                query,
                user_id=payload.get("user_id"),
                context_type=payload.get("context_type"),
                limit=int(payload.get("limit", 10)),
                connect_metadata=payload.get("connect_metadata"),
            )
        }
    if route == "POST /context/assemble":
        query = _required_str(payload, "query", route)
        return client.assemble_context(
            query,
            user_id=payload.get("user_id"),
            token_budget=int(payload.get("token_budget", 2000)),
            context_types=payload.get("context_types"),
            limit=int(payload.get("limit", 20)),
            connect_metadata=payload.get("connect_metadata"),
        )
    if route == "POST /sessions/commit":
        user_id = _required_str(payload, "user_id", route)
        session_id = _required_str(payload, "session_id", route)
        client.commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            messages=payload.get("messages"),
            used_contexts=payload.get("used_contexts"),
            tool_results=payload.get("tool_results"),
            connect_metadata=payload.get("connect_metadata"),
            async_commit=bool(payload.get("async_commit", True)),
        )
        return {"status": "accepted"}
    raise KeyError(f"Unknown route: {route}")


def _required_dict(payload: dict, key: str, route: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{route} requires object field: {key}")
    return value


def _required_str(payload: dict, key: str, route: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{route} requires non-empty string field: {key}")
    return value
