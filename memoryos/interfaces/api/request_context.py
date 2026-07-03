from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.config.settings import Settings


@dataclass(frozen=True)
class APIRequestContext:
    user_id: str
    request_id: str = ""
    is_internal_worker: bool = False
    token: str = ""
    metadata: dict = field(default_factory=dict)


def user_id_from_context_or_payload(context: APIRequestContext | None, payload: dict) -> str:
    if context and context.user_id:
        return context.user_id
    return str(payload["user_id"])


def require_internal_worker(context: APIRequestContext | None, settings: Settings) -> None:
    if not settings.worker_internal_token:
        return
    if not context or not context.is_internal_worker or context.token != settings.worker_internal_token:
        raise PermissionError("worker API requires internal worker context")
