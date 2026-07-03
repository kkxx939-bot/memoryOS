from __future__ import annotations

from memoryos.interfaces.api.behavior_api import behavior_patterns
from memoryos.interfaces.api.episode_api import process_episode
from memoryos.interfaces.api.feedback_api import record_feedback
from memoryos.interfaces.api.health_api import health
from memoryos.interfaces.api.memory_api import build_digest, delete_memory, search_memory
from memoryos.interfaces.api.worker_api import (
    process_feedback_outbox,
    run_memory_consolidation,
    run_reindex,
    run_replay,
)
from memoryos.ports.repositories.memory_repository import MemoryRepository


def routes() -> dict[str, str]:
    return {
        "GET /health": "health",
        "POST /episodes": "process_episode",
        "POST /episodes/feedback": "record_feedback",
        "POST /workers/feedback": "process_feedback_outbox",
        "POST /workers/reindex": "run_reindex",
        "POST /workers/replay": "run_replay",
        "POST /workers/memory-consolidation": "run_memory_consolidation",
        "GET /memory/digest": "build_digest",
        "GET /memory/search": "search_memory",
        "POST /memory/delete": "delete_memory",
        "GET /behavior/patterns": "behavior_patterns",
    }


def handle(route: str, store: MemoryRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    if route not in routes():
        raise KeyError(f"Unknown API route: {route}")
    if route == "GET /health":
        return health()
    if route == "POST /episodes":
        return process_episode(store, payload)
    if route == "POST /episodes/feedback":
        return record_feedback(store, payload)
    if route == "POST /workers/feedback":
        return process_feedback_outbox(store, payload)
    if route == "POST /workers/reindex":
        return run_reindex(store, payload)
    if route == "POST /workers/replay":
        return run_replay(store, payload)
    if route == "POST /workers/memory-consolidation":
        return run_memory_consolidation(store, payload)
    if route == "GET /memory/digest":
        return build_digest(store, payload)
    if route == "GET /memory/search":
        return search_memory(store, payload)
    if route == "POST /memory/delete":
        return delete_memory(store, payload)
    if route == "GET /behavior/patterns":
        return behavior_patterns(store, payload)
    raise KeyError(f"Unknown API route: {route}")
