from __future__ import annotations

from memoryos.interfaces.hooks.memory_digest_hook import MemoryHook
from memoryos.ports.repositories.memory_repository import MemoryRepository


def build_digest(store: MemoryRepository, payload: dict) -> dict:
    user_id = str(payload["user_id"])
    query = str(payload.get("query", ""))
    limit = int(payload.get("limit", 6))
    return {
        "user_id": user_id,
        "query": query,
        "digest": MemoryHook(store).build_digest(user_id, query, limit=limit),
    }


def delete_memory(store: MemoryRepository, payload: dict) -> dict:
    return store.delete_memory(str(payload["id"]), user_id=str(payload["user_id"]))


def search_memory(store: MemoryRepository, payload: dict) -> dict:
    rows = store.hybrid_search(
        str(payload.get("query", "")),
        user_id=str(payload["user_id"]),
        memory_type=payload.get("memory_type"),
        limit=int(payload.get("limit", 8)),
    )
    return {"results": rows}
