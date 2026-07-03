from __future__ import annotations

from memoryos.interfaces.api.request_context import APIRequestContext, user_id_from_context_or_payload
from memoryos.interfaces.hooks.memory_digest_hook import MemoryHook
from memoryos.ports.repositories.memory_repository import MemoryRepository


def build_digest(store: MemoryRepository, payload: dict, context: APIRequestContext | None = None) -> dict:
    user_id = user_id_from_context_or_payload(context, payload)
    query = str(payload.get("query", ""))
    limit = int(payload.get("limit", 6))
    return {
        "user_id": user_id,
        "query": query,
        "digest": MemoryHook(store).build_digest(user_id, query, limit=limit),
    }


def delete_memory(store: MemoryRepository, payload: dict, context: APIRequestContext | None = None) -> dict:
    return store.delete_memory(str(payload["id"]), user_id=user_id_from_context_or_payload(context, payload))


def search_memory(store: MemoryRepository, payload: dict, context: APIRequestContext | None = None) -> dict:
    rows = store.hybrid_search(
        str(payload.get("query", "")),
        user_id=user_id_from_context_or_payload(context, payload),
        memory_type=payload.get("memory_type"),
        limit=int(payload.get("limit", 8)),
    )
    return {"results": rows}
