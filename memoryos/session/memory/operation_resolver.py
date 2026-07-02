from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .extractor import MemoryOperation
from ...storage.memory_store import MemoryStore


@dataclass
class ResolvedMemoryOperation:
    operation: MemoryOperation
    target: str | None
    current: dict[str, Any] | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    page_id: int | None = None
    links: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_edit(self) -> bool:
        return self.current is not None


class MemoryOperationResolver:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def resolve(self, operation: MemoryOperation, user_id: str) -> ResolvedMemoryOperation:
        current = None
        if operation.target:
            current = self.store.resolve_memory(operation.target, user_id)
        return ResolvedMemoryOperation(
            operation=operation,
            target=operation.target,
            current=current,
            fields=self._operation_fields(operation),
            page_id=operation.page_id,
            links=self._normalize_links(operation.links),
        )

    def _operation_fields(self, operation: MemoryOperation) -> dict[str, Any]:
        return {
            "title": operation.title,
            "content": operation.text,
            "tags": operation.tags,
            "confidence": operation.confidence,
        }

    def _normalize_links(self, links: list[dict]) -> list[dict[str, Any]]:
        normalized = []
        for link in links:
            if not isinstance(link, dict):
                continue
            target = str(link.get("to") or link.get("target") or link.get("target_uri") or "").strip()
            if not target:
                continue
            normalized.append(
                {
                    "to": target,
                    "link_type": str(link.get("link_type", "related_to")),
                    "description": str(link.get("description", "")),
                    "weight": float(link.get("weight", 0.5) or 0.5),
                }
            )
        return normalized
