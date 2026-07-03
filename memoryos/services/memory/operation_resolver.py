from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.services.memory.extractor import MemoryOperation


@dataclass
class ResolvedMemoryOperation:
    operation: MemoryOperation
    target: str | None
    current: dict[str, Any] | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    page_id: int | None = None
    links: list[dict[str, Any]] = field(default_factory=list)
    candidate_targets: list[dict[str, Any]] = field(default_factory=list)
    resolution_confidence: float = 0.0
    needs_confirmation: bool = False

    @property
    def is_edit(self) -> bool:
        return self.current is not None


class MemoryOperationResolver:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def resolve(self, operation: MemoryOperation, user_id: str) -> ResolvedMemoryOperation:
        current = None
        target = operation.target
        candidate_targets: list[dict[str, Any]] = []
        resolution_confidence = 0.0
        needs_confirmation = False
        if target:
            current = self.store.resolve_memory(target, user_id)
            resolution_confidence = 1.0
        elif operation.action in {"update", "delete"}:
            candidate_targets = self.find_candidate_targets(operation, user_id)
            if candidate_targets:
                best = candidate_targets[0]
                score = float(best.get("score", best.get("final_score", 0.0)) or 0.0)
                resolution_confidence = max(0.0, min(1.0, score))
                if resolution_confidence >= 0.35:
                    target = str(best["path"])
                    current = self.store.resolve_memory(target, user_id)
                else:
                    needs_confirmation = True
            else:
                needs_confirmation = True
        return ResolvedMemoryOperation(
            operation=operation,
            target=target,
            current=current,
            fields=self._operation_fields(operation),
            page_id=operation.page_id,
            links=self._normalize_links(operation.links),
            candidate_targets=candidate_targets,
            resolution_confidence=resolution_confidence,
            needs_confirmation=needs_confirmation,
        )

    def find_candidate_targets(self, operation: MemoryOperation, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        query = " ".join([operation.title, operation.text, *operation.tags]).strip()
        if not query:
            return []
        rows = self.store.hybrid_search(query, user_id=user_id, memory_type=operation.memory_type, limit=limit)
        return [
            {
                "id": row.get("id"),
                "path": row.get("path"),
                "title": row.get("title"),
                "type": row.get("type"),
                "score": row.get("score", row.get("final_score")),
                "abstract": row.get("abstract"),
            }
            for row in rows
            if row.get("path")
        ]

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
