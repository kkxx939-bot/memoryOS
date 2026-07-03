from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from memoryos.services.memory.lifecycle import classify_lifecycle, hotness_score
from memoryos.services.memory.schema import MEMORY_TYPE_SPECS, memory_type_spec, type_description
from memoryos.services.memory.weights import default_base_weight, default_temporal_scope, score_memory_weight

MEMORY_TYPES = set(MEMORY_TYPE_SPECS)

TYPE_DIR = {
    memory_type: spec.directory
    for memory_type, spec in MEMORY_TYPE_SPECS.items()
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    allowed = []
    for ch in value.lower().strip():
        if ch.isalnum():
            allowed.append(ch)
        elif ch in {" ", "-", "_", ".", "/"}:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or f"memory-{uuid4().hex[:8]}"


def summarize_text(value: str, limit: int = 240) -> str:
    lines = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return " ".join(lines)[:limit]


@dataclass
class MemoryItem:
    user_id: str
    memory_type: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)
    memory_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    source: str = "manual"
    confidence: float = 1.0
    path: str | None = None
    last_accessed_at: str | None = None
    active_count: int = 0
    temporal_scope: str | None = None
    base_weight: float | None = None
    evidence_count: int = 1
    positive_count: int = 1
    negative_count: int = 0

    def __post_init__(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            known = ", ".join(sorted(MEMORY_TYPES))
            raise ValueError(f"Unknown memory type: {self.memory_type}. Known types: {known}")
        if self.path is None:
            directory = memory_type_spec(self.memory_type).directory
            slug = slugify(self.title)
            self.path = str(PurePosixPath("user") / self.user_id / directory / f"{slug}-{self.memory_id[:8]}.md")

    @property
    def abstract(self) -> str:
        return summarize_text(self.text)

    def metadata(self) -> dict[str, Any]:
        hotness = hotness_score(self.active_count, self.updated_at)
        base_weight = self.base_weight if self.base_weight is not None else default_base_weight(self.memory_type)
        temporal_scope = self.temporal_scope or default_temporal_scope(self.memory_type)
        weight = score_memory_weight(
            {
                "type": self.memory_type,
                "base_weight": base_weight,
                "temporal_scope": temporal_scope,
                "confidence": self.confidence,
                "evidence_count": self.evidence_count,
                "positive_count": self.positive_count,
                "negative_count": self.negative_count,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )
        return {
            "id": self.memory_id,
            "user_id": self.user_id,
            "type": self.memory_type,
            "type_description": type_description(self.memory_type),
            "title": self.title,
            "path": self.path,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "active_count": self.active_count,
            "hotness": hotness,
            "lifecycle_state": classify_lifecycle(hotness),
            "temporal_scope": temporal_scope,
            "base_weight": base_weight,
            "evidence_count": self.evidence_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "effective_weight": weight.effective_weight,
            "abstract": self.abstract,
        }
