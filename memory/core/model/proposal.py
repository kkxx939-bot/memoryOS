"""从可信证据形成的记忆编辑候选。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class MemoryCandidateKind(str, Enum):
    PROFILE_FACT = "profile_fact"
    PREFERENCE = "preference"
    ENTITY_NOTE = "entity_note"
    TOPIC_NOTE = "topic_note"
    EPISODE = "episode"
    OPEN_LOOP = "open_loop"
    EXPERIENCE = "experience"


@dataclass(frozen=True)
class MemoryEditProposal:
    candidate_kind: MemoryCandidateKind
    title: str
    body: str
    evidence_refs: tuple[str, ...]
    subject: str = ""
    entity_hints: tuple[str, ...] = ()
    topic_hints: tuple[str, ...] = ()
    occurred_at: str = ""
    temporal_status: str = ""
    relation_hints: tuple[str, ...] = ()
    field_evidence_refs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.body.strip() or not self.evidence_refs:
            raise ValueError("memory edit proposal requires title, body and evidence")
        if not isinstance(self.candidate_kind, MemoryCandidateKind):
            raise ValueError("memory edit proposal requires a configured candidate kind")
        if any(not str(item).strip() for item in self.evidence_refs):
            raise ValueError("memory edit proposal evidence references cannot be empty")
        confidence = self.confidence
        if isinstance(confidence, bool):
            raise ValueError("memory edit proposal confidence must be numeric")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory edit proposal confidence must be numeric") from exc
        if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
            raise ValueError("memory edit proposal confidence must be between zero and one")
        evidence_refs = tuple(dict.fromkeys(str(item).strip() for item in self.evidence_refs))
        normalized_field_refs = {
            str(key): tuple(dict.fromkeys(str(item).strip() for item in values))
            for key, values in self.field_evidence_refs.items()
        }
        if any(not key or any(not item for item in values) for key, values in normalized_field_refs.items()):
            raise ValueError("memory field evidence references cannot be empty")
        if any(not set(values).issubset(evidence_refs) for values in normalized_field_refs.values()):
            raise ValueError("memory field evidence must belong to proposal evidence")
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "confidence", confidence_value)
        object.__setattr__(self, "field_evidence_refs", MappingProxyType(normalized_field_refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_kind": self.candidate_kind.value,
            "title": self.title,
            "body": self.body,
            "evidence_refs": list(self.evidence_refs),
            "subject": self.subject,
            "entity_hints": list(self.entity_hints),
            "topic_hints": list(self.topic_hints),
            "occurred_at": self.occurred_at,
            "temporal_status": self.temporal_status,
            "relation_hints": list(self.relation_hints),
            "field_evidence_refs": {key: list(values) for key, values in sorted(self.field_evidence_refs.items())},
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryEditProposal:
        allowed = {
            "candidate_kind",
            "title",
            "body",
            "evidence_refs",
            "subject",
            "entity_hints",
            "topic_hints",
            "occurred_at",
            "temporal_status",
            "relation_hints",
            "field_evidence_refs",
            "confidence",
        }
        if set(payload) - allowed:
            raise ValueError("sealed memory proposal contains unsupported fields")
        field_refs = payload.get("field_evidence_refs", {})
        if not isinstance(field_refs, Mapping):
            raise ValueError("sealed field evidence must be an object")
        return cls(
            candidate_kind=MemoryCandidateKind(str(payload["candidate_kind"])),
            title=str(payload["title"]),
            body=str(payload["body"]),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
            subject=str(payload.get("subject") or ""),
            entity_hints=tuple(str(item) for item in payload.get("entity_hints", [])),
            topic_hints=tuple(str(item) for item in payload.get("topic_hints", [])),
            occurred_at=str(payload.get("occurred_at") or ""),
            temporal_status=str(payload.get("temporal_status") or ""),
            relation_hints=tuple(str(item) for item in payload.get("relation_hints", [])),
            field_evidence_refs={str(key): tuple(str(item) for item in values) for key, values in field_refs.items()},
            confidence=float(payload.get("confidence", 1.0)),
        )


__all__ = ["MemoryCandidateKind", "MemoryEditProposal"]
