"""Finite schema for model-authored semantic memory candidates."""

from __future__ import annotations

import builtins
from dataclasses import dataclass

from memoryos.memory.documents.model import MemoryCandidateKind

MEMORY_SCHEMA_VERSION = "markdown_memory_candidate_v1"


@dataclass(frozen=True)
class MemoryCandidateSchema:
    candidate_kind: MemoryCandidateKind
    description: str
    requires_occurred_at: bool = False
    allow_assistant_source: bool = False


class MemoryCandidateRegistry:
    def __init__(self, schemas: builtins.list[MemoryCandidateSchema] | None = None) -> None:
        rows = schemas or self._builtins()
        self._schemas = {item.candidate_kind: item for item in rows}

    def get(self, kind: MemoryCandidateKind | str) -> MemoryCandidateSchema:
        return self._schemas[MemoryCandidateKind(kind)]

    def list(self) -> builtins.list[MemoryCandidateSchema]:
        return list(self._schemas.values())

    @staticmethod
    def _builtins() -> builtins.list[MemoryCandidateSchema]:
        return [
            MemoryCandidateSchema(
                MemoryCandidateKind.PROFILE_FACT,
                "Stable user identity, background, or self-description.",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.PREFERENCE,
                "Durable preference, communication habit, or long-lived constraint.",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.ENTITY_NOTE,
                "Knowledge about a person, organization, product, system, or other entity.",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.TOPIC_NOTE,
                "Cross-event knowledge organized by topic rather than project directory.",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.EPISODE,
                "A time-bound discussion, event, decision, or outcome.",
                requires_occurred_at=True,
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.OPEN_LOOP,
                "An unresolved question, pending confirmation, or future follow-up.",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.EXPERIENCE,
                "A reusable experience distilled from an observed result.",
                requires_occurred_at=True,
                allow_assistant_source=True,
            ),
        ]


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryCandidateKind",
    "MemoryCandidateRegistry",
    "MemoryCandidateSchema",
]
