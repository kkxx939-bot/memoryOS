from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.scope import ScopeRef


class EpistemicStatus(str, Enum):
    EXPLICIT = "EXPLICIT"
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESIZED = "HYPOTHESIZED"


class SpeechAct(str, Enum):
    OBSERVATION = "OBSERVATION"
    PROPOSAL = "PROPOSAL"
    EVALUATION_REQUEST = "EVALUATION_REQUEST"
    CONFIRMATION = "CONFIRMATION"
    CORRECTION = "CORRECTION"
    RETRACTION = "RETRACTION"
    REJECTION = "REJECTION"


class Commitment(str, Enum):
    WEAK = "WEAK"
    EXPLORATORY = "EXPLORATORY"
    INTENDED = "INTENDED"
    CONFIRMED = "CONFIRMED"


class TemporalScope(str, Enum):
    PAST = "PAST"
    CURRENT = "CURRENT"
    FUTURE = "FUTURE"
    UNSPECIFIED = "UNSPECIFIED"


class SemanticRelation(str, Enum):
    UNRELATED = "UNRELATED"
    DUPLICATE = "DUPLICATE"
    SUPPLEMENTS = "SUPPLEMENTS"
    ALTERNATIVE = "ALTERNATIVE"
    CONTRADICTS = "CONTRADICTS"
    CORRECTS = "CORRECTS"
    SUPERSEDES = "SUPERSEDES"


@dataclass(frozen=True)
class SemanticAssessment:
    speech_act: str
    commitment: str
    temporal_scope: str
    relation_to_existing: str = "unrelated"


@dataclass(frozen=True)
class NormalizedSemanticAssessment:
    speech_act: SpeechAct
    commitment: Commitment
    temporal_scope: TemporalScope
    relation_to_existing: SemanticRelation

    def to_dict(self) -> dict[str, str]:
        return {
            "speech_act": self.speech_act.value,
            "commitment": self.commitment.value,
            "temporal_scope": self.temporal_scope.value,
            "relation_to_existing": self.relation_to_existing.value,
        }


@dataclass(frozen=True)
class MemorySemanticProposal:
    proposal_id: str
    memory_type: str
    identity_fields: Mapping[str, Any]
    value_fields: Mapping[str, Any]
    semantic: SemanticAssessment | NormalizedSemanticAssessment
    epistemic_status: EpistemicStatus
    suggested_scope_refs: tuple[ScopeRef, ...]
    related_memory_ids: tuple[str, ...]
    evidence_refs: tuple[Any, ...]
    confidence: float
    extractor_version: str
    model_id: str | None = None
    prompt_version: str = "memory_semantic_proposal_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.memory_type:
            raise ValueError("proposal_id and memory_type are required")
        object.__setattr__(self, "identity_fields", MappingProxyType(dict(self.identity_fields)))
        object.__setattr__(self, "value_fields", MappingProxyType(dict(self.value_fields)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        if isinstance(self.epistemic_status, str):
            object.__setattr__(self, "epistemic_status", EpistemicStatus(self.epistemic_status.upper()))

    @property
    def fingerprint(self) -> str:
        semantic = self.semantic.to_dict() if isinstance(self.semantic, NormalizedSemanticAssessment) else {
            "speech_act": self.semantic.speech_act,
            "commitment": self.semantic.commitment,
            "temporal_scope": self.semantic.temporal_scope,
            "relation_to_existing": self.semantic.relation_to_existing,
        }
        return stable_hash(
            [
                self.memory_type,
                dict(self.identity_fields),
                dict(self.value_fields),
                semantic,
                self.epistemic_status.value,
                sorted(scope.key for scope in self.suggested_scope_refs),
                sorted((ref.event_id, ref.content_hash, ref.span_start, ref.span_end) for ref in self.evidence_refs),
            ],
            length=40,
        )
