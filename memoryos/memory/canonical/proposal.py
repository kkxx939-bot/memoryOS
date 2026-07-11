"""记忆系统里的提案。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.scope import ScopeRef

if TYPE_CHECKING:
    from memoryos.memory.canonical.evidence import EvidenceRef


class EpistemicStatus(str, Enum):
    """负责 EpistemicStatus 这部分逻辑。"""

    EXPLICIT = "EXPLICIT"
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESIZED = "HYPOTHESIZED"


class SpeechAct(str, Enum):
    """列出提案支持的表达行为。"""

    OBSERVATION = "OBSERVATION"
    PROPOSAL = "PROPOSAL"
    EVALUATION_REQUEST = "EVALUATION_REQUEST"
    CONFIRMATION = "CONFIRMATION"
    CORRECTION = "CORRECTION"
    RETRACTION = "RETRACTION"
    REJECTION = "REJECTION"


class Commitment(str, Enum):
    """表示提案里的承诺强弱，不等同于 Claim 状态。"""

    WEAK = "WEAK"
    EXPLORATORY = "EXPLORATORY"
    INTENDED = "INTENDED"
    CONFIRMED = "CONFIRMED"


class TemporalScope(str, Enum):
    """表示一条语义说的是过去、现在还是未来。"""

    PAST = "PAST"
    CURRENT = "CURRENT"
    FUTURE = "FUTURE"
    UNSPECIFIED = "UNSPECIFIED"


class SemanticRelation(str, Enum):
    """列出新提案和已有记忆之间的关系。"""

    UNRELATED = "UNRELATED"
    DUPLICATE = "DUPLICATE"
    SUPPLEMENTS = "SUPPLEMENTS"
    ALTERNATIVE = "ALTERNATIVE"
    CONTRADICTS = "CONTRADICTS"
    CORRECTS = "CORRECTS"
    SUPERSEDES = "SUPERSEDES"


@dataclass(frozen=True)
class SemanticAssessment:
    """保存 SemanticAssessment 需要的这组数据。"""

    speech_act: str
    commitment: str
    temporal_scope: str
    relation_to_existing: str = "unrelated"


@dataclass(frozen=True)
class NormalizedSemanticAssessment:
    """保存 NormalizedSemanticAssessment 需要的这组数据。"""

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
    """保存进入准入和状态机之前的语义提案。"""

    proposal_id: str
    memory_type: str
    identity_fields: Mapping[str, Any]
    value_fields: Mapping[str, Any]
    semantic: SemanticAssessment | NormalizedSemanticAssessment
    epistemic_status: EpistemicStatus
    suggested_scope_refs: tuple[ScopeRef, ...]
    related_memory_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: float
    extractor_version: str
    field_evidence_refs: Mapping[str, tuple[EvidenceRef, ...]] = field(default_factory=dict)
    related_slot_ids: tuple[str, ...] = ()
    related_claim_ids: tuple[str, ...] = ()
    model_id: str | None = None
    prompt_version: str = "memory_semantic_proposal_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.memory_type:
            raise ValueError("proposal_id and memory_type are required")
        object.__setattr__(self, "identity_fields", MappingProxyType(dict(self.identity_fields)))
        object.__setattr__(self, "value_fields", MappingProxyType(dict(self.value_fields)))
        object.__setattr__(
            self,
            "field_evidence_refs",
            MappingProxyType({str(key): tuple(refs) for key, refs in dict(self.field_evidence_refs).items()}),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be a finite number between 0 and 1") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite number between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
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
                sorted(self.all_related_memory_ids),
                sorted((ref.event_id, ref.content_hash, ref.span_start, ref.span_end) for ref in self.evidence_refs),
                {
                    key: sorted((ref.event_id, ref.content_hash, ref.span_start, ref.span_end) for ref in refs)
                    for key, refs in self.field_evidence_refs.items()
                },
            ],
            length=40,
        )

    @property
    def all_related_memory_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.related_memory_ids, *self.related_slot_ids, *self.related_claim_ids)))
