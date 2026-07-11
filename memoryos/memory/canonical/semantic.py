"""Fail-closed semantic normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from enum import Enum
from typing import TypeVar

from memoryos.memory.canonical.proposal import (
    Commitment,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
)

SemanticEnum = TypeVar("SemanticEnum", bound=Enum)


class MemorySemanticNormalizer:
    """Normalize only explicit, versioned aliases; preserve all uncertainty."""

    VERSION = "memory_semantic_alias_v2"

    _SPEECH = {
        "observation": SpeechAct.OBSERVATION,
        "proposal": SpeechAct.PROPOSAL,
        "recommendation": SpeechAct.PROPOSAL,
        "future_option": SpeechAct.PROPOSAL,
        "possible_alternative": SpeechAct.PROPOSAL,
        "exploratory_alternative": SpeechAct.PROPOSAL,
        "under_consideration": SpeechAct.EVALUATION_REQUEST,
        "evaluation_request": SpeechAct.EVALUATION_REQUEST,
        "confirmation": SpeechAct.CONFIRMATION,
        "correction": SpeechAct.CORRECTION,
        "retraction": SpeechAct.RETRACTION,
        "rejection": SpeechAct.REJECTION,
        "unknown": SpeechAct.UNKNOWN,
        "schema_mismatch": SpeechAct.SCHEMA_MISMATCH,
    }
    _COMMITMENT = {
        "weak": Commitment.WEAK,
        "possible": Commitment.WEAK,
        "exploratory": Commitment.EXPLORATORY,
        "exploratory_alternative": Commitment.EXPLORATORY,
        "future_option": Commitment.EXPLORATORY,
        "recommendation": Commitment.EXPLORATORY,
        "intended": Commitment.INTENDED,
        "plan": Commitment.INTENDED,
        "confirmed": Commitment.CONFIRMED,
        "committed": Commitment.CONFIRMED,
        "unknown": Commitment.UNKNOWN,
        "schema_mismatch": Commitment.SCHEMA_MISMATCH,
    }
    _TEMPORAL = {item.value.lower(): item for item in TemporalScope}
    _RELATION = {item.value.lower(): item for item in SemanticRelation}
    _RELATION.update(
        {"possible_alternative": SemanticRelation.ALTERNATIVE, "exploratory_alternative": SemanticRelation.ALTERNATIVE}
    )

    def normalize(self, proposal: MemorySemanticProposal) -> MemorySemanticProposal:
        semantic = proposal.semantic
        if isinstance(semantic, NormalizedSemanticAssessment):
            metadata = {
                **dict(proposal.metadata),
                "semantic_normalization_version": self.VERSION,
                "semantic_normalization_errors": list(semantic.schema_errors),
            }
            return replace(proposal, metadata=metadata)
        normalized = NormalizedSemanticAssessment(
            speech_act=self._map(self._SPEECH, semantic.speech_act, SpeechAct.UNKNOWN, SpeechAct.SCHEMA_MISMATCH),
            commitment=self._map(
                self._COMMITMENT,
                semantic.commitment,
                Commitment.UNKNOWN,
                Commitment.SCHEMA_MISMATCH,
            ),
            temporal_scope=self._map(
                self._TEMPORAL,
                semantic.temporal_scope,
                TemporalScope.UNKNOWN,
                TemporalScope.SCHEMA_MISMATCH,
            ),
            relation_to_existing=self._map(
                self._RELATION,
                semantic.relation_to_existing,
                SemanticRelation.UNKNOWN,
                SemanticRelation.SCHEMA_MISMATCH,
            ),
        )
        metadata = {
            **dict(proposal.metadata),
            "semantic_normalization_version": self.VERSION,
            "semantic_normalization_errors": list(normalized.schema_errors),
        }
        return replace(proposal, semantic=normalized, metadata=metadata)

    def _map(
        self,
        mapping: Mapping[str, SemanticEnum],
        value: str,
        unknown: SemanticEnum,
        mismatch: SemanticEnum,
    ) -> SemanticEnum:
        raw = value.value if isinstance(value, Enum) else value
        normalized = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            return unknown
        return mapping.get(normalized, mismatch)
