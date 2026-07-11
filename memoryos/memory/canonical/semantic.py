"""记忆系统里的语义。"""

from __future__ import annotations

from dataclasses import replace

from memoryos.memory.canonical.proposal import (
    Commitment,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
)


class MemorySemanticNormalizer:
    """负责 MemorySemanticNormalizer 这部分逻辑。"""

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
    }
    _TEMPORAL = {item.value.lower(): item for item in TemporalScope}
    _RELATION = {item.value.lower(): item for item in SemanticRelation}
    _RELATION.update(
        {"possible_alternative": SemanticRelation.ALTERNATIVE, "exploratory_alternative": SemanticRelation.ALTERNATIVE}
    )

    def normalize(self, proposal: MemorySemanticProposal) -> MemorySemanticProposal:
        """处理 normalize 这一步。"""

        semantic = proposal.semantic
        if isinstance(semantic, NormalizedSemanticAssessment):
            return proposal
        normalized = NormalizedSemanticAssessment(
            speech_act=self._map(self._SPEECH, semantic.speech_act, SpeechAct.OBSERVATION),
            commitment=self._map(self._COMMITMENT, semantic.commitment, Commitment.WEAK),
            temporal_scope=self._map(self._TEMPORAL, semantic.temporal_scope, TemporalScope.UNSPECIFIED),
            relation_to_existing=self._map(self._RELATION, semantic.relation_to_existing, SemanticRelation.UNRELATED),
        )
        return replace(proposal, semantic=normalized)

    def _map(self, mapping, value: str, default):  # noqa: ANN001, ANN202
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return mapping.get(normalized, default)
