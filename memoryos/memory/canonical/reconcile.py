"""记忆系统里的对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memoryos.memory.canonical.identity import ResolvedMemoryIdentity
from memoryos.memory.canonical.proposal import (
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticRelation,
)
from memoryos.memory.canonical.state import MemoryClaim, MemorySlot


@dataclass(frozen=True)
class ReconciliationResult:
    """保存 ReconciliationResult 需要的这组数据。"""

    relation: SemanticRelation
    slot: MemorySlot | None
    claim: MemoryClaim | None
    active_claim: MemoryClaim | None
    claims: tuple[MemoryClaim, ...]
    deterministic: bool = True


class AmbiguousSemanticReconciler(Protocol):
    """负责 AmbiguousSemanticReconciler 这部分逻辑。"""

    def reconcile_relation(
        self,
        proposal: MemorySemanticProposal,
        active_claim: MemoryClaim | None,
        claims: tuple[MemoryClaim, ...],
    ) -> SemanticRelation:
        """结合当前状态解析出确定结果。"""

        ...


class MemorySemanticReconciler:
    """负责 MemorySemanticReconciler 这部分逻辑。"""

    def __init__(self, ambiguous_reconciler: AmbiguousSemanticReconciler | None = None) -> None:
        self.ambiguous_reconciler = ambiguous_reconciler

    def reconcile(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        *,
        slot: MemorySlot | None,
        claims: tuple[MemoryClaim, ...],
    ) -> ReconciliationResult:
        """结合当前状态解析出确定结果。"""

        claim = next((item for item in claims if item.claim_id == identity.claim_id), None)
        active = next((item for item in claims if item.current.state == "ACTIVE"), None)
        deterministic = True
        if claim is not None:
            relation = self._same_claim_relation(proposal, claim)
        elif slot is not None and claims:
            semantic = proposal.semantic
            relation = (
                semantic.relation_to_existing
                if isinstance(semantic, NormalizedSemanticAssessment)
                else SemanticRelation.UNRELATED
            )
            if relation == SemanticRelation.UNRELATED:
                relation = self._structured_relation(proposal, active, claims)
            if relation == SemanticRelation.UNRELATED and self.ambiguous_reconciler is not None:
                relation = SemanticRelation(self.ambiguous_reconciler.reconcile_relation(proposal, active, claims))
                deterministic = False
            if relation == SemanticRelation.UNRELATED:
                relation = SemanticRelation.ALTERNATIVE
        else:
            relation = SemanticRelation.UNRELATED
        return ReconciliationResult(relation, slot, claim, active, claims, deterministic)

    def _same_claim_relation(self, proposal: MemorySemanticProposal, claim: MemoryClaim) -> SemanticRelation:
        incoming = dict(proposal.value_fields)
        current = dict(claim.current.value_fields)
        if incoming == current:
            return SemanticRelation.DUPLICATE
        if all(key not in current or current[key] == value for key, value in incoming.items()):
            return SemanticRelation.SUPPLEMENTS
        semantic = proposal.semantic
        suggested = (
            semantic.relation_to_existing
            if isinstance(semantic, NormalizedSemanticAssessment)
            else SemanticRelation.UNRELATED
        )
        return suggested if suggested != SemanticRelation.UNRELATED else SemanticRelation.CORRECTS

    def _structured_relation(
        self,
        proposal: MemorySemanticProposal,
        active: MemoryClaim | None,
        claims: tuple[MemoryClaim, ...],
    ) -> SemanticRelation:
        related = set(proposal.all_related_memory_ids)
        if related and any(claim.claim_id in related or claim.uri in related for claim in claims):
            return SemanticRelation.ALTERNATIVE
        if active is not None and active.canonical_value != "":
            return SemanticRelation.ALTERNATIVE
        return SemanticRelation.UNRELATED
