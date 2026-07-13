"""记忆系统里的提案。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.ids import stable_hash
from memoryos.core.time import utc_now
from memoryos.memory.canonical.event import canonicalize, immutable_snapshot
from memoryos.memory.canonical.scope import MemoryScope, ScopeRef

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
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class Commitment(str, Enum):
    """表示提案里的承诺强弱，不等同于 Claim 状态。"""

    WEAK = "WEAK"
    EXPLORATORY = "EXPLORATORY"
    INTENDED = "INTENDED"
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class TemporalScope(str, Enum):
    """表示一条语义说的是过去、现在还是未来。"""

    PAST = "PAST"
    CURRENT = "CURRENT"
    FUTURE = "FUTURE"
    UNSPECIFIED = "UNSPECIFIED"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class SemanticRelation(str, Enum):
    """列出新提案和已有记忆之间的关系。"""

    UNRELATED = "UNRELATED"
    DUPLICATE = "DUPLICATE"
    SUPPLEMENTS = "SUPPLEMENTS"
    ALTERNATIVE = "ALTERNATIVE"
    CONTRADICTS = "CONTRADICTS"
    CORRECTS = "CORRECTS"
    SUPERSEDES = "SUPERSEDES"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class UtteranceMode(str, Enum):
    """Describe how the source presents the candidate proposition."""

    ASSERTION = "ASSERTION"
    DIRECTIVE = "DIRECTIVE"
    QUESTION = "QUESTION"
    HYPOTHETICAL = "HYPOTHETICAL"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class Attribution(str, Enum):
    """Distinguish a source actor's claim from quoted or reported content."""

    SOURCE_ACTOR = "SOURCE_ACTOR"
    THIRD_PARTY = "THIRD_PARTY"
    QUOTED = "QUOTED"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class Durability(str, Enum):
    """State whether a proposition remains useful beyond the current turn."""

    DURABLE = "DURABLE"
    TRANSIENT = "TRANSIENT"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class ModalForce(str, Enum):
    """Represent the normative direction carried by a proposition."""

    NONE = "NONE"
    REQUIRE = "REQUIRE"
    FORBID = "FORBID"
    ALLOW = "ALLOW"
    PREFER = "PREFER"
    DISCOURAGE = "DISCOURAGE"
    CONDITIONAL_REQUIRE = "CONDITIONAL_REQUIRE"
    CONDITIONAL_FORBID = "CONDITIONAL_FORBID"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class Atomicity(str, Enum):
    """State whether one proposal contains exactly one proposition."""

    ATOMIC = "ATOMIC"
    COMPOUND = "COMPOUND"
    UNKNOWN = "UNKNOWN"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


@dataclass(frozen=True)
class SemanticAssessment:
    """保存 SemanticAssessment 需要的这组数据。"""

    speech_act: str
    commitment: str
    temporal_scope: str
    relation_to_existing: str = "unrelated"
    utterance_mode: str = "unknown"
    attribution: str = "unknown"
    durability: str = "unknown"
    modal_force: str = "unknown"
    atomicity: str = "unknown"


@dataclass(frozen=True)
class NormalizedSemanticAssessment:
    """保存 NormalizedSemanticAssessment 需要的这组数据。"""

    speech_act: SpeechAct
    commitment: Commitment
    temporal_scope: TemporalScope
    relation_to_existing: SemanticRelation
    utterance_mode: UtteranceMode = UtteranceMode.UNKNOWN
    attribution: Attribution = Attribution.UNKNOWN
    durability: Durability = Durability.UNKNOWN
    modal_force: ModalForce = ModalForce.UNKNOWN
    atomicity: Atomicity = Atomicity.UNKNOWN

    @property
    def schema_safe(self) -> bool:
        return (
            self.speech_act not in {SpeechAct.UNKNOWN, SpeechAct.SCHEMA_MISMATCH}
            and self.commitment not in {Commitment.UNKNOWN, Commitment.SCHEMA_MISMATCH}
            and self.temporal_scope not in {TemporalScope.UNKNOWN, TemporalScope.SCHEMA_MISMATCH}
            and self.relation_to_existing
            not in {SemanticRelation.UNKNOWN, SemanticRelation.AMBIGUOUS, SemanticRelation.SCHEMA_MISMATCH}
            and self.utterance_mode not in {UtteranceMode.UNKNOWN, UtteranceMode.SCHEMA_MISMATCH}
            and self.attribution not in {Attribution.UNKNOWN, Attribution.SCHEMA_MISMATCH}
            and self.durability not in {Durability.UNKNOWN, Durability.SCHEMA_MISMATCH}
            and self.modal_force not in {ModalForce.UNKNOWN, ModalForce.SCHEMA_MISMATCH}
            and self.atomicity not in {Atomicity.UNKNOWN, Atomicity.SCHEMA_MISMATCH}
        )

    @property
    def schema_errors(self) -> tuple[str, ...]:
        errors = []
        for field_name in (
            "speech_act",
            "commitment",
            "temporal_scope",
            "relation_to_existing",
            "utterance_mode",
            "attribution",
            "durability",
            "modal_force",
            "atomicity",
        ):
            value = getattr(self, field_name)
            if str(value.value) in {"UNKNOWN", "AMBIGUOUS", "SCHEMA_MISMATCH"}:
                errors.append(f"semantic_{field_name}_{str(value.value).lower()}")
        return tuple(errors)

    def to_dict(self) -> dict[str, str]:
        return {
            "speech_act": self.speech_act.value,
            "commitment": self.commitment.value,
            "temporal_scope": self.temporal_scope.value,
            "relation_to_existing": self.relation_to_existing.value,
            "utterance_mode": self.utterance_mode.value,
            "attribution": self.attribution.value,
            "durability": self.durability.value,
            "modal_force": self.modal_force.value,
            "atomicity": self.atomicity.value,
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
    prompt_version: str = "memory_semantic_proposal_v2"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    semantic_contract_version: str = "v2"
    atomic_evidence_ref: EvidenceRef | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.memory_type:
            raise ValueError("proposal_id and memory_type are required")
        object.__setattr__(self, "identity_fields", immutable_snapshot(dict(self.identity_fields)))
        object.__setattr__(self, "value_fields", immutable_snapshot(dict(self.value_fields)))
        object.__setattr__(
            self,
            "field_evidence_refs",
            MappingProxyType({str(key): tuple(refs) for key, refs in dict(self.field_evidence_refs).items()}),
        )
        object.__setattr__(self, "metadata", immutable_snapshot(dict(self.metadata)))
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
        semantic = (
            self.semantic.to_dict()
            if isinstance(self.semantic, NormalizedSemanticAssessment)
            else {
                "speech_act": self.semantic.speech_act,
                "commitment": self.semantic.commitment,
                "temporal_scope": self.semantic.temporal_scope,
                "relation_to_existing": self.semantic.relation_to_existing,
                "utterance_mode": self.semantic.utterance_mode,
                "attribution": self.semantic.attribution,
                "durability": self.semantic.durability,
                "modal_force": self.semantic.modal_force,
                "atomicity": self.semantic.atomicity,
            }
        )
        return stable_hash(
            [
                self.memory_type,
                dict(self.identity_fields),
                dict(self.value_fields),
                semantic,
                self.epistemic_status.value,
                self.semantic_contract_version,
                (
                    (
                        self.atomic_evidence_ref.event_id,
                        self.atomic_evidence_ref.content_hash,
                        self.atomic_evidence_ref.span_start,
                        self.atomic_evidence_ref.span_end,
                    )
                    if self.atomic_evidence_ref is not None
                    else None
                ),
                sorted(scope.key for scope in self.suggested_scope_refs),
                sorted(self.all_related_memory_ids),
                sorted(
                    (
                        ref.event_id,
                        ref.content_hash,
                        ref.span_start if ref.span_start is not None else -1,
                        ref.span_end if ref.span_end is not None else -1,
                    )
                    for ref in self.evidence_refs
                ),
                {
                    key: sorted(
                        (
                            ref.event_id,
                            ref.content_hash,
                            ref.span_start if ref.span_start is not None else -1,
                            ref.span_end if ref.span_end is not None else -1,
                        )
                        for ref in refs
                    )
                    for key, refs in self.field_evidence_refs.items()
                },
            ],
            length=40,
        )

    @property
    def all_related_memory_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.related_memory_ids, *self.related_slot_ids, *self.related_claim_ids)))

    def to_dict(self) -> dict[str, Any]:
        semantic = (
            self.semantic.to_dict()
            if isinstance(self.semantic, NormalizedSemanticAssessment)
            else {
                "speech_act": self.semantic.speech_act,
                "commitment": self.semantic.commitment,
                "temporal_scope": self.semantic.temporal_scope,
                "relation_to_existing": self.semantic.relation_to_existing,
                "utterance_mode": self.semantic.utterance_mode,
                "attribution": self.semantic.attribution,
                "durability": self.semantic.durability,
                "modal_force": self.semantic.modal_force,
                "atomicity": self.semantic.atomicity,
            }
        )
        return {
            "proposal_id": self.proposal_id,
            "memory_type": self.memory_type,
            "identity_fields": canonicalize(self.identity_fields),
            "value_fields": canonicalize(self.value_fields),
            "semantic": semantic,
            "epistemic_status": self.epistemic_status.value,
            "suggested_scope_refs": [scope.to_dict() for scope in self.suggested_scope_refs],
            "related_memory_ids": list(self.related_memory_ids),
            "related_slot_ids": list(self.related_slot_ids),
            "related_claim_ids": list(self.related_claim_ids),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "field_evidence_refs": {
                field_name: [ref.to_dict() for ref in refs] for field_name, refs in self.field_evidence_refs.items()
            },
            "confidence": self.confidence,
            "extractor_version": self.extractor_version,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "semantic_contract_version": self.semantic_contract_version,
            "atomic_evidence_ref": self.atomic_evidence_ref.to_dict() if self.atomic_evidence_ref is not None else None,
            "metadata": canonicalize(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemorySemanticProposal:
        # Imported lazily because evidence imports this module.
        from memoryos.memory.canonical.evidence import EvidenceRef

        semantic_payload = dict(payload.get("semantic", {}) or {})
        try:
            semantic: SemanticAssessment | NormalizedSemanticAssessment = NormalizedSemanticAssessment(
                SpeechAct(str(semantic_payload["speech_act"]).upper()),
                Commitment(str(semantic_payload["commitment"]).upper()),
                TemporalScope(str(semantic_payload["temporal_scope"]).upper()),
                SemanticRelation(str(semantic_payload.get("relation_to_existing", "UNRELATED")).upper()),
                UtteranceMode(str(semantic_payload.get("utterance_mode", "UNKNOWN")).upper()),
                Attribution(str(semantic_payload.get("attribution", "UNKNOWN")).upper()),
                Durability(str(semantic_payload.get("durability", "UNKNOWN")).upper()),
                ModalForce(str(semantic_payload.get("modal_force", "UNKNOWN")).upper()),
                Atomicity(str(semantic_payload.get("atomicity", "UNKNOWN")).upper()),
            )
        except (KeyError, ValueError):
            semantic = SemanticAssessment(
                str(semantic_payload.get("speech_act", "unknown")),
                str(semantic_payload.get("commitment", "unknown")),
                str(semantic_payload.get("temporal_scope", "unknown")),
                str(semantic_payload.get("relation_to_existing", "unknown")),
                str(semantic_payload.get("utterance_mode", "unknown")),
                str(semantic_payload.get("attribution", "unknown")),
                str(semantic_payload.get("durability", "unknown")),
                str(semantic_payload.get("modal_force", "unknown")),
                str(semantic_payload.get("atomicity", "unknown")),
            )
        evidence_refs = tuple(EvidenceRef(**dict(item)) for item in payload.get("evidence_refs", []) or [])
        by_payload = {stable_hash(dict(ref.to_dict()), length=40): ref for ref in evidence_refs}

        def evidence(item: Mapping[str, Any]) -> EvidenceRef:
            key = stable_hash(dict(item), length=40)
            return by_payload.get(key) or EvidenceRef(**dict(item))

        atomic_payload = payload.get("atomic_evidence_ref")
        atomic_evidence_ref = evidence(dict(atomic_payload)) if isinstance(atomic_payload, Mapping) else None

        return cls(
            proposal_id=str(payload["proposal_id"]),
            memory_type=str(payload["memory_type"]),
            identity_fields=dict(payload.get("identity_fields", {}) or {}),
            value_fields=dict(payload.get("value_fields", {}) or {}),
            semantic=semantic,
            epistemic_status=EpistemicStatus(str(payload.get("epistemic_status", "INFERRED")).upper()),
            suggested_scope_refs=tuple(
                ScopeRef.from_dict(dict(item)) for item in payload.get("suggested_scope_refs", []) or []
            ),
            related_memory_ids=tuple(str(item) for item in payload.get("related_memory_ids", []) or []),
            related_slot_ids=tuple(str(item) for item in payload.get("related_slot_ids", []) or []),
            related_claim_ids=tuple(str(item) for item in payload.get("related_claim_ids", []) or []),
            evidence_refs=evidence_refs,
            field_evidence_refs={
                str(field_name): tuple(evidence(dict(item)) for item in refs)
                for field_name, refs in dict(payload.get("field_evidence_refs", {}) or {}).items()
            },
            confidence=float(payload.get("confidence", 0.0)),
            extractor_version=str(payload.get("extractor_version", "")),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            prompt_version=str(payload.get("prompt_version", "")),
            semantic_contract_version=str(payload.get("semantic_contract_version") or "v2"),
            atomic_evidence_ref=atomic_evidence_ref,
            metadata=dict(payload.get("metadata", {}) or {}),
        )


PENDING_PROPOSAL_STATES = frozenset(
    {
        LifecycleState.PENDING,
        LifecycleState.CONFIRMED,
        LifecycleState.REJECTED,
        LifecycleState.EXPIRED,
        LifecycleState.RETRYABLE,
        LifecycleState.RESOLVED,
    }
)

PENDING_PROPOSAL_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.PENDING: frozenset(
        {
            LifecycleState.CONFIRMED,
            LifecycleState.REJECTED,
            LifecycleState.EXPIRED,
            LifecycleState.RETRYABLE,
        }
    ),
    LifecycleState.RETRYABLE: frozenset(
        {
            LifecycleState.PENDING,
            LifecycleState.CONFIRMED,
            LifecycleState.REJECTED,
            LifecycleState.EXPIRED,
        }
    ),
    LifecycleState.CONFIRMED: frozenset({LifecycleState.RESOLVED, LifecycleState.REJECTED, LifecycleState.EXPIRED}),
    LifecycleState.REJECTED: frozenset(),
    LifecycleState.EXPIRED: frozenset(),
    LifecycleState.RESOLVED: frozenset(),
}


class PendingReason(str, Enum):
    REVIEWABLE_DESTRUCTIVE = "REVIEWABLE_DESTRUCTIVE"
    REVIEWABLE_LOW_CONFIDENCE = "REVIEWABLE_LOW_CONFIDENCE"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    NEEDS_SCHEMA_REPAIR = "NEEDS_SCHEMA_REPAIR"
    NEEDS_SCOPE_RESOLUTION = "NEEDS_SCOPE_RESOLUTION"
    RETRYABLE_BACKEND = "RETRYABLE_BACKEND"
    FALLBACK_REQUIRES_REEXTRACTION = "FALLBACK_REQUIRES_REEXTRACTION"
    POLICY_RESTRICTED = "POLICY_RESTRICTED"


@dataclass(frozen=True)
class PendingReasonPolicy:
    confirm: bool
    confirm_and_apply: bool
    retry: bool
    requires_new_proposal: bool = False
    requires_reextraction: bool = False


PENDING_REASON_POLICIES: dict[PendingReason, PendingReasonPolicy] = {
    PendingReason.REVIEWABLE_DESTRUCTIVE: PendingReasonPolicy(True, True, False),
    PendingReason.REVIEWABLE_LOW_CONFIDENCE: PendingReasonPolicy(True, True, False),
    PendingReason.NEEDS_EVIDENCE: PendingReasonPolicy(False, False, False, requires_new_proposal=True),
    PendingReason.NEEDS_SCHEMA_REPAIR: PendingReasonPolicy(False, False, False, requires_new_proposal=True),
    PendingReason.NEEDS_SCOPE_RESOLUTION: PendingReasonPolicy(False, False, False, requires_new_proposal=True),
    PendingReason.RETRYABLE_BACKEND: PendingReasonPolicy(False, False, True),
    PendingReason.FALLBACK_REQUIRES_REEXTRACTION: PendingReasonPolicy(
        False,
        False,
        False,
        requires_new_proposal=True,
        requires_reextraction=True,
    ),
    PendingReason.POLICY_RESTRICTED: PendingReasonPolicy(False, False, False, requires_new_proposal=True),
}


def classify_pending_reason(value: str | PendingReason) -> PendingReason:
    if isinstance(value, PendingReason):
        return value
    raw = str(value or "").strip()
    try:
        return PendingReason(raw)
    except ValueError:
        normalized = raw.casefold()
    # Finite legacy spellings retained only for migration/test fixtures.  Do
    # not infer review authority from an arbitrary internal reason merely
    # because it contains words such as "review" or "confidence".
    if normalized in {
        "low_confidence_review",
        "review_required",
        "needs review",
        "manual_review",
        "secondary_manual_review",
    }:
        return PendingReason.REVIEWABLE_LOW_CONFIDENCE
    if "fallback" in normalized or "reextract" in normalized:
        return PendingReason.FALLBACK_REQUIRES_REEXTRACTION
    if any(token in normalized for token in ("backend", "transport", "timeout", "rate_limit", "retryable")):
        return PendingReason.RETRYABLE_BACKEND
    if any(token in normalized for token in ("privacy", "policy", "restricted", "security")):
        return PendingReason.POLICY_RESTRICTED
    if "scope" in normalized or "subject" in normalized:
        return PendingReason.NEEDS_SCOPE_RESOLUTION
    # These states describe a proposition that cannot become canonical merely
    # because a reviewer toggles its lifecycle flag.  In particular, a
    # compound/ambiguous v3 candidate must be replaced by new atomic semantic
    # proposals; classifying the word "ambiguous" as low confidence would let
    # it enter CONFIRMED even though resolution must (correctly) reject it.
    if any(
        token in normalized
        for token in (
            "schema",
            "identity",
            "unsupported_memory",
            "not_normalized",
            "contract_not_validated",
            "ambiguous",
            "ambiguous_or_compound",
            "not_active_eligible",
            "modal_force_inconsistent",
            "semantic_inconsistent",
            "commitment_pending",
            "temporality_pending",
            "nonfinal_relation_requires_review",
        )
    ):
        return PendingReason.NEEDS_SCHEMA_REPAIR
    if any(
        token in normalized
        for token in (
            "evidence",
            "atomic",
            "source_ground",
            "source_role",
            "source_not_authoritative",
            "hypothesis_requires_confirmation",
            "admission_score_below_threshold",
            "uncertain",
            "confidence",
        )
    ):
        return PendingReason.NEEDS_EVIDENCE
    if any(
        token in normalized
        for token in (
            "destructive",
            "replacement",
            "retraction",
            "supersed",
            "correct",
            "relation_requires_confirmation",
            "nonfinal_relation_requires_review",
        )
    ):
        return PendingReason.REVIEWABLE_DESTRUCTIVE
    # Unknown internal reasons are never silently upgraded into user
    # authorization.  They require a corrected proposal under a typed policy.
    return PendingReason.POLICY_RESTRICTED


@dataclass(frozen=True)
class PendingMemoryProposal:
    """A durable, reviewable proposal that is never an authoritative Claim."""

    uri: str
    proposal: MemorySemanticProposal
    scope: MemoryScope
    source_role: str
    pending_reason_code: PendingReason | str
    pending_reason_detail: str = ""
    request_identity: str = ""
    related_existing_memory_ids: tuple[str, ...] = ()
    retrieval_views: tuple[str, ...] = ()
    lifecycle_state: LifecycleState = LifecycleState.PENDING
    retry_count: int = 0
    lifecycle_revision: int = 1
    lifecycle_history: tuple[Mapping[str, Any], ...] = ()
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    SCHEMA_VERSION = "canonical_pending_proposal_v1"

    def __post_init__(self) -> None:
        if self.lifecycle_state not in PENDING_PROPOSAL_STATES:
            raise ValueError(f"invalid pending proposal lifecycle state: {self.lifecycle_state.value}")
        if self.retry_count < 0:
            raise ValueError("pending proposal retry count cannot be negative")
        if self.lifecycle_revision < 1:
            raise ValueError("pending proposal lifecycle revision must be positive")
        object.__setattr__(self, "related_existing_memory_ids", tuple(self.related_existing_memory_ids))
        original_reason = (
            self.pending_reason_code.value
            if isinstance(self.pending_reason_code, PendingReason)
            else str(self.pending_reason_code)
        )
        reason = classify_pending_reason(self.pending_reason_code)
        object.__setattr__(self, "pending_reason_code", reason)
        if not self.pending_reason_detail and original_reason != reason.value:
            object.__setattr__(self, "pending_reason_detail", original_reason)
        object.__setattr__(self, "retrieval_views", tuple(self.retrieval_views))
        object.__setattr__(
            self, "lifecycle_history", tuple(immutable_snapshot(dict(item)) for item in self.lifecycle_history)
        )

    @classmethod
    def create(
        cls,
        proposal: MemorySemanticProposal,
        scope: MemoryScope,
        *,
        tenant_id: str,
        owner_user_id: str,
        source_role: str,
        pending_reason_code: str,
        request_identity: str = "",
        related_existing_memory_ids: tuple[str, ...] = (),
        retrieval_views: tuple[str, ...] = (),
        created_at: str = "",
    ) -> PendingMemoryProposal:
        digest = stable_hash(
            [
                tenant_id,
                owner_user_id,
                request_identity,
                proposal.fingerprint,
                scope.canonical_subject.key if scope.canonical_subject is not None else "",
            ],
            length=32,
        )
        timestamp = created_at or utc_now()
        return cls(
            uri=f"memoryos://user/{owner_user_id}/memories/pending/{digest}",
            proposal=proposal,
            scope=scope,
            source_role=source_role,
            pending_reason_code=classify_pending_reason(pending_reason_code),
            pending_reason_detail=(
                pending_reason_code
                if str(pending_reason_code) != classify_pending_reason(pending_reason_code).value
                else ""
            ),
            request_identity=request_identity,
            related_existing_memory_ids=tuple(dict.fromkeys(related_existing_memory_ids)),
            retrieval_views=tuple(dict.fromkeys(retrieval_views)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @property
    def proposal_id(self) -> str:
        return self.proposal.proposal_id

    def to_payload(self) -> dict[str, Any]:
        semantic = self.proposal.to_dict()["semantic"]
        return {
            "proposal_id": self.proposal.proposal_id,
            "memory_type": self.proposal.memory_type,
            "identity_fields": canonicalize(self.proposal.identity_fields),
            "value_fields": canonicalize(self.proposal.value_fields),
            "scope": self.scope.to_dict(),
            "source_role": self.source_role,
            "semantic_assessment": semantic,
            "speech_act": semantic.get("speech_act", ""),
            "commitment": semantic.get("commitment", ""),
            "temporality": semantic.get("temporal_scope", ""),
            "relation": semantic.get("relation_to_existing", ""),
            "semantic_contract_version": self.proposal.semantic_contract_version,
            "atomic_evidence_ref": self.proposal.atomic_evidence_ref.to_dict()
            if self.proposal.atomic_evidence_ref is not None
            else None,
            "epistemic_status": self.proposal.epistemic_status.value,
            "related_memory_ids": list(self.proposal.related_memory_ids),
            "related_slot_ids": list(self.proposal.related_slot_ids),
            "related_claim_ids": list(self.proposal.related_claim_ids),
            "related_existing_memory_ids": list(self.related_existing_memory_ids),
            "evidence_refs": [ref.to_dict() for ref in self.proposal.evidence_refs],
            "field_evidence_refs": {
                field_name: [ref.to_dict() for ref in refs]
                for field_name, refs in self.proposal.field_evidence_refs.items()
            },
            "pending_reason_code": PendingReason(self.pending_reason_code).value,
            "pending_reason_detail": self.pending_reason_detail,
            "request_identity": self.request_identity,
            "extractor_name": self.proposal.extractor_version,
            "model_id": self.proposal.model_id,
            "prompt_version": self.proposal.prompt_version,
            "proposal_fingerprint": self.proposal.fingerprint,
            "model_confidence": self.proposal.confidence,
            "admission_score": self.proposal.metadata.get("admission_score"),
            "admission_threshold": self.proposal.metadata.get("admission_threshold"),
            "admission_score_components": canonicalize(self.proposal.metadata.get("admission_score_components", {})),
            "retrieval_views": list(self.retrieval_views),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "retry_count": self.retry_count,
            "lifecycle_revision": self.lifecycle_revision,
            "lifecycle_history": [dict(item) for item in self.lifecycle_history],
            "lifecycle_state": self.lifecycle_state.value,
            "proposal": self.proposal.to_dict(),
        }

    def content(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)

    def to_context_object(self, *, tenant_id: str, owner_user_id: str) -> ContextObject:
        payload = self.to_payload()
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=f"pending {self.proposal.memory_type}: {self.proposal.proposal_id}",
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            layers=ContextLayers(
                l0_uri=f"{self.uri}/.abstract.md",
                l1_uri=f"{self.uri}/.overview.md",
                l2_uri=f"{self.uri}/content.md",
            ),
            lifecycle_state=self.lifecycle_state,
            metadata={
                "canonical_kind": "pending_proposal",
                "admission": {"decision": "pending", "reason": self.pending_reason_code},
                **payload,
            },
            created_at=self.created_at,
            updated_at=self.updated_at,
            schema_version=self.SCHEMA_VERSION,
        )

    @classmethod
    def from_context_object(cls, obj: ContextObject) -> PendingMemoryProposal:
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "pending_proposal" or obj.schema_version != cls.SCHEMA_VERSION:
            raise ValueError(f"not a canonical pending proposal: {obj.uri}")
        return cls(
            uri=obj.uri,
            proposal=MemorySemanticProposal.from_dict(dict(metadata.get("proposal", {}) or {})),
            scope=MemoryScope.from_dict(dict(metadata.get("scope", {}) or {})),
            source_role=str(metadata.get("source_role", "")),
            pending_reason_code=classify_pending_reason(str(metadata.get("pending_reason_code", ""))),
            pending_reason_detail=str(metadata.get("pending_reason_detail", "")),
            request_identity=str(metadata.get("request_identity", "")),
            related_existing_memory_ids=tuple(
                str(item) for item in metadata.get("related_existing_memory_ids", []) or []
            ),
            retrieval_views=tuple(str(item) for item in metadata.get("retrieval_views", []) or []),
            lifecycle_state=LifecycleState(str(metadata.get("lifecycle_state", obj.lifecycle_state.value))),
            retry_count=int(metadata.get("retry_count", 0)),
            lifecycle_revision=int(metadata.get("lifecycle_revision", 1)),
            lifecycle_history=tuple(
                dict(item) for item in metadata.get("lifecycle_history", []) or [] if isinstance(item, Mapping)
            ),
            created_at=str(metadata.get("created_at") or obj.created_at),
            updated_at=str(metadata.get("updated_at") or obj.updated_at),
        )

    @property
    def reason_policy(self) -> PendingReasonPolicy:
        return PENDING_REASON_POLICIES[PendingReason(self.pending_reason_code)]

    def assert_review_decision(self, decision: str) -> None:
        normalized = str(decision or "").strip().upper()
        allowed = {
            "CONFIRM": self.reason_policy.confirm,
            "CONFIRM_AND_APPLY": self.reason_policy.confirm_and_apply,
            "CORRECT": self.reason_policy.requires_new_proposal,
            "RETRY": self.reason_policy.retry,
            "REJECT": True,
            "EXPIRE": True,
        }
        if normalized not in allowed or not allowed[normalized]:
            raise ValueError(
                f"pending reason {PendingReason(self.pending_reason_code).value} does not allow {normalized or 'empty'}"
            )

    def with_lifecycle(
        self,
        lifecycle_state: LifecycleState,
        *,
        updated_at: str = "",
        retry_increment: bool = False,
        reason: str = "",
        review_command_id: str = "",
        review_decision: str = "",
        review_request_digest: str = "",
    ) -> PendingMemoryProposal:
        review_binding: dict[str, str] = {}
        if review_command_id or review_decision or review_request_digest:
            if not review_command_id or not review_decision or not review_request_digest:
                raise ValueError("pending review lifecycle binding must be complete")
            normalized_decision = review_decision.strip().upper()
            if normalized_decision not in {
                "CONFIRM",
                "CONFIRM_AND_APPLY",
                "CORRECT",
                "REJECT",
                "EXPIRE",
                "RETRY",
            }:
                raise ValueError("pending review lifecycle binding has an invalid decision")
            review_binding = {
                "review_command_id": review_command_id,
                "review_decision": normalized_decision,
                "review_request_digest": review_request_digest,
            }
        if lifecycle_state == self.lifecycle_state:
            if retry_increment and lifecycle_state == LifecycleState.RETRYABLE:
                timestamp = updated_at or utc_now()
                return replace(
                    self,
                    retry_count=self.retry_count + 1,
                    lifecycle_revision=self.lifecycle_revision + 1,
                    lifecycle_history=(
                        *self.lifecycle_history,
                        immutable_snapshot(
                            {
                                "from": self.lifecycle_state.value,
                                "to": lifecycle_state.value,
                                "from_revision": self.lifecycle_revision,
                                "to_revision": self.lifecycle_revision + 1,
                                "reason": reason,
                                "updated_at": timestamp,
                                **review_binding,
                            }
                        ),
                    ),
                    updated_at=timestamp,
                )
            return self
        allowed = PENDING_PROPOSAL_TRANSITIONS.get(self.lifecycle_state, frozenset())
        if lifecycle_state not in allowed:
            raise ValueError(
                f"illegal pending proposal lifecycle transition: {self.lifecycle_state.value}->{lifecycle_state.value}"
            )
        timestamp = updated_at or utc_now()
        return replace(
            self,
            lifecycle_state=lifecycle_state,
            retry_count=self.retry_count + (1 if retry_increment else 0),
            lifecycle_revision=self.lifecycle_revision + 1,
            lifecycle_history=(
                *self.lifecycle_history,
                immutable_snapshot(
                    {
                        "from": self.lifecycle_state.value,
                        "to": lifecycle_state.value,
                        "from_revision": self.lifecycle_revision,
                        "to_revision": self.lifecycle_revision + 1,
                        "reason": reason,
                        "updated_at": timestamp,
                        **review_binding,
                    }
                ),
            ),
            updated_at=timestamp,
        )
