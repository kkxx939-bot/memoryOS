"""Stable, lazily resolved Canonical Memory exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_ATTRS: dict[str, tuple[str, str]] = {}


def _exports(module: str, *names: str) -> None:
    _PUBLIC_ATTRS.update({name: (module, name) for name in names})


_exports(
    "memoryos.memory.canonical.admission",
    "ProposalAdmissionDecision",
    "ProposalAdmissionGate",
    "ProposalAdmissionResult",
)
_exports("memoryos.memory.canonical.episode", "EvidenceEpisode", "SessionArchiveEpisodeAdapter")
_exports("memoryos.memory.canonical.event", "ActorRef", "EventEnvelope", "OriginContext", "SubjectRef")
_exports(
    "memoryos.memory.canonical.evidence",
    "EVIDENCE_SIGNAL_PHRASES",
    "EvidenceRef",
    "EvidenceSignalKind",
    "EvidenceSignalMatch",
    "EvidenceSignalMatcher",
    "ProposalEvidenceValidator",
    "ProposalValidationResult",
    "bind_field_evidence",
)
_exports(
    "memoryos.memory.canonical.formation",
    "CandidateProposalAdapter",
    "CanonicalFormationResult",
    "CanonicalMemoryFormationService",
)
_exports(
    "memoryos.memory.canonical.identity",
    "IDENTITY_ALGORITHM_V2",
    "AliasRegistry",
    "ResolvedMemoryIdentity",
    "StableMemoryIdentityResolver",
    "canonical_identity_json",
    "canonical_identity_value",
)
_exports("memoryos.memory.canonical.prefetch", "ExistingMemoryPrefetcher", "PrefetchedMemory")
_exports(
    "memoryos.memory.canonical.projection",
    "CanonicalMemoryProjector",
    "MemoryProjectionWorker",
    "ProjectionResult",
)
_exports(
    "memoryos.memory.canonical.projection_state",
    "ProjectionIntegrityError",
    "ProjectionRecord",
    "ProjectionRecordStore",
    "ProjectionStatus",
    "ProjectionStepStatus",
)
_exports(
    "memoryos.memory.canonical.promotion_policy",
    "CANONICAL_PIPELINE_GATES",
    "CanonicalPromotionDecision",
    "CanonicalPromotionFacts",
    "CanonicalPromotionPolicy",
    "CanonicalPromotionResult",
)
_exports(
    "memoryos.memory.canonical.proposal",
    "Atomicity",
    "Attribution",
    "Commitment",
    "Durability",
    "EpistemicStatus",
    "MemorySemanticProposal",
    "ModalForce",
    "NormalizedSemanticAssessment",
    "PendingMemoryProposal",
    "PendingReason",
    "PendingReasonPolicy",
    "SemanticAssessment",
    "SemanticRelation",
    "SpeechAct",
    "TemporalScope",
    "UtteranceMode",
)
_exports("memoryos.memory.canonical.reconcile", "AmbiguousSemanticReconciler", "MemorySemanticReconciler")
_exports("memoryos.memory.canonical.repository", "CanonicalMemoryRepository")
_exports(
    "memoryos.memory.canonical.retrieval",
    "CanonicalInvariantViolation",
    "CanonicalMemoryQuery",
    "CanonicalQueryIntent",
    "OfflineCanonicalMemoryRetriever",
)
_exports("memoryos.memory.canonical.salience", "EpisodeSalienceGate", "SalienceDecision")
_exports(
    "memoryos.memory.canonical.scope",
    "CORE_SCOPE_KINDS",
    "HIERARCHICAL_SCOPE_KINDS",
    "AuthorityPolicy",
    "MemoryScope",
    "ScopeRef",
    "ScopeResolutionSource",
    "ScopeSelector",
    "VisibilityPolicy",
    "scope_from_external",
    "scope_key_candidates_from_payload",
    "scope_key_from_payload",
)
_exports("memoryos.memory.canonical.semantic", "MemorySemanticNormalizer")
_exports(
    "memoryos.memory.canonical.state",
    "ActiveClaimInvariantError",
    "CanonicalMemoryInvariantError",
    "ClaimState",
    "MemoryClaim",
    "MemoryRevision",
    "MemorySlot",
    "MissingClaimInvariantError",
    "RevisionSequenceError",
    "TransitionProfile",
)
_exports(
    "memoryos.memory.canonical.transaction",
    "MemoryTransactionPlan",
    "MemoryTransactionPlanner",
    "PlannedMemoryOperation",
    "RevisionConflictError",
)
_exports(
    "memoryos.memory.canonical.transition",
    "MemoryStateTransition",
    "MemoryTransitionPolicy",
    "PendingSemanticReconciliation",
)

del _exports


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = list(_PUBLIC_ATTRS)
