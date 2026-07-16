"""Deterministic boundary between ordinary context and canonical memory.

The policy decides whether a context may enter the existing canonical
formation pipeline.  ``PROMOTE`` is not a storage decision: evidence, schema,
identity, scope, authority, admission, reconciliation, transition, and the
canonical transaction still have to succeed.  Ordinary context remains owned
by the context catalog and must not acquire a Slot/Claim merely to be
retrievable.

All inputs are trusted, structured facts produced by deterministic code.  The
policy deliberately accepts neither free-form model output nor an LLM score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from memoryos.memory.canonical.admission import ProposalAdmissionDecision
from memoryos.memory.canonical.semantic import EligibilityDisposition
from memoryos.memory.canonical.state import TransitionProfile, profile_for
from memoryos.memory.schema import MemoryType, MemoryTypeRegistry

CANONICAL_PIPELINE_GATES = (
    "evidence",
    "schema",
    "identity",
    "scope",
    "authority",
    "admission",
    "reconcile",
    "transition",
    "transaction",
)


class CanonicalPromotionDecision(str, Enum):
    """Serving/canonical routing decision for one context."""

    CATALOG_ONLY = "CATALOG_ONLY"
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CanonicalPromotionFacts:
    """Trusted facts used by :class:`CanonicalPromotionPolicy`.

    ``explicit_remember`` must only be set for the structured, authenticated
    ``remember()`` command path.  The validation flags must come from the
    existing evidence, identity, scope, authority, semantic, and admission
    components; callers must not infer them from an LLM response.
    """

    explicit_remember: bool = False
    evidence_complete: bool = False
    stable_identity: bool = False
    scope_resolved: bool = False
    authority_resolved: bool = False
    deterministic_rule_approved: bool = False

    distilled_experience: bool = False
    cross_session_reusable: bool = False
    admission_threshold_met: bool = False
    raw_tool_log: bool = False
    raw_agent_log: bool = False
    one_off_failure: bool = False
    transient_task_state: bool = False

    semantic_eligibility: EligibilityDisposition | None = None
    admission_decision: ProposalAdmissionDecision | None = None


@dataclass(frozen=True)
class CanonicalPromotionResult:
    """A serializable, explainable promotion decision."""

    decision: CanonicalPromotionDecision
    reason: str
    profile: TransitionProfile | None
    memory_type: MemoryType | None
    unmet_requirements: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()
    policy_version: str = "canonical_promotion_v1"

    @property
    def should_promote(self) -> bool:
        return self.decision == CanonicalPromotionDecision.PROMOTE


class CanonicalPromotionPolicy:
    """Route contexts without allowing model output to create canonical state.

    Authoritative state types are candidates for the canonical pipeline by
    default.  Observational contexts stay catalog-only unless they are an
    authenticated explicit ``remember()`` command or their schema is in the
    trusted stateful observational set and a deterministic rule approves the
    instance.  Agent experience has a stricter all-of gate.
    """

    VERSION = "canonical_promotion_v1"

    def __init__(
        self,
        registry: MemoryTypeRegistry | None = None,
        *,
        stateful_observational_types: Iterable[MemoryType | str] = (),
    ) -> None:
        self._registry = registry or MemoryTypeRegistry()
        declared: set[MemoryType] = set()
        for item in stateful_observational_types:
            memory_type = MemoryType(item)
            self._registry.get(memory_type)
            if profile_for(memory_type.value) != TransitionProfile.OBSERVATIONAL:
                raise ValueError("stateful observational declarations must use an OBSERVATIONAL memory type")
            declared.add(memory_type)
        self._stateful_observational_types = frozenset(declared)

    @property
    def stateful_observational_types(self) -> frozenset[MemoryType]:
        return self._stateful_observational_types

    def evaluate(
        self,
        memory_type: MemoryType | str,
        *,
        facts: CanonicalPromotionFacts | None = None,
    ) -> CanonicalPromotionResult:
        """Return the deterministic routing decision for one context."""

        trusted = facts or CanonicalPromotionFacts()
        try:
            normalized_type = MemoryType(memory_type)
        except (TypeError, ValueError):
            return self._result(
                CanonicalPromotionDecision.REJECT,
                "unsupported_memory_type",
                memory_type=None,
                profile=None,
            )
        try:
            self._registry.get(normalized_type)
        except (KeyError, ValueError):
            return self._result(
                CanonicalPromotionDecision.REJECT,
                "unsupported_memory_schema",
                memory_type=normalized_type,
                profile=profile_for(normalized_type.value),
            )

        profile = profile_for(normalized_type.value)
        prior_policy_result = self._apply_existing_policy_results(
            memory_type=normalized_type,
            profile=profile,
            facts=trusted,
        )
        if prior_policy_result is not None:
            return prior_policy_result

        if trusted.raw_tool_log or trusted.raw_agent_log:
            return self._result(
                CanonicalPromotionDecision.CATALOG_ONLY,
                "raw_log_requires_catalog",
                memory_type=normalized_type,
                profile=profile,
                unmet_requirements=("distilled_content",),
            )

        if profile == TransitionProfile.AUTHORITATIVE_STATE:
            return self._result(
                CanonicalPromotionDecision.PROMOTE,
                "authoritative_state_candidate",
                memory_type=normalized_type,
                profile=profile,
                required_gates=CANONICAL_PIPELINE_GATES,
            )
        if profile == TransitionProfile.EXPERIENCE:
            return self._experience_result(normalized_type, trusted)
        return self._observational_result(normalized_type, trusted)

    def _experience_result(
        self,
        memory_type: MemoryType,
        facts: CanonicalPromotionFacts,
    ) -> CanonicalPromotionResult:
        requirements: tuple[tuple[str, bool], ...] = (
            ("distilled_experience", facts.distilled_experience),
            ("cross_session_reusable", facts.cross_session_reusable),
            ("evidence_complete", facts.evidence_complete),
            ("stable_identity", facts.stable_identity),
            ("scope_resolved", facts.scope_resolved),
            ("authority_resolved", facts.authority_resolved),
            (
                "admission_threshold_met",
                facts.admission_threshold_met
                or facts.admission_decision == ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
            ),
            ("not_one_off_failure", not facts.one_off_failure),
            ("not_transient_task_state", not facts.transient_task_state),
        )
        unmet = tuple(name for name, satisfied in requirements if not satisfied)
        if unmet:
            return self._result(
                CanonicalPromotionDecision.CATALOG_ONLY,
                "experience_requirements_not_met",
                memory_type=memory_type,
                profile=TransitionProfile.EXPERIENCE,
                unmet_requirements=unmet,
            )
        return self._result(
            CanonicalPromotionDecision.PROMOTE,
            "reusable_experience_candidate",
            memory_type=memory_type,
            profile=TransitionProfile.EXPERIENCE,
            required_gates=CANONICAL_PIPELINE_GATES,
        )

    def _observational_result(
        self,
        memory_type: MemoryType,
        facts: CanonicalPromotionFacts,
    ) -> CanonicalPromotionResult:
        schema_stateful = memory_type in self._stateful_observational_types
        if not facts.explicit_remember and not schema_stateful:
            return self._result(
                CanonicalPromotionDecision.CATALOG_ONLY,
                "observational_context_default_catalog",
                memory_type=memory_type,
                profile=TransitionProfile.OBSERVATIONAL,
                unmet_requirements=("explicit_remember_or_stateful_schema",),
            )

        requirements: tuple[tuple[str, bool], ...] = (
            ("evidence_complete", facts.evidence_complete),
            ("stable_identity", facts.stable_identity),
            ("scope_resolved", facts.scope_resolved),
            ("authority_resolved", facts.authority_resolved),
        )
        if schema_stateful and not facts.explicit_remember:
            requirements = (*requirements, ("deterministic_rule_approved", facts.deterministic_rule_approved))
        unmet = tuple(name for name, satisfied in requirements if not satisfied)
        if unmet:
            return self._result(
                CanonicalPromotionDecision.CATALOG_ONLY,
                "observational_promotion_requirements_not_met",
                memory_type=memory_type,
                profile=TransitionProfile.OBSERVATIONAL,
                unmet_requirements=unmet,
            )
        reason = "explicit_observational_candidate" if facts.explicit_remember else "stateful_observational_candidate"
        return self._result(
            CanonicalPromotionDecision.PROMOTE,
            reason,
            memory_type=memory_type,
            profile=TransitionProfile.OBSERVATIONAL,
            required_gates=CANONICAL_PIPELINE_GATES,
        )

    def _apply_existing_policy_results(
        self,
        *,
        memory_type: MemoryType,
        profile: TransitionProfile,
        facts: CanonicalPromotionFacts,
    ) -> CanonicalPromotionResult | None:
        if facts.semantic_eligibility == EligibilityDisposition.REJECT:
            return self._result(
                CanonicalPromotionDecision.REJECT,
                "semantic_eligibility_rejected",
                memory_type=memory_type,
                profile=profile,
            )
        if facts.semantic_eligibility in {
            EligibilityDisposition.ARCHIVE_ONLY,
            EligibilityDisposition.PENDING,
        }:
            return self._result(
                CanonicalPromotionDecision.CATALOG_ONLY,
                "semantic_eligibility_not_canonical",
                memory_type=memory_type,
                profile=profile,
                unmet_requirements=("semantic_eligibility",),
            )
        if facts.admission_decision in {
            ProposalAdmissionDecision.REJECT,
            ProposalAdmissionDecision.PRIVATE_ONLY,
            ProposalAdmissionDecision.RESTRICTED,
        }:
            return self._result(
                CanonicalPromotionDecision.REJECT,
                "admission_policy_rejected",
                memory_type=memory_type,
                profile=profile,
            )
        if facts.admission_decision in {
            ProposalAdmissionDecision.ARCHIVE_ONLY,
            ProposalAdmissionDecision.PENDING,
        }:
            return self._result(
                CanonicalPromotionDecision.CATALOG_ONLY,
                "admission_policy_not_canonical",
                memory_type=memory_type,
                profile=profile,
                unmet_requirements=("admission",),
            )
        return None

    def _result(
        self,
        decision: CanonicalPromotionDecision,
        reason: str,
        *,
        memory_type: MemoryType | None,
        profile: TransitionProfile | None,
        unmet_requirements: tuple[str, ...] = (),
        required_gates: tuple[str, ...] = (),
    ) -> CanonicalPromotionResult:
        return CanonicalPromotionResult(
            decision=decision,
            reason=reason,
            profile=profile,
            memory_type=memory_type,
            unmet_requirements=unmet_requirements,
            required_gates=required_gates,
            policy_version=self.VERSION,
        )
