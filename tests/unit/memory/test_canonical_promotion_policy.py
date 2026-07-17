from __future__ import annotations

from dataclasses import fields, replace

import pytest

from memoryos.memory.canonical import (
    CANONICAL_PIPELINE_GATES,
    CanonicalPromotionDecision,
    CanonicalPromotionFacts,
    CanonicalPromotionPolicy,
    TransitionProfile,
)
from memoryos.memory.canonical.admission import ProposalAdmissionDecision
from memoryos.memory.canonical.semantic import EligibilityDisposition
from memoryos.memory.schema import MemoryType

_VALID_CONTEXT_FACTS = CanonicalPromotionFacts(
    evidence_complete=True,
    stable_identity=True,
    scope_resolved=True,
    authority_resolved=True,
)

_VALID_EXPERIENCE_FACTS = replace(
    _VALID_CONTEXT_FACTS,
    distilled_experience=True,
    cross_session_reusable=True,
    admission_threshold_met=True,
)


@pytest.mark.parametrize(
    "memory_type",
    [
        MemoryType.PROFILE,
        MemoryType.PREFERENCE,
        MemoryType.PROJECT_RULE,
        MemoryType.PROJECT_DECISION,
    ],
)
def test_authoritative_state_types_enter_existing_canonical_pipeline(memory_type: MemoryType) -> None:
    result = CanonicalPromotionPolicy().evaluate(memory_type)

    assert result.decision == CanonicalPromotionDecision.PROMOTE
    assert result.profile == TransitionProfile.AUTHORITATIVE_STATE
    assert result.reason == "authoritative_state_candidate"
    assert result.required_gates == CANONICAL_PIPELINE_GATES


@pytest.mark.parametrize("memory_type", [MemoryType.ENTITY, MemoryType.EVENT])
def test_ordinary_observational_context_is_catalog_only(memory_type: MemoryType) -> None:
    result = CanonicalPromotionPolicy().evaluate(memory_type, facts=_VALID_CONTEXT_FACTS)

    assert result.decision == CanonicalPromotionDecision.CATALOG_ONLY
    assert result.profile == TransitionProfile.OBSERVATIONAL
    assert result.unmet_requirements == ("explicit_remember_or_stateful_schema",)


def test_explicit_remember_preserves_canonical_event_support() -> None:
    result = CanonicalPromotionPolicy().evaluate(
        MemoryType.EVENT,
        facts=replace(_VALID_CONTEXT_FACTS, explicit_remember=True),
    )

    assert result.decision == CanonicalPromotionDecision.PROMOTE
    assert result.reason == "explicit_observational_candidate"
    assert result.required_gates == CANONICAL_PIPELINE_GATES


def test_explicit_observational_memory_fails_closed_without_structural_requirements() -> None:
    result = CanonicalPromotionPolicy().evaluate(
        MemoryType.EVENT,
        facts=replace(_VALID_CONTEXT_FACTS, explicit_remember=True, stable_identity=False),
    )

    assert result.decision == CanonicalPromotionDecision.CATALOG_ONLY
    assert result.unmet_requirements == ("stable_identity",)


def test_schema_stateful_observation_needs_deterministic_rule_approval() -> None:
    policy = CanonicalPromotionPolicy(stateful_observational_types=(MemoryType.ENTITY,))

    denied = policy.evaluate(MemoryType.ENTITY, facts=_VALID_CONTEXT_FACTS)
    promoted = policy.evaluate(
        MemoryType.ENTITY,
        facts=replace(_VALID_CONTEXT_FACTS, deterministic_rule_approved=True),
    )

    assert denied.decision == CanonicalPromotionDecision.CATALOG_ONLY
    assert denied.unmet_requirements == ("deterministic_rule_approved",)
    assert promoted.decision == CanonicalPromotionDecision.PROMOTE
    assert promoted.reason == "stateful_observational_candidate"


def test_stateful_observational_declaration_cannot_reclassify_other_profiles() -> None:
    with pytest.raises(ValueError, match="OBSERVATIONAL"):
        CanonicalPromotionPolicy(stateful_observational_types=(MemoryType.PREFERENCE,))


def test_distilled_reusable_experience_can_enter_canonical_pipeline() -> None:
    result = CanonicalPromotionPolicy().evaluate(
        MemoryType.AGENT_EXPERIENCE,
        facts=_VALID_EXPERIENCE_FACTS,
    )

    assert result.decision == CanonicalPromotionDecision.PROMOTE
    assert result.profile == TransitionProfile.EXPERIENCE
    assert result.reason == "reusable_experience_candidate"


@pytest.mark.parametrize(
    ("facts", "unmet"),
    [
        (replace(_VALID_EXPERIENCE_FACTS, distilled_experience=False), "distilled_experience"),
        (replace(_VALID_EXPERIENCE_FACTS, cross_session_reusable=False), "cross_session_reusable"),
        (replace(_VALID_EXPERIENCE_FACTS, evidence_complete=False), "evidence_complete"),
        (replace(_VALID_EXPERIENCE_FACTS, stable_identity=False), "stable_identity"),
        (replace(_VALID_EXPERIENCE_FACTS, scope_resolved=False), "scope_resolved"),
        (replace(_VALID_EXPERIENCE_FACTS, authority_resolved=False), "authority_resolved"),
        (replace(_VALID_EXPERIENCE_FACTS, admission_threshold_met=False), "admission_threshold_met"),
        (replace(_VALID_EXPERIENCE_FACTS, one_off_failure=True), "not_one_off_failure"),
        (replace(_VALID_EXPERIENCE_FACTS, transient_task_state=True), "not_transient_task_state"),
    ],
)
def test_experience_requires_every_deterministic_condition(facts: CanonicalPromotionFacts, unmet: str) -> None:
    result = CanonicalPromotionPolicy().evaluate(
        MemoryType.AGENT_EXPERIENCE,
        facts=facts,
    )

    assert result.decision == CanonicalPromotionDecision.CATALOG_ONLY
    assert unmet in result.unmet_requirements


@pytest.mark.parametrize(
    "facts",
    [
        replace(_VALID_EXPERIENCE_FACTS, raw_tool_log=True),
        replace(_VALID_EXPERIENCE_FACTS, raw_agent_log=True),
    ],
)
def test_raw_logs_never_enter_slot_claim_state(facts: CanonicalPromotionFacts) -> None:
    result = CanonicalPromotionPolicy().evaluate(
        MemoryType.AGENT_EXPERIENCE,
        facts=facts,
    )

    assert result.decision == CanonicalPromotionDecision.CATALOG_ONLY
    assert result.reason == "raw_log_requires_catalog"


def test_existing_semantic_and_admission_policies_remain_authoritative() -> None:
    semantic_archive = CanonicalPromotionPolicy().evaluate(
        MemoryType.PREFERENCE,
        facts=CanonicalPromotionFacts(semantic_eligibility=EligibilityDisposition.ARCHIVE_ONLY),
    )
    admission_reject = CanonicalPromotionPolicy().evaluate(
        MemoryType.PREFERENCE,
        facts=CanonicalPromotionFacts(admission_decision=ProposalAdmissionDecision.RESTRICTED),
    )

    assert semantic_archive.decision == CanonicalPromotionDecision.CATALOG_ONLY
    assert admission_reject.decision == CanonicalPromotionDecision.REJECT


def test_accepted_admission_result_satisfies_experience_threshold_proof() -> None:
    facts = replace(
        _VALID_EXPERIENCE_FACTS,
        admission_threshold_met=False,
        admission_decision=ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
    )

    result = CanonicalPromotionPolicy().evaluate(MemoryType.AGENT_EXPERIENCE, facts=facts)

    assert result.decision == CanonicalPromotionDecision.PROMOTE


def test_unsupported_type_is_rejected_without_falling_back_to_observational() -> None:
    result = CanonicalPromotionPolicy().evaluate("made_up_memory_type")

    assert result.decision == CanonicalPromotionDecision.REJECT
    assert result.profile is None
    assert result.memory_type is None


def test_policy_contract_has_no_model_score_or_free_form_recommendation() -> None:
    fact_names = {field.name for field in fields(CanonicalPromotionFacts)}

    assert "model_score" not in fact_names
    assert "llm_recommendation" not in fact_names
    assert "text" not in fact_names
