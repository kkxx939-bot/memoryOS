from __future__ import annotations

from dataclasses import replace

import pytest

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import (
    Commitment,
    EpistemicStatus,
    EvidenceRef,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    ProposalAdmissionDecision,
    ProposalAdmissionGate,
    ProposalEvidenceValidator,
    ScopeSelector,
    SemanticAssessment,
    SemanticRelation,
    SessionArchiveEpisodeAdapter,
    SpeechAct,
    TemporalScope,
    VisibilityPolicy,
)


def _episode():  # noqa: ANN202
    return SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "PostgreSQL is a future option; SQLite remains active."}],
            metadata={"tenant_id": "t1", "project_id": "memoryos", "connect": {"adapter_id": "codex"}},
        )
    )


def _proposal(episode=None):  # noqa: ANN001, ANN202
    episode = episode or _episode()
    event = episode.events[0]
    assert episode.origin.primary_scope is not None
    return MemorySemanticProposal(
        proposal_id="p1",
        memory_type="project_decision",
        identity_fields={"decision_topic": "PostgreSQL"},
        value_fields={"value": "future option"},
        semantic=SemanticAssessment("future_option", "exploratory_alternative", "future", "alternative"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=(EvidenceRef.from_event(event, source_uri=episode.source_uris[0]),),
        confidence=0.9,
        extractor_version="fake-v1",
        model_id="fake",
    )


def test_evidence_ref_validates_event_hash_and_supported_fields() -> None:
    episode = _episode()
    result = ProposalEvidenceValidator().validate(_proposal(episode), episode)
    assert result.valid
    assert result.unsupported_fields == ()


def test_bad_event_hash_span_and_missing_core_support_cannot_be_explicit_fact() -> None:
    episode = _episode()
    proposal = _proposal(episode)
    bad_ref = replace(proposal.evidence_refs[0], content_hash="bad", span_start=0, span_end=999)
    unsupported = replace(
        proposal, value_fields={"reason": "because the benchmark was faster"}, evidence_refs=(bad_ref,)
    )
    result = ProposalEvidenceValidator().validate(unsupported, episode)
    assert not result.valid
    assert result.proposal.epistemic_status == EpistemicStatus.INFERRED
    assert "value.reason" in result.unsupported_fields


def test_unknown_event_is_rejected() -> None:
    episode = _episode()
    proposal = _proposal(episode)
    result = ProposalEvidenceValidator().validate(
        replace(proposal, evidence_refs=(replace(proposal.evidence_refs[0], event_id="missing"),)), episode
    )
    assert not result.valid and "unknown_event:missing" in result.errors


def test_semantic_aliases_normalize_to_finite_enums() -> None:
    normalized = MemorySemanticNormalizer().normalize(_proposal())
    assert normalized.semantic.speech_act == SpeechAct.PROPOSAL
    assert normalized.semantic.commitment == Commitment.EXPLORATORY
    assert normalized.semantic.temporal_scope == TemporalScope.FUTURE
    assert normalized.semantic.relation_to_existing == SemanticRelation.ALTERNATIVE


@pytest.mark.parametrize("alias", ["future_option", "recommendation", "exploratory_alternative"])
def test_semantic_proposal_aliases_converge_to_proposal_exploratory(alias: str) -> None:
    proposal = _proposal()
    normalized = MemorySemanticNormalizer().normalize(
        replace(
            proposal,
            semantic=SemanticAssessment(alias, alias, "future", "exploratory_alternative"),
        )
    )
    assert normalized.semantic.speech_act == SpeechAct.PROPOSAL
    assert normalized.semantic.commitment == Commitment.EXPLORATORY
    assert normalized.semantic.relation_to_existing == SemanticRelation.ALTERNATIVE


def test_admission_rejects_llm_scope_outside_legal_candidates_and_cross_tenant_visibility() -> None:
    episode = _episode()
    proposal = _proposal(episode)
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert episode.origin.primary_scope is not None
    legal_scope = MemoryScope(
        applicability=ScopeSelector((episode.origin.primary_scope,)),
        visibility=VisibilityPolicy("t1"),
        origin_refs=episode.origin.scope_refs,
    )
    assert (
        ProposalAdmissionGate()
        .evaluate(validation, episode=episode, memory_scope=legal_scope, source_role="user")
        .decision
        == ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE
    )
    cross_tenant = replace(legal_scope, visibility=VisibilityPolicy("t2"))
    assert (
        ProposalAdmissionGate()
        .evaluate(validation, episode=episode, memory_scope=cross_tenant, source_role="user")
        .decision
        == ProposalAdmissionDecision.REJECT
    )


def test_direct_semantic_proposal_with_secret_is_restricted() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="secret-session",
            archive_uri="memoryos://user/u1/sessions/history/secret-session",
            messages=[{"id": "m1", "role": "user", "content": "My preference is OPENAI_API_KEY=sk-secret"}],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    assert episode.origin.primary_scope is not None
    proposal = MemorySemanticProposal(
        proposal_id="secret-proposal",
        memory_type="preference",
        identity_fields={"subject": "OPENAI_API_KEY", "dimension": "secret"},
        value_fields={"preference": "OPENAI_API_KEY=sk-secret"},
        semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=(EvidenceRef.from_event(episode.events[0]),),
        confidence=0.99,
        extractor_version="fake",
    )
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    result = ProposalAdmissionGate().evaluate(validation, episode=episode, memory_scope=scope, source_role="user")
    assert result.decision == ProposalAdmissionDecision.RESTRICTED
