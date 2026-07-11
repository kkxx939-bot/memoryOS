from __future__ import annotations

from dataclasses import replace

import pytest

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import (
    Commitment,
    EpistemicStatus,
    EvidenceRef,
    EvidenceSignalKind,
    EvidenceSignalMatcher,
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
    bind_field_evidence,
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


def _bind(proposal: MemorySemanticProposal) -> MemorySemanticProposal:
    return replace(
        proposal,
        field_evidence_refs=bind_field_evidence(
            proposal.identity_fields,
            proposal.value_fields,
            proposal.evidence_refs,
        ),
    )


def _proposal(episode=None):  # noqa: ANN001, ANN202
    episode = episode or _episode()
    event = episode.events[0]
    assert episode.origin.primary_scope is not None
    identity_fields = {"decision_topic": "PostgreSQL"}
    value_fields = {"value": "future option"}
    evidence_refs = (EvidenceRef.from_event(event, source_uri=episode.source_uris[0]),)
    return MemorySemanticProposal(
        proposal_id="p1",
        memory_type="project_decision",
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=SemanticAssessment("future_option", "exploratory_alternative", "future", "alternative"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=evidence_refs,
        field_evidence_refs=bind_field_evidence(identity_fields, value_fields, evidence_refs),
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


def test_field_evidence_cannot_borrow_support_from_an_unbound_event() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="field-binding",
            archive_uri="memoryos://user/u1/sessions/history/field-binding",
            messages=[
                {"id": "m1", "role": "user", "content": "We are discussing storage."},
                {"id": "m2", "role": "user", "content": "SQLite remains active."},
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    first = EvidenceRef.from_event(episode.events[0], source_uri=episode.source_uris[0])
    second = EvidenceRef.from_event(episode.events[1], source_uri=episode.source_uris[0])
    identity_fields = {"decision_topic": "storage"}
    value_fields = {"canonical_value": "SQLite"}
    bindings = bind_field_evidence(identity_fields, value_fields, (first,))
    proposal = MemorySemanticProposal(
        proposal_id="p-bound-fields",
        memory_type="project_decision",
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(),
        related_memory_ids=(),
        evidence_refs=(first, second),
        field_evidence_refs=bindings,
        confidence=0.9,
        extractor_version="test",
    )

    validation = ProposalEvidenceValidator().validate(proposal, episode)

    assert not validation.valid
    assert "value.canonical_value" in validation.unsupported_fields


def test_missing_field_level_evidence_is_rejected() -> None:
    validation = ProposalEvidenceValidator().validate(
        replace(_proposal(), field_evidence_refs={}),
        _episode(),
    )

    assert not validation.valid
    assert any(error.startswith("missing_field_evidence:") for error in validation.errors)


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
    identity_fields = {"subject": "OPENAI_API_KEY", "dimension": "secret"}
    value_fields = {"preference": "OPENAI_API_KEY=sk-secret"}
    evidence_refs = (EvidenceRef.from_event(episode.events[0], source_uri=episode.source_uris[0]),)
    proposal = _bind(MemorySemanticProposal(
        proposal_id="secret-proposal",
        memory_type="preference",
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=evidence_refs,
        field_evidence_refs=bind_field_evidence(identity_fields, value_fields, evidence_refs),
        confidence=0.99,
        extractor_version="fake",
    ))
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    result = ProposalAdmissionGate().evaluate(validation, episode=episode, memory_scope=scope, source_role="user")
    assert result.decision == ProposalAdmissionDecision.RESTRICTED


@pytest.mark.parametrize(
    ("text", "kind", "flags"),
    [
        ("我喜欢 PostgreSQL。", EvidenceSignalKind.PREFERENCE, {}),
        ("我不喜欢 PostgreSQL。", EvidenceSignalKind.PREFERENCE, {"negated": False}),
        ("I do not like PostgreSQL.", EvidenceSignalKind.PREFERENCE, {"negated": True}),
        ("这还没有确认。", EvidenceSignalKind.CONFIRMATION, {"negated": True}),
        ("未确认采用 PostgreSQL。", EvidenceSignalKind.CONFIRMATION, {"negated": True}),
        ("PostgreSQL is unconfirmed.", EvidenceSignalKind.NEGATION, {"negated": True}),
        ("I have not confirmed PostgreSQL.", EvidenceSignalKind.CONFIRMATION, {"negated": True}),
        ("如果以后决定采用 PostgreSQL，需要重新压测。", EvidenceSignalKind.CONFIRMATION, {"hypothetical": True}),
        ("If we decide to adopt PostgreSQL later, rerun tests.", EvidenceSignalKind.CONFIRMATION, {"hypothetical": True}),
        ("不要把“可以考虑”理解成“确认采用”。", EvidenceSignalKind.CONFIRMATION, {"quoted": True, "metalinguistic": True}),
        ("“必须使用 Redis”只是文档中的反例。", EvidenceSignalKind.CONSTRAINT, {"quoted": True, "metalinguistic": True}),
        ("Codex 建议正式改成 PostgreSQL，但我没有同意。", EvidenceSignalKind.CONFIRMATION, {"negated": True, "metalinguistic": True}),
        ("The agent recommended adopting PostgreSQL, but the user did not approve.", EvidenceSignalKind.PROPOSAL, {"metalinguistic": True}),
        ('The phrase "must use Redis" is only an example.', EvidenceSignalKind.CONSTRAINT, {"quoted": True, "metalinguistic": True}),
    ],
)
def test_typed_evidence_signal_counterexamples(text: str, kind: EvidenceSignalKind, flags: dict[str, bool]) -> None:
    matches = [match for match in EvidenceSignalMatcher().match(text) if match.kind == kind]
    assert matches
    assert any(all(getattr(match, key) is value for key, value in flags.items()) for match in matches)


def test_preference_signal_cannot_support_project_decision_confirmation() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="preference-only",
            archive_uri="memoryos://user/u1/sessions/history/preference-only",
            messages=[{"id": "m1", "role": "user", "content": "我喜欢 PostgreSQL。"}],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    assert episode.origin.primary_scope is not None
    proposal = _bind(MemorySemanticProposal(
        proposal_id="p-like-as-decision",
        memory_type="project_decision",
        identity_fields={"decision_topic": "PostgreSQL"},
        value_fields={"canonical_value": "PostgreSQL"},
        semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=(EvidenceRef.from_event(episode.events[0]),),
        confidence=0.99,
        extractor_version="test",
    ))
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert not validation.valid
    assert "semantic_confirmation_unsupported" in validation.errors


def test_attributed_agent_recommendation_cannot_support_user_confirmation() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="agent-recommendation",
            archive_uri="memoryos://user/u1/sessions/history/agent-recommendation",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": "Codex 建议正式改成 PostgreSQL，但我没有同意。",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    assert episode.origin.primary_scope is not None
    proposal = _bind(MemorySemanticProposal(
        proposal_id="p-attributed",
        memory_type="project_decision",
        identity_fields={"decision_topic": "PostgreSQL"},
        value_fields={"canonical_value": "PostgreSQL"},
        semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=(EvidenceRef.from_event(episode.events[0]),),
        confidence=0.99,
        extractor_version="test",
    ))
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert not validation.valid
    assert "semantic_confirmation_unsupported" in validation.errors


def test_future_temporal_scope_requires_evidence() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="invented-time",
            archive_uri="memoryos://user/u1/sessions/history/invented-time",
            messages=[{"id": "m1", "role": "user", "content": "PostgreSQL is a database option."}],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    assert episode.origin.primary_scope is not None
    proposal = _bind(MemorySemanticProposal(
        proposal_id="p-invented-time",
        memory_type="project_decision",
        identity_fields={"decision_topic": "PostgreSQL"},
        value_fields={"canonical_value": "PostgreSQL"},
        semantic=SemanticAssessment("proposal", "exploratory", "future", "alternative"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=(EvidenceRef.from_event(episode.events[0], source_uri=episode.source_uris[0]),),
        confidence=0.99,
        extractor_version="test",
    ))
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert not validation.valid
    assert "temporal_scope_unsupported" in validation.errors


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf"), 1.1, -0.1, "not-a-number"])
def test_semantic_proposal_rejects_non_finite_or_out_of_range_confidence(confidence: object) -> None:
    with pytest.raises(ValueError, match="confidence must be a finite number"):
        replace(_proposal(), confidence=confidence)
