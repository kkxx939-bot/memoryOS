from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import (
    CandidateProposalAdapter,
    CanonicalMemoryFormationService,
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
from memoryos.memory.canonical.evidence import ConstraintPolarity
from memoryos.memory.canonical.literal_grounding import literal_value_supported
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType


def _explicit_bindings(
    identity_fields: Mapping[str, object],
    value_fields: Mapping[str, object],
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, tuple[EvidenceRef, ...]]:
    bindings = {
        **{f"identity.{key}": evidence_refs for key in identity_fields},
        **{f"value.{key}": evidence_refs for key in value_fields},
        "semantic.speech_act": evidence_refs,
        "semantic.commitment": evidence_refs,
        "semantic.temporal_scope": evidence_refs,
        "semantic.relation_to_existing": evidence_refs,
        "transition": evidence_refs,
    }
    return bind_field_evidence(identity_fields, value_fields, evidence_refs, bindings=bindings)


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
        field_evidence_refs=_explicit_bindings(proposal.identity_fields, proposal.value_fields, proposal.evidence_refs),
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
        field_evidence_refs=_explicit_bindings(identity_fields, value_fields, evidence_refs),
        confidence=0.9,
        extractor_version="fake-v1",
        model_id="fake",
    )


def _constraint_validation(
    text: str,
    canonical_value: str,
    *,
    extra_value_fields: Mapping[str, object] | None = None,
):  # noqa: ANN202
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id=f"constraint-{canonical_value}",
            archive_uri=f"memoryos://user/u1/sessions/history/constraint-{canonical_value}",
            messages=[{"id": "m1", "role": "user", "content": text}],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    assert episode.origin.primary_scope is not None
    identity_fields = {"rule_topic": "redis_usage"}
    value_fields = {"canonical_value": canonical_value, **dict(extra_value_fields or {})}
    evidence_refs = [EvidenceRef.from_event(episode.events[0], source_uri=episode.source_uris[0])]
    bindings = _explicit_bindings(identity_fields, value_fields, tuple(evidence_refs))
    for field_name, value in dict(extra_value_fields or {}).items():
        literal = str(value)
        start = text.index(literal)
        child = EvidenceRef.from_event(
            episode.events[0],
            source_uri=episode.source_uris[0],
            span_start=start,
            span_end=start + len(literal),
        )
        evidence_refs.append(child)
        bindings[f"value.{field_name}"] = (child,)
    proposal = MemorySemanticProposal(
        proposal_id=f"p-{canonical_value}",
        memory_type="project_rule",
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=tuple(evidence_refs),
        field_evidence_refs=bindings,
        confidence=0.99,
        extractor_version="test",
        metadata={"source_role": "user", "system_identity_fields": ["rule_topic"]},
    )
    return ProposalEvidenceValidator().validate(proposal, episode)


def _v3_case(
    *,
    text: str = "opaque source proposition PostgreSQL",
    memory_type: str = "project_decision",
    value_fields: Mapping[str, object] | None = None,
    utterance_mode: str = "assertion",
    attribution: str = "source_actor",
    durability: str = "durable",
    modal_force: str = "none",
    atomicity: str = "atomic",
    speech_act: str = "confirmation",
    commitment: str = "confirmed",
    temporal_scope: str = "current",
    relation: str = "unrelated",
    source_role: str = "user",
    epistemic_status: EpistemicStatus | None = None,
    related_memory_ids: tuple[str, ...] = (),
    signal_matcher: EvidenceSignalMatcher | None = None,
):  # noqa: ANN202
    archive = SessionArchive(
        user_id="u1",
        session_id=f"v3-{memory_type}-{source_role}",
        archive_uri=f"memoryos://user/u1/sessions/history/v3-{memory_type}-{source_role}",
        messages=[{"id": "m1", "role": source_role, "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    event = episode.events[0]
    assert episode.origin.primary_scope is not None
    atomic = EvidenceRef.from_event(
        event,
        source_uri=episode.source_uris[0],
        span_start=0,
        span_end=len(text),
    )
    if memory_type == "project_rule":
        identity_fields = {"rule_topic": "redis_usage"}
    elif memory_type == "preference":
        identity_fields = {"subject": "answers", "dimension": "length"}
    elif memory_type == "event":
        identity_fields = {"event_key": "deployment_completed"}
    elif memory_type == "agent_experience":
        identity_fields = {"task_pattern": "deploy", "environment_signature": "memoryos"}
    elif memory_type == "entity":
        identity_fields = {"entity_type": "project", "canonical_entity_id": "memoryos"}
    else:
        identity_fields = {"decision_topic": "storage_backend"}
    values = dict(value_fields or {"canonical_value": "PostgreSQL"})
    bindings = {
        **{f"identity.{key}": (atomic,) for key in identity_fields},
        **{f"value.{key}": (atomic,) for key in values},
        "semantic.speech_act": (atomic,),
        "semantic.commitment": (atomic,),
        "semantic.temporal_scope": (atomic,),
        "semantic.relation_to_existing": (atomic,),
        "semantic.utterance_mode": (atomic,),
        "semantic.attribution": (atomic,),
        "semantic.durability": (atomic,),
        "semantic.modal_force": (atomic,),
        "semantic.atomicity": (atomic,),
        "transition": (atomic,),
    }
    proposal = MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=f"v3-{memory_type}-{source_role}",
            memory_type=memory_type,
            identity_fields=identity_fields,
            value_fields=values,
            semantic=SemanticAssessment(
                speech_act,
                commitment,
                temporal_scope,
                relation,
                utterance_mode,
                attribution,
                durability,
                modal_force,
                atomicity,
            ),
            epistemic_status=epistemic_status
            or (EpistemicStatus.EXPLICIT if source_role in {"user", "system"} else EpistemicStatus.INFERRED),
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=related_memory_ids,
            evidence_refs=(atomic,),
            field_evidence_refs=bind_field_evidence(
                identity_fields,
                values,
                (atomic,),
                bindings=bindings,
                semantic_contract_version="v3",
            ),
            confidence=0.99,
            extractor_version="v3-test",
            semantic_contract_version="v3",
            atomic_evidence_ref=atomic,
            metadata={"source_role": source_role, "system_identity_fields": list(identity_fields)},
        )
    )
    validator = ProposalEvidenceValidator(signal_matcher=signal_matcher)
    validation = validator.validate(proposal, episode)
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
        canonical_subject=episode.origin.primary_scope,
    )
    return episode, proposal, validation, scope


def test_evidence_ref_validates_event_hash_and_supported_fields() -> None:
    episode = _episode()
    result = ProposalEvidenceValidator().validate(_proposal(episode), episode)
    assert result.valid
    assert result.unsupported_fields == ()


@pytest.mark.parametrize(
    ("value", "evidence"),
    [
        ("Redis", "redistribution is enabled"),
        ("SQL", "NoSQL is selected"),
        (1, "retry limit is 10"),
    ],
)
def test_literal_grounding_rejects_identifier_substrings(value: object, evidence: str) -> None:
    assert not literal_value_supported(value, (evidence,))


def test_literal_grounding_splits_no_space_cjk_and_latin_identifiers() -> None:
    assert literal_value_supported("Redis", ("项目使用Redis数据库",))
    assert literal_value_supported("ＲＥＤＩＳ", ("项目使用redis数据库",))


def test_literal_grounding_requires_exact_child_span_for_cjk_values() -> None:
    assert not literal_value_supported("短期缓存", ("不要使用Redis，除非只是短期缓存",))
    assert literal_value_supported("短期缓存", ("短期缓存",))


def test_v3_replacement_target_binding_rejects_multiple_declared_targets() -> None:
    _, _, validation, _ = _v3_case(
        relation="supersedes",
        related_memory_ids=("memory-1", "memory-2"),
    )

    assert validation.proposal.metadata["relation_target_binding_validated"] is False
    assert validation.proposal.metadata["replacement_evidence_validated"] is False
    assert "semantic_relation_structure_invalid" in validation.errors


def test_v3_relation_target_representations_must_identify_one_claim_and_slot() -> None:
    episode, proposal, _validation, _scope = _v3_case(
        text="Formally change the database to PostgreSQL.",
        relation="supersedes",
        related_memory_ids=("memoryos://user/u1/memories/canonical/slots/slot-a/claims/claim-a",),
    )
    inconsistent = replace(
        proposal,
        related_claim_ids=("claim-b",),
        related_slot_ids=("slot-a",),
    )

    validation = ProposalEvidenceValidator().validate(inconsistent, episode)

    assert validation.proposal.metadata["relation_target_binding_validated"] is False
    assert "semantic_relation_structure_invalid" in validation.errors


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
    bindings = _explicit_bindings(identity_fields, value_fields, (first,))
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
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
        canonical_subject=episode.origin.primary_scope,
    )
    admission = ProposalAdmissionGate().evaluate(
        result,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )
    assert admission.decision == ProposalAdmissionDecision.REJECT
    assert admission.reason.startswith("evidence_integrity_failed:")


def test_missing_candidate_source_never_falls_back_to_first_episode_event() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="missing-source",
        archive_uri="memoryos://user/u1/sessions/history/missing-source",
        messages=[{"id": "first", "role": "user", "content": "SQLite remains active."}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidate = MemoryCandidateDraft(
        memory_type=MemoryType.PROJECT_DECISION,
        title="PostgreSQL",
        content="PostgreSQL is now active.",
        fields={
            "decision_topic": "storage_backend",
            "canonical_value": "PostgreSQL",
            "project_id": "memoryos",
        },
        confidence=0.99,
        source_role="user",
        source_adapter_id="codex",
        source_session_id=archive.session_id,
        source_message_ids=["does-not-exist"],
        merge_key="storage-backend",
    )

    proposal = CandidateProposalAdapter().adapt(candidate, episode, archive)
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    formed = CanonicalMemoryFormationService(None).plan(
        proposal,
        archive=archive,
        episode=episode,
    )

    assert proposal.evidence_refs == ()
    assert all(refs == () for refs in proposal.field_evidence_refs.values())
    assert not validation.valid
    assert "missing_evidence" in validation.errors
    assert formed.decision == ProposalAdmissionDecision.PENDING
    assert len(formed.operations) == 1
    pending = formed.operations[0]
    assert pending.payload["canonical_pending_proposal"] is True
    assert (
        pending.payload["context_object"]["metadata"]["pending_reason_code"]
        == "FALLBACK_REQUIRES_REEXTRACTION"
    )
    assert (
        pending.payload["context_object"]["metadata"]["pending_reason_detail"]
        == CandidateProposalAdapter.FALLBACK_PENDING_REASON
    )


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
    proposal = replace(_proposal(episode), value_fields={"canonical_value": "PostgreSQL"})
    proposal = replace(
        proposal,
        field_evidence_refs=_explicit_bindings(
            proposal.identity_fields,
            proposal.value_fields,
            proposal.evidence_refs,
        ),
    )
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert episode.origin.primary_scope is not None
    legal_scope = MemoryScope(
        applicability=ScopeSelector((episode.origin.primary_scope,)),
        visibility=VisibilityPolicy("t1"),
        origin_refs=episode.origin.scope_refs,
        canonical_subject=episode.origin.primary_scope,
    )
    legacy = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=legal_scope,
        source_role="user",
    )
    assert legacy.decision == ProposalAdmissionDecision.PENDING
    assert legacy.reason == "semantic_contract_v3_required"
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
    proposal = _bind(
        MemorySemanticProposal(
            proposal_id="secret-proposal",
            memory_type="preference",
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=_explicit_bindings(identity_fields, value_fields, evidence_refs),
            confidence=0.99,
            extractor_version="fake",
        )
    )
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
        canonical_subject=episode.origin.primary_scope,
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
        (
            "If we decide to adopt PostgreSQL later, rerun tests.",
            EvidenceSignalKind.CONFIRMATION,
            {"hypothetical": True},
        ),
        (
            "不要把“可以考虑”理解成“确认采用”。",
            EvidenceSignalKind.CONFIRMATION,
            {"quoted": True, "metalinguistic": True},
        ),
        ("“必须使用 Redis”只是文档中的反例。", EvidenceSignalKind.CONSTRAINT, {"quoted": True, "metalinguistic": True}),
        (
            "Codex 建议正式改成 PostgreSQL，但我没有同意。",
            EvidenceSignalKind.CONFIRMATION,
            {"negated": True, "metalinguistic": True},
        ),
        (
            "The agent recommended adopting PostgreSQL, but the user did not approve.",
            EvidenceSignalKind.PROPOSAL,
            {"metalinguistic": True},
        ),
        (
            'The phrase "must use Redis" is only an example.',
            EvidenceSignalKind.CONSTRAINT,
            {"quoted": True, "metalinguistic": True},
        ),
    ],
)
def test_typed_evidence_signal_counterexamples(text: str, kind: EvidenceSignalKind, flags: dict[str, bool]) -> None:
    matches = [match for match in EvidenceSignalMatcher().match(text) if match.kind == kind]
    assert matches
    assert any(all(getattr(match, key) is value for key, value in flags.items()) for match in matches)


@pytest.mark.parametrize(
    ("text", "polarity"),
    [
        ("项目必须使用 Redis", ConstraintPolarity.REQUIRE),
        ("项目禁止使用 Redis", ConstraintPolarity.FORBID),
        ("不得使用 Redis", ConstraintPolarity.FORBID),
        ("Redis 可以使用", ConstraintPolarity.ALLOW),
        ("建议优先使用 Redis", ConstraintPolarity.PREFER),
        ("尽量避免 Redis", ConstraintPolarity.DISCOURAGE),
        ("如果是生产环境，必须使用 Redis", ConstraintPolarity.CONDITIONAL_REQUIRE),
        ("不要使用 Redis，除非只是短期缓存", ConstraintPolarity.CONDITIONAL_FORBID),
    ],
)
def test_constraint_signal_has_explicit_direction(text: str, polarity: ConstraintPolarity) -> None:
    matches = [
        match
        for match in EvidenceSignalMatcher().match(text)
        if match.polarity == polarity and not (match.negated or match.quoted or match.metalinguistic)
    ]
    assert matches


def test_constraint_contrast_uses_final_positive_requirement() -> None:
    matches = EvidenceSignalMatcher().match("不是禁止 Redis，而是要求 Redis")
    usable = [
        match.polarity
        for match in matches
        if match.kind == EvidenceSignalKind.CONSTRAINT
        and not (match.negated or match.hypothetical or match.quoted or match.metalinguistic)
    ]
    assert usable == [ConstraintPolarity.REQUIRE]


@pytest.mark.parametrize(
    ("text", "accepted_value", "rejected_value"),
    [
        ("项目必须使用 Redis", "required", "forbidden"),
        ("项目禁止使用 Redis", "forbidden", "required"),
        ("不得使用 Redis", "forbidden", "required"),
        ("Redis 可以使用", "allowed", "forbidden"),
        ("不是禁止 Redis，而是要求 Redis", "required", "forbidden"),
    ],
)
def test_constraint_value_requires_matching_evidence_polarity(
    text: str,
    accepted_value: str,
    rejected_value: str,
) -> None:
    accepted = _constraint_validation(text, accepted_value)
    rejected = _constraint_validation(text, rejected_value)

    assert accepted.valid, accepted.errors
    assert not rejected.valid
    assert "value.canonical_value" in rejected.unsupported_fields


def test_constraint_polarity_must_bind_to_the_same_rule_subject() -> None:
    text = "不要使用 Redis，但 PostgreSQL 必须使用"

    redis_required = _constraint_validation(text, "required")
    redis_forbidden = _constraint_validation(text, "forbidden")

    assert not redis_required.valid
    assert "value.canonical_value" in redis_required.unsupported_fields
    assert redis_forbidden.valid, redis_forbidden.errors


def test_indirect_attribution_cannot_support_authoritative_constraint_but_project_requirement_can() -> None:
    for text in (
        "经理说项目必须使用 Redis",
        "我听说项目禁止使用 Redis",
        "据说项目必须使用 Redis",
    ):
        value = "forbidden" if "禁止" in text else "required"
        attributed = _constraint_validation(text, value)
        assert not attributed.valid

    direct = _constraint_validation("项目要求必须使用 Redis", "required")
    assert direct.valid, direct.errors


def test_conditional_forbid_requires_preserved_exception() -> None:
    text = "不要使用 Redis，除非只是短期缓存"
    flattened = _constraint_validation(text, "forbidden")
    preserved = _constraint_validation(
        text,
        "forbidden",
        extra_value_fields={"exception": "短期缓存"},
    )

    assert not flattened.valid
    assert "value.canonical_value" in flattened.unsupported_fields
    assert preserved.valid, preserved.errors


@pytest.mark.parametrize(
    ("text", "exception"),
    [
        ("除非只是缓存，否则不得使用 Redis", "只是缓存"),
        (
            "Unless Redis is used only as a cache, it must not be enabled.",
            "Redis is used only as a cache",
        ),
    ],
)
def test_preposed_exception_is_conditional_and_must_be_preserved(text: str, exception: str) -> None:
    signals = [
        match
        for match in EvidenceSignalMatcher().match(text)
        if match.polarity == ConstraintPolarity.CONDITIONAL_FORBID
    ]
    flattened = _constraint_validation(text, "forbidden")
    preserved = _constraint_validation(text, "forbidden", extra_value_fields={"exception": exception})

    assert signals
    assert not flattened.valid
    assert "value.canonical_value" in flattened.unsupported_fields
    assert preserved.valid, preserved.errors


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
    proposal = _bind(
        MemorySemanticProposal(
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
        )
    )
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
    proposal = _bind(
        MemorySemanticProposal(
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
        )
    )
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
    proposal = _bind(
        MemorySemanticProposal(
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
        )
    )
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert not validation.valid
    assert "temporal_scope_unsupported" in validation.errors


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf"), 1.1, -0.1, "not-a-number"])
def test_semantic_proposal_rejects_non_finite_or_out_of_range_confidence(confidence: object) -> None:
    with pytest.raises(ValueError, match="confidence must be a finite number"):
        replace(_proposal(), confidence=confidence)


class _NoLexicalSemanticProof(EvidenceSignalMatcher):
    def match(self, text: str):  # noqa: ANN201
        raise AssertionError(f"V3 semantic validation must not inspect lexical signals: {text}")


def test_v3_semantics_are_source_grounded_without_lexical_positive_proof() -> None:
    episode, _proposal_v3, validation, scope = _v3_case(
        text="DB := PostgreSQL",
        signal_matcher=_NoLexicalSemanticProof(),
    )

    assert validation.valid, validation.errors
    assert validation.proposal.metadata["atomic_evidence_validated"] is True
    assert validation.proposal.metadata["semantic_contract_validated"] is True
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )
    assert admission.decision == ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE


def test_v3_atomic_span_is_required_and_all_semantic_bindings_use_it() -> None:
    episode, proposal, _validation, scope = _v3_case()
    missing = replace(proposal, atomic_evidence_ref=None)
    missing_validation = ProposalEvidenceValidator().validate(missing, episode)

    assert not missing_validation.valid
    assert "missing_atomic_evidence_ref" in missing_validation.errors
    assert (
        ProposalAdmissionGate()
        .evaluate(missing_validation, episode=episode, memory_scope=scope, source_role="user")
        .decision
        == ProposalAdmissionDecision.PENDING
    )

    wrong_bindings = dict(proposal.field_evidence_refs)
    wrong_bindings["semantic.attribution"] = ()
    mismatched = replace(proposal, field_evidence_refs=wrong_bindings)
    mismatch_validation = ProposalEvidenceValidator().validate(mismatched, episode)
    assert not mismatch_validation.valid
    assert "atomic_evidence_binding_mismatch:semantic.attribution" in mismatch_validation.errors


def test_v3_model_field_can_bind_an_exact_child_span_inside_atomic_span() -> None:
    episode, proposal, _validation, _scope = _v3_case(text="backend PostgreSQL selected")
    event = episode.events[0]
    start = event.text().index("PostgreSQL")
    child = EvidenceRef.from_event(
        event,
        source_uri=episode.source_uris[0],
        span_start=start,
        span_end=start + len("PostgreSQL"),
    )
    bindings = dict(proposal.field_evidence_refs)
    bindings["value.canonical_value"] = (child,)
    grounded = replace(
        proposal,
        evidence_refs=(*proposal.evidence_refs, child),
        field_evidence_refs=bindings,
    )

    validation = ProposalEvidenceValidator().validate(grounded, episode)

    assert validation.valid, validation.errors
    assert validation.unsupported_fields == ()


def test_v3_tampered_atomic_evidence_is_rejected_not_pending() -> None:
    episode, proposal, _validation, scope = _v3_case()
    assert proposal.atomic_evidence_ref is not None
    bad = replace(proposal.atomic_evidence_ref, content_hash="forged")
    bindings = {field_name: (bad,) for field_name in proposal.field_evidence_refs}
    forged = replace(
        proposal,
        evidence_refs=(bad,),
        atomic_evidence_ref=bad,
        field_evidence_refs=bindings,
    )
    validation = ProposalEvidenceValidator().validate(forged, episode)

    assert not validation.valid
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )
    assert admission.decision == ProposalAdmissionDecision.REJECT
    assert admission.reason.startswith("evidence_integrity_failed:")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"utterance_mode": "question", "speech_act": "evaluation_request", "commitment": "exploratory"}, ProposalAdmissionDecision.ARCHIVE_ONLY),
        ({"utterance_mode": "hypothetical", "speech_act": "proposal", "commitment": "exploratory"}, ProposalAdmissionDecision.ARCHIVE_ONLY),
        ({"attribution": "quoted"}, ProposalAdmissionDecision.ARCHIVE_ONLY),
        ({"durability": "transient"}, ProposalAdmissionDecision.ARCHIVE_ONLY),
        ({"utterance_mode": "mixed"}, ProposalAdmissionDecision.PENDING),
        ({"attribution": "third_party", "epistemic_status": EpistemicStatus.INFERRED}, ProposalAdmissionDecision.PENDING),
        ({"atomicity": "compound"}, ProposalAdmissionDecision.PENDING),
    ],
)
def test_v3_admission_matrix_fails_closed(
    overrides: dict[str, object],
    expected: ProposalAdmissionDecision,
) -> None:
    episode, _proposal_v3, validation, scope = _v3_case(**overrides)  # type: ignore[arg-type]
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )

    assert admission.decision == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"temporal_scope": "past"},
        {"temporal_scope": "future"},
        {"temporal_scope": "unspecified"},
        {"commitment": "intended"},
        {"commitment": "weak"},
        {"epistemic_status": EpistemicStatus.INFERRED},
        {"speech_act": "proposal", "commitment": "exploratory"},
        {"relation": "alternative"},
        {"relation": "contradicts"},
    ],
)
def test_v3_nonfinal_authoritative_semantics_are_pending(overrides: dict[str, object]) -> None:
    episode, _proposal_v3, validation, scope = _v3_case(**overrides)  # type: ignore[arg-type]
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )

    assert admission.decision == ProposalAdmissionDecision.PENDING


@pytest.mark.parametrize(
    ("text", "modal_force", "value_fields", "expected"),
    [
        (
            "Redis is required",
            "require",
            {"canonical_value": "required"},
            ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
        ),
        (
            "Redis is forbidden",
            "require",
            {"canonical_value": "forbidden"},
            ProposalAdmissionDecision.PENDING,
        ),
        (
            "Do not use Redis unless it is a short-term cache",
            "conditional_forbid",
            {"canonical_value": "forbidden"},
            ProposalAdmissionDecision.PENDING,
        ),
        (
            "Do not use Redis unless it is a short-term cache",
            "conditional_forbid",
            {"canonical_value": "forbidden", "exception": "short-term cache"},
            ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
        ),
        (
            "Do not use Redis unless it is a short-term cache",
            "forbid",
            {"canonical_value": "forbidden", "exception": "short-term cache"},
            ProposalAdmissionDecision.PENDING,
        ),
    ],
)
def test_v3_project_rule_uses_structural_modal_consistency(
    text: str,
    modal_force: str,
    value_fields: Mapping[str, object],
    expected: ProposalAdmissionDecision,
) -> None:
    episode, _proposal_v3, validation, scope = _v3_case(
        text=text,
        memory_type="project_rule",
        value_fields=value_fields,
        utterance_mode="directive",
        modal_force=modal_force,
    )
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )

    assert admission.decision == expected


@pytest.mark.parametrize(
    ("modal_force", "expected"),
    [
        ("prefer", ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE),
        ("discourage", ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE),
        ("none", ProposalAdmissionDecision.PENDING),
        ("require", ProposalAdmissionDecision.PENDING),
        ("forbid", ProposalAdmissionDecision.PENDING),
    ],
)
def test_v3_preference_modal_force_is_non_authoritative_direction(
    modal_force: str,
    expected: ProposalAdmissionDecision,
) -> None:
    episode, _proposal_v3, validation, scope = _v3_case(
        memory_type="preference",
        modal_force=modal_force,
    )
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )

    assert admission.decision == expected


def test_v3_authority_comes_from_atomic_transition_actor() -> None:
    user_episode, _proposal_user, user_validation, user_scope = _v3_case(source_role="user")
    assert (
        ProposalAdmissionGate()
        .evaluate(user_validation, episode=user_episode, memory_scope=user_scope, source_role="assistant")
        .decision
        == ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE
    )

    assistant_episode, _proposal_agent, assistant_validation, assistant_scope = _v3_case(
        source_role="assistant",
        epistemic_status=EpistemicStatus.INFERRED,
    )
    assert assistant_validation.valid, assistant_validation.errors
    assert (
        ProposalAdmissionGate()
        .evaluate(
            assistant_validation,
            episode=assistant_episode,
            memory_scope=assistant_scope,
            source_role="user",
        )
        .decision
        == ProposalAdmissionDecision.PENDING
    )


@pytest.mark.parametrize(
    (
        "memory_type",
        "source_role",
        "temporal_scope",
        "epistemic_status",
        "expected",
        "reason",
    ),
    [
        (
            "event",
            "user",
            "past",
            EpistemicStatus.OBSERVED,
            ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
            "validated",
        ),
        (
            "event",
            "assistant",
            "past",
            EpistemicStatus.INFERRED,
            ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
            "validated",
        ),
        (
            "agent_experience",
            "assistant",
            "past",
            EpistemicStatus.INFERRED,
            ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
            "validated",
        ),
        (
            "agent_experience",
            "user",
            "past",
            EpistemicStatus.INFERRED,
            ProposalAdmissionDecision.REJECT,
            "user_source_not_allowed",
        ),
        (
            "event",
            "user",
            "future",
            EpistemicStatus.OBSERVED,
            ProposalAdmissionDecision.PENDING,
            "semantic_v3_non_authoritative_temporality_pending",
        ),
        (
            "entity",
            "user",
            "past",
            EpistemicStatus.OBSERVED,
            ProposalAdmissionDecision.PENDING,
            "semantic_v3_non_authoritative_temporality_pending",
        ),
    ],
)
def test_v3_memory_type_eligibility_is_schema_and_temporality_aware(
    memory_type: str,
    source_role: str,
    temporal_scope: str,
    epistemic_status: EpistemicStatus,
    expected: ProposalAdmissionDecision,
    reason: str,
) -> None:
    episode, _proposal_v3, validation, scope = _v3_case(
        text="Redis is required" if memory_type == "project_rule" else "opaque source proposition PostgreSQL",
        memory_type=memory_type,
        source_role=source_role,
        temporal_scope=temporal_scope,
        speech_act="observation",
        commitment="weak",
        epistemic_status=epistemic_status,
    )
    assert validation.valid, validation.errors

    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role=source_role,
    )

    assert admission.decision == expected
    assert admission.reason == reason


@pytest.mark.parametrize(
    "memory_type",
    [
        "profile",
        "preference",
        "entity",
        "event",
        "project_rule",
        "project_decision",
        "agent_experience",
    ],
)
def test_v3_hypothesized_proposal_is_pending_for_every_memory_type(memory_type: str) -> None:
    source_role = "assistant" if memory_type == "agent_experience" else "user"
    modal_force = "prefer" if memory_type == "preference" else "require" if memory_type == "project_rule" else "none"
    value_fields = {"canonical_value": "required"} if memory_type == "project_rule" else None
    episode, _proposal_v3, validation, scope = _v3_case(
        text="Redis is required" if memory_type == "project_rule" else "opaque source proposition PostgreSQL",
        memory_type=memory_type,
        source_role=source_role,
        epistemic_status=EpistemicStatus.HYPOTHESIZED,
        modal_force=modal_force,
        value_fields=value_fields,
    )
    assert validation.valid, validation.errors

    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role=source_role,
    )

    assert admission.decision == ProposalAdmissionDecision.PENDING
    assert admission.reason == "hypothesis_requires_confirmation"


def test_v3_project_rule_retraction_does_not_redeclare_modal_force() -> None:
    episode, _proposal_v3, validation, scope = _v3_case(
        memory_type="project_rule",
        speech_act="retraction",
        relation="corrects",
        modal_force="none",
        related_memory_ids=("claim-redis-rule",),
    )

    assert validation.valid, validation.errors
    admission = ProposalAdmissionGate().evaluate(
        validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )
    assert admission.decision == ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE

    _episode_non_retraction, _proposal_non_retraction, invalid, _scope_non_retraction = _v3_case(
        memory_type="project_rule",
        modal_force="none",
    )
    assert not invalid.valid
    assert "project_rule_modal_force_missing" in invalid.errors
