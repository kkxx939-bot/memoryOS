from __future__ import annotations

from dataclasses import replace

import pytest

from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical.admission import ProposalAdmissionDecision, ProposalAdmissionGate
from memoryos.memory.canonical.episode import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.evidence import EvidenceRef, ProposalValidationResult
from memoryos.memory.canonical.formation import CandidateProposalAdapter, CanonicalMemoryFormationService
from memoryos.memory.canonical.identity import StableMemoryIdentityResolver
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticAssessment,
)
from memoryos.memory.canonical.scope import AuthorityPolicy, MemoryScope, ScopeSelector, VisibilityPolicy
from memoryos.memory.canonical.semantic import MemorySemanticNormalizer
from memoryos.memory.extraction import RuleFallbackExtractor
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeRegistry


def _archive(text: str, *, session_id: str = "s1") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        messages=[{"id": "m1", "role": "user", "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )


def _candidate(text: str, memory_type: MemoryType) -> MemoryCandidateDraft:
    candidates = RuleFallbackExtractor().extract_drafts(_archive(text), MemoryTypeRegistry().list())
    return next(item for item in candidates if item.memory_type == memory_type)


def test_rule_fallback_preserves_constraint_polarity_and_exception() -> None:
    required = _candidate("项目必须使用 Redis", MemoryType.PROJECT_RULE)
    forbidden = _candidate("项目禁止使用 Redis", MemoryType.PROJECT_RULE)
    short_forbid = _candidate("不得使用 Redis", MemoryType.PROJECT_RULE)
    allowed = _candidate("Redis 可以使用", MemoryType.PROJECT_RULE)
    conditional = _candidate("不要使用 Redis，除非只是短期缓存", MemoryType.PROJECT_RULE)
    semicolon_conditional = _candidate("不要使用 Redis；除非只是短期缓存", MemoryType.PROJECT_RULE)
    conditional_required = _candidate("如果是生产环境，必须使用 Redis", MemoryType.PROJECT_RULE)
    preposed_exception = _candidate("除非只是缓存，否则不得使用 Redis", MemoryType.PROJECT_RULE)
    english_preposed = _candidate(
        "Unless Redis is used only as a cache, it must not be enabled.",
        MemoryType.PROJECT_RULE,
    )
    corrected = _candidate("不是禁止 Redis，而是要求 Redis", MemoryType.PROJECT_RULE)

    assert required.fields["canonical_value"] == "required"
    assert forbidden.fields["canonical_value"] == "forbidden"
    assert short_forbid.fields["canonical_value"] == "forbidden"
    assert allowed.fields["canonical_value"] == "allowed"
    assert conditional.fields["canonical_value"] == "forbidden"
    assert conditional.fields["exception"] == "只是短期缓存"
    assert semicolon_conditional.fields["canonical_value"] == "forbidden"
    assert semicolon_conditional.fields["exception"] == "只是短期缓存"
    assert conditional_required.fields["canonical_value"] == "required"
    assert conditional_required.fields["condition"] == "是生产环境"
    assert preposed_exception.fields["polarity"] == "CONDITIONAL_FORBID"
    assert preposed_exception.fields["exception"] == "只是缓存"
    assert english_preposed.fields["polarity"] == "CONDITIONAL_FORBID"
    assert english_preposed.fields["exception"] == "Redis is used only as a cache"
    assert corrected.fields["canonical_value"] == "required"


@pytest.mark.parametrize(
    ("text", "memory_type"),
    [
        ("项目必须使用 Redis", MemoryType.PROJECT_RULE),
        ("主存储正式改为 PostgreSQL", MemoryType.PROJECT_DECISION),
        ("我是软件测试工程师", MemoryType.PROFILE),
    ],
)
def test_fallback_discovery_is_always_durable_pending_not_canonical_authority(
    text: str,
    memory_type: MemoryType,
) -> None:
    archive = _archive(text, session_id=f"pending-{memory_type.value}")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidate = _candidate(text, memory_type)
    proposal = CandidateProposalAdapter().adapt(candidate, episode, archive)

    assert RuleFallbackExtractor.pending_only is True
    assert proposal.metadata["fallback_pending_only"] is True
    formed = CanonicalMemoryFormationService(None).plan(
        proposal,
        archive=archive,
        episode=episode,
    )

    assert formed.decision == ProposalAdmissionDecision.PENDING
    assert formed.reason == CandidateProposalAdapter.FALLBACK_PENDING_REASON
    assert len(formed.operations) == 1
    assert formed.operations[0].payload["canonical_pending_proposal"] is True
    assert formed.operations[0].payload.get("canonical_memory") is not True


def test_default_planner_archives_without_a_parallel_rule_extractor() -> None:
    planner = MemoryCommitPlanner()
    planned = planner.plan(_archive("主存储正式改为 PostgreSQL", session_id="planner-no-extractor"))

    assert planner.extractor is None
    assert planned.operations == ()
    assert planned.context.proposal_inputs == ()


def test_rule_fallback_selects_final_choice_and_is_conservative_for_ambiguity() -> None:
    final = _candidate("之前考虑 MySQL，最终决定使用 PostgreSQL", MemoryType.PROJECT_DECISION)
    undecided = _candidate("PostgreSQL 或 MySQL 都可以，暂时没决定", MemoryType.PROJECT_DECISION)
    attributed = RuleFallbackExtractor().extract_drafts(
        _archive("他说必须使用 MySQL，但我不同意"),
        MemoryTypeRegistry().list(),
    )

    assert final.fields["canonical_value"] == "postgresql"
    assert "canonical_value" not in undecided.fields
    assert undecided.fields["_semantic_commitment"] == "unknown"
    assert not any(
        item.memory_type == MemoryType.PROJECT_DECISION and item.fields.get("canonical_value") == "mysql"
        for item in attributed
    )


def test_rule_fallback_rejects_competing_constraint_subjects_and_pends_uncertain_modality() -> None:
    extractor = RuleFallbackExtractor()
    mixed = extractor.extract_drafts(
        _archive("不要使用 Redis，但 PostgreSQL 必须使用"),
        MemoryTypeRegistry().list(),
    )
    uncertain_archive = _archive("项目可能需要使用 Redis", session_id="uncertain-rule")
    uncertain = next(
        item
        for item in extractor.extract_drafts(uncertain_archive, MemoryTypeRegistry().list())
        if item.memory_type == MemoryType.PROJECT_RULE
    )
    episode = SessionArchiveEpisodeAdapter().adapt(uncertain_archive)
    proposal = CandidateProposalAdapter().adapt(uncertain, episode, uncertain_archive)
    formed = CanonicalMemoryFormationService(None).plan(
        proposal,
        archive=uncertain_archive,
        episode=episode,
    )

    assert not any(item.memory_type == MemoryType.PROJECT_RULE for item in mixed)
    assert uncertain.fields["_semantic_commitment"] == "unknown"
    assert uncertain.fields["_semantic_temporal_scope"] == "unknown"
    assert formed.decision == ProposalAdmissionDecision.PENDING
    assert len(formed.operations) == 1
    assert formed.operations[0].payload["canonical_pending_proposal"] is True


def test_rule_fallback_does_not_create_authoritative_rule_from_tool_message() -> None:
    archive = replace(
        _archive("项目必须使用 Redis"),
        messages=[{"id": "m1", "role": "tool", "content": "项目必须使用 Redis"}],
    )

    candidates = RuleFallbackExtractor().extract_drafts(archive, MemoryTypeRegistry().list())

    assert not any(item.memory_type == MemoryType.PROJECT_RULE for item in candidates)


def test_rule_fallback_blocks_indirect_attribution_and_single_constraint_multi_subjects() -> None:
    extractor = RuleFallbackExtractor()
    schemas = MemoryTypeRegistry().list()

    for text in (
        "经理说项目必须使用 Redis",
        "我听说项目禁止使用 Redis",
        "据说项目必须使用 Redis",
        "Redis 或 PostgreSQL 必须使用",
        "Redis and PostgreSQL must be used",
    ):
        candidates = extractor.extract_drafts(_archive(text), schemas)
        assert not any(item.memory_type == MemoryType.PROJECT_RULE for item in candidates)

    direct = _candidate("项目要求必须使用 Redis", MemoryType.PROJECT_RULE)
    assert direct.fields["canonical_value"] == "required"


@pytest.mark.parametrize(
    "text",
    [
        "我们讨论过改成 MySQL，不过最后决定继续使用 PostgreSQL",
        "不是把 PostgreSQL 改为 MySQL，而是继续使用 PostgreSQL",
    ],
)
def test_fallback_discussion_or_negated_switch_is_current_confirmation_not_replacement(text: str) -> None:
    candidate = _candidate(text, MemoryType.PROJECT_DECISION)

    assert candidate.fields["canonical_value"] == "postgresql"
    assert candidate.fields["_semantic_relation"] == "unrelated"
    assert candidate.fields["_semantic_commitment"] == "confirmed"
    assert "_replacement_explicit" not in candidate.fields


def test_profile_candidate_without_explicit_attribute_key_becomes_durable_identity_pending() -> None:
    archive = _archive("I confirm a durable personal detail.", session_id="profile-missing-key")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidate = MemoryCandidateDraft(
        memory_type=MemoryType.PROFILE,
        title="Personal detail",
        content="I confirm a durable personal detail.",
        fields={"summary": "I confirm a durable personal detail."},
        confidence=0.99,
        source_role="user",
        source_adapter_id="codex",
        source_session_id=archive.session_id,
        source_message_ids=["m1"],
    )

    proposal = CandidateProposalAdapter().adapt(candidate, episode, archive)
    formed = CanonicalMemoryFormationService(None).plan(
        proposal,
        archive=archive,
        episode=episode,
    )

    assert dict(proposal.identity_fields) == {}
    assert formed.decision == ProposalAdmissionDecision.PENDING
    assert formed.reason == CandidateProposalAdapter.FALLBACK_PENDING_REASON
    assert len(formed.operations) == 1
    assert formed.operations[0].payload["canonical_pending_proposal"] is True
    pending_metadata = formed.operations[0].payload["context_object"]["metadata"]
    assert pending_metadata["identity_fields"] == {}
    assert "profile_summary" not in str(pending_metadata)


def test_fallback_does_not_promote_availability_or_attributed_speech() -> None:
    archive = _archive("确认 MySQL 可用", session_id="mysql-availability")
    candidate = _candidate("确认 MySQL 可用", MemoryType.PROJECT_DECISION)
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(candidate, episode, archive)

    formed = CanonicalMemoryFormationService(None).plan(
        proposal,
        archive=archive,
        episode=episode,
    )

    assert formed.decision == ProposalAdmissionDecision.PENDING
    assert len(formed.operations) == 1
    assert formed.operations[0].payload["canonical_pending_proposal"] is True
    assert candidate.fields["_semantic_temporal_scope"] == "unknown"

    extractor = RuleFallbackExtractor()
    for text in ("有人建议必须使用 Redis", "他说必须使用 Redis"):
        attributed = extractor.extract_drafts(_archive(text), MemoryTypeRegistry().list())
        assert not any(
            item.memory_type in {MemoryType.PROJECT_RULE, MemoryType.PROJECT_DECISION}
            for item in attributed
        )


def test_fallback_keeps_the_users_final_choice_after_attributed_advice() -> None:
    final = _candidate(
        "他说必须使用 MySQL，但我最终决定使用 PostgreSQL",
        MemoryType.PROJECT_DECISION,
    )

    assert final.fields["canonical_value"] == "postgresql"
    assert final.fields["_semantic_commitment"] == "confirmed"
    assert final.fields["_semantic_temporal_scope"] == "current"


def test_profile_fallback_uses_stable_distinct_attributes_without_profile_summary() -> None:
    occupation = _candidate("我是软件测试工程师", MemoryType.PROFILE)
    location = _candidate("我在上海工作", MemoryType.PROFILE)
    role = _candidate("我是 MemoryOS 的负责人", MemoryType.PROFILE)
    language = _candidate("我的常用语言是中文", MemoryType.PROFILE)
    active_project = _candidate("我当前的项目是 MemoryOS", MemoryType.PROFILE)
    unknown = RuleFallbackExtractor().extract_drafts(
        _archive("我是一个愿意持续学习的人"),
        MemoryTypeRegistry().list(),
    )

    assert occupation.fields["attribute_key"] == "occupation"
    assert location.fields["attribute_key"] == "work_location"
    assert role.fields["attribute_key"] == "project_role"
    assert language.fields == {
        "attribute_key": "language",
        "canonical_value": "中文",
        "summary": "我的常用语言是中文",
    }
    assert active_project.fields == {
        "attribute_key": "active_project",
        "canonical_value": "MemoryOS",
        "summary": "我当前的项目是 MemoryOS",
    }
    assert len(
        {
            occupation.merge_key,
            location.merge_key,
            role.merge_key,
            language.merge_key,
            active_project.merge_key,
        }
    ) == 5
    assert all(item.fields.get("attribute_key") != "profile_summary" for item in unknown)


def _identity_scope() -> MemoryScope:
    episode = SessionArchiveEpisodeAdapter().adapt(_archive("数据库继续使用 PostgreSQL"))
    assert episode.origin.primary_scope is not None
    return MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
        canonical_subject=episode.origin.primary_scope,
        authority=AuthorityPolicy(principal_ids=("u1",)),
    )


def _identity_proposal(value_fields: dict, proposal_id: str = "p1") -> MemorySemanticProposal:
    return MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields={"decision_topic": "primary_storage_backend"},
            value_fields=value_fields,
            semantic=SemanticAssessment("confirmation", "confirmed", "current", "unrelated"),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(),
            related_memory_ids=(),
            evidence_refs=(),
            confidence=0.9,
            extractor_version="test",
        )
    )


def test_claim_identity_ignores_wording_and_rationale_but_keeps_applicability() -> None:
    resolver = StableMemoryIdentityResolver()
    scope = _identity_scope()
    first = resolver.resolve(
        _identity_proposal(
            {"canonical_value": "Postgres", "decision": "数据库继续使用 PostgreSQL", "rationale": "stable"}
        ),
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    paraphrase = resolver.resolve(
        _identity_proposal(
            {"canonical_value": "postgresql", "decision": "数据库仍然保持 PostgreSQL", "rationale": "cheaper"},
            "p2",
        ),
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    production = resolver.resolve(
        _identity_proposal({"canonical_value": "PostgreSQL", "environment": "production"}, "p3"),
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    testing = resolver.resolve(
        _identity_proposal({"canonical_value": "PostgreSQL", "environment": "testing"}, "p4"),
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )

    assert first.claim_id == paraphrase.claim_id
    assert first.canonical_value == paraphrase.canonical_value == "postgresql"
    assert production.claim_id != testing.claim_id


def test_claim_identity_rejects_applicability_without_canonical_core_value() -> None:
    schema = MemoryTypeRegistry().get(MemoryType.PROJECT_DECISION)
    qualifier_only = _identity_proposal(
        {
            "decision": "生产环境数据库使用 PostgreSQL",
            "environment": "production",
        },
        "qualifier-only",
    )

    assert schema.claim_identity_keys(dict(qualifier_only.value_fields)) == ()
    with pytest.raises(ValueError, match="claim requires canonical semantic value"):
        StableMemoryIdentityResolver().resolve(
            qualifier_only,
            _identity_scope(),
            tenant_id="t1",
            owner_user_id="u1",
        )


def _admission_fixture(model_confidence: float) -> tuple[ProposalValidationResult, EvidenceEpisode, MemoryScope]:
    archive = _archive("数据库正式决定使用 PostgreSQL")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    assert episode.origin.primary_scope is not None
    event = episode.events[0]
    ref = EvidenceRef.from_event(
        event,
        source_uri=episode.source_uris[0],
        span_start=0,
        span_end=len(event.text()),
    )
    proposal = _identity_proposal({"canonical_value": "PostgreSQL"})
    proposal = replace(
        proposal,
        semantic=SemanticAssessment(
            "confirmation",
            "confirmed",
            "current",
            "unrelated",
            "assertion",
            "source_actor",
            "durable",
            "none",
            "atomic",
        ),
        evidence_refs=(ref,),
        suggested_scope_refs=(episode.origin.primary_scope,),
        field_evidence_refs={
            "identity.decision_topic": (ref,),
            "value.canonical_value": (ref,),
            "semantic.speech_act": (ref,),
            "semantic.commitment": (ref,),
            "semantic.temporal_scope": (ref,),
            "semantic.relation_to_existing": (ref,),
            "semantic.utterance_mode": (ref,),
            "semantic.attribution": (ref,),
            "semantic.durability": (ref,),
            "semantic.modal_force": (ref,),
            "semantic.atomicity": (ref,),
            "transition": (ref,),
        },
        confidence=model_confidence,
        semantic_contract_version="v3",
        atomic_evidence_ref=ref,
        metadata={
            "source_role": "user",
            "semantic_contract_validated": True,
            "atomic_evidence_validated": True,
            "transition_evidence_validated": True,
        },
    )
    proposal = MemorySemanticNormalizer().normalize(proposal)
    assert isinstance(proposal.semantic, NormalizedSemanticAssessment)
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
        canonical_subject=episode.origin.primary_scope,
        authority=AuthorityPolicy(principal_ids=("u1",)),
    )
    return ProposalValidationResult(True, proposal), episode, scope


def test_model_confidence_is_only_one_admission_signal() -> None:
    validation, episode, scope = _admission_fixture(0.99)
    weak_identity = replace(
        validation,
        proposal=replace(validation.proposal, field_evidence_refs={}),
        unsupported_fields=("identity.decision_topic",),
    )
    rejected_high_model = ProposalAdmissionGate().evaluate(
        weak_identity,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )
    strong_validation, episode, scope = _admission_fixture(0.10)
    accepted_low_model = ProposalAdmissionGate().evaluate(
        strong_validation,
        episode=episode,
        memory_scope=scope,
        source_role="user",
    )

    assert rejected_high_model.decision == ProposalAdmissionDecision.PENDING
    assert rejected_high_model.reason == "system_admission_score_below_threshold"
    assert rejected_high_model.score_components["model_confidence"] == 0.99
    assert accepted_low_model.decision == ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE
    assert accepted_low_model.score_components["model_confidence"] == 0.10
