from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

import memoryos.memory.canonical as canonical_api
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import (
    AliasRegistry,
    CanonicalMemoryFormationService,
    Commitment,
    EpistemicStatus,
    EvidenceRef,
    MemoryClaim,
    MemoryRevision,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemoryTransitionPolicy,
    PendingMemoryProposal,
    PendingReason,
    PendingSemanticReconciliation,
    ProposalEvidenceValidator,
    ScopeRef,
    ScopeSelector,
    SemanticAssessment,
    SemanticRelation,
    SessionArchiveEpisodeAdapter,
    SpeechAct,
    StableMemoryIdentityResolver,
    TransitionProfile,
    VisibilityPolicy,
    bind_field_evidence,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler, RelationAuthority
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def _explicit_bindings(
    identity_fields: Mapping[str, object],
    value_fields: Mapping[str, object],
    evidence_refs: tuple[EvidenceRef, ...],
    *,
    semantic_contract_version: str = "v2",
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
    if semantic_contract_version == "v3":
        bindings.update(
            {
                "semantic.utterance_mode": evidence_refs,
                "semantic.attribution": evidence_refs,
                "semantic.durability": evidence_refs,
                "semantic.modal_force": evidence_refs,
                "semantic.atomicity": evidence_refs,
            }
        )
    return bind_field_evidence(
        identity_fields,
        value_fields,
        evidence_refs,
        bindings=bindings,
        semantic_contract_version=semantic_contract_version,
    )


def _context():  # noqa: ANN202
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": "The primary storage backend is SQLite. PostgreSQL is a future option.",
                }
            ],
            metadata={
                "tenant_id": "t1",
                "project_id": "memoryos",
                "connect": {"adapter_id": "codex"},
            },
        )
    )
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    return episode, scope


def _proposal(episode, value: str, speech: str, commitment: str, proposal_id: str):  # noqa: ANN001, ANN202
    assert episode.origin.primary_scope is not None
    identity_fields = {"decision_topic": "primary storage backend"}
    value_fields = {"canonical_value": value}
    text = episode.events[0].text()
    atomic_ref = EvidenceRef.from_event(
        episode.events[0],
        source_uri=episode.source_uris[0],
        span_start=0,
        span_end=len(text),
    )
    evidence_refs = (atomic_ref,)
    return MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(
                speech,
                commitment,
                "future" if speech in {"future_option", "proposal", "evaluation_request"} else "current",
                "alternative",
                "assertion",
                "source_actor",
                "durable",
                "none",
                "atomic",
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=_explicit_bindings(
                identity_fields,
                value_fields,
                evidence_refs,
                semantic_contract_version="v3",
            ),
            confidence=0.95,
            extractor_version="fake",
            semantic_contract_version="v3",
            atomic_evidence_ref=atomic_ref,
            metadata={
                "source_role": "user",
                "transition_evidence_validated": True,
                "semantic_contract_validated": True,
                "atomic_evidence_validated": True,
            },
        )
    )


def _apply(  # noqa: ANN001, ANN202
    proposal,
    scope,
    slot=None,
    claims=(),
    *,
    destructive_effect_authorized: bool = False,
):
    identity = StableMemoryIdentityResolver().resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    reconciliation = MemorySemanticReconciler().reconcile(proposal, identity, slot=slot, claims=claims)
    policy = MemoryTransitionPolicy()
    return identity, (
        policy._apply_confirmed_pending_review(
            _confirmed_pending(proposal, scope),
            proposal,
            identity,
            reconciliation,
            authorization_id=f"test-review:{proposal.proposal_id}",
            owner_user_id="u1",
            tenant_id="t1",
        )
        if destructive_effect_authorized
        else policy.apply(proposal, identity, reconciliation)
    )


def _confirmed_pending(proposal, scope):  # noqa: ANN001, ANN202
    pending = PendingMemoryProposal.create(
        proposal,
        scope,
        tenant_id="t1",
        owner_user_id="u1",
        source_role="user",
        pending_reason_code=PendingReason.REVIEWABLE_DESTRUCTIVE,
        request_identity=proposal.proposal_id,
    )
    return replace(
        pending,
        lifecycle_state=LifecycleState.CONFIRMED,
        lifecycle_revision=2,
    )


def _validated_decision_proposal(
    text: str,
    value: str,
    *,
    proposal_id: str,
    speech_act: str,
    commitment: str,
    relation: str,
    related_claim_ids: tuple[str, ...] = (),
    value_fields: Mapping[str, object] | None = None,
):  # noqa: ANN202
    archive = SessionArchive(
        user_id="u1",
        session_id=proposal_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{proposal_id}",
        messages=[{"id": "m1", "role": "user", "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    identity_fields = {"decision_topic": "primary_storage_backend"}
    values = {"canonical_value": value, **dict(value_fields or {})}
    atomic_ref = EvidenceRef.from_event(
        episode.events[0],
        source_uri=episode.source_uris[0],
        span_start=0,
        span_end=len(text),
    )
    evidence_refs = [atomic_ref]
    bindings = _explicit_bindings(
        identity_fields,
        values,
        (atomic_ref,),
        semantic_contract_version="v3",
    )
    for field_name, field_value in values.items():
        literal = str(field_value)
        if literal not in text:
            continue
        start = text.index(literal)
        child = EvidenceRef.from_event(
            episode.events[0],
            source_uri=episode.source_uris[0],
            span_start=start,
            span_end=start + len(literal),
        )
        evidence_refs.append(child)
        bindings[f"value.{field_name}"] = (child,)
    proposal = MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields=identity_fields,
            value_fields=values,
            semantic=SemanticAssessment(
                speech_act,
                commitment,
                "current",
                relation,
                "assertion",
                "source_actor",
                "durable",
                "none",
                "atomic",
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=(),
            related_claim_ids=related_claim_ids,
            evidence_refs=tuple(evidence_refs),
            field_evidence_refs=bindings,
            confidence=0.99,
            extractor_version="test",
            semantic_contract_version="v3",
            atomic_evidence_ref=atomic_ref,
            metadata={
                "source_role": "user",
                "system_identity_fields": ["decision_topic"],
            },
        )
    )
    validation = ProposalEvidenceValidator().validate(proposal, episode)
    assert validation.valid, validation.errors
    return validation.proposal, scope


def _v3_decision_proposal(
    text: str,
    value: str,
    *,
    proposal_id: str,
    temporal_scope: str = "current",
    relation: str = "unrelated",
    utterance_mode: str = "assertion",
    attribution: str = "source_actor",
    durability: str = "durable",
    modal_force: str = "none",
    atomicity: str = "atomic",
    related_claim_ids: tuple[str, ...] = (),
    include_atomic_ref: bool = True,
) -> tuple[MemorySemanticProposal, MemoryScope]:
    archive = SessionArchive(
        user_id="u1",
        session_id=proposal_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{proposal_id}",
        messages=[{"id": "m1", "role": "user", "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    identity_fields = {"decision_topic": "primary_storage_backend"}
    value_fields = {"canonical_value": value}
    atomic_ref = EvidenceRef.from_event(
        episode.events[0],
        source_uri=episode.source_uris[0],
        span_start=0,
        span_end=len(text),
    )
    proposal = MemorySemanticProposal(
        proposal_id=proposal_id,
        memory_type="project_decision",
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=SemanticAssessment(
            "correction" if relation in {"corrects", "supersedes"} else "confirmation",
            "confirmed",
            temporal_scope,
            relation,
            utterance_mode,
            attribution,
            durability,
            modal_force,
            atomicity,
        ),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        related_claim_ids=related_claim_ids,
        evidence_refs=(atomic_ref,),
        field_evidence_refs=_explicit_bindings(
            identity_fields,
            value_fields,
            (atomic_ref,),
            semantic_contract_version="v3",
        ),
        confidence=0.99,
        extractor_version="test-v3",
        semantic_contract_version="v3",
        atomic_evidence_ref=atomic_ref if include_atomic_ref else None,
        metadata={
            "source_role": "user",
            "semantic_contract_validated": True,
            "atomic_evidence_validated": True,
            "transition_evidence_validated": True,
            "semantic_relation_evidence_validated": True,
            "replacement_evidence_validated": True,
        },
    )
    return MemorySemanticNormalizer().normalize(proposal), scope


def _v3_non_authoritative_proposal(
    *,
    memory_type: str,
    source_role: str,
    temporal_scope: str,
    epistemic_status: EpistemicStatus,
) -> tuple[MemorySemanticProposal, MemoryScope]:
    text = "The deployment completed and the verified approach can be reused."
    archive = SessionArchive(
        user_id="u1",
        session_id=f"typed-{memory_type}-{source_role}",
        archive_uri=f"memoryos://user/u1/sessions/history/typed-{memory_type}-{source_role}",
        messages=[{"id": "m1", "role": source_role, "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    identity_fields = (
        {"event_key": "deployment_completed"}
        if memory_type == "event"
        else {"task_pattern": "deploy", "environment_signature": "memoryos"}
    )
    value_fields = {"canonical_value": "deployment completed"}
    atomic_ref = EvidenceRef.from_event(
        episode.events[0],
        source_uri=episode.source_uris[0],
        span_start=0,
        span_end=len(text),
    )
    proposal = MemorySemanticProposal(
        proposal_id=f"typed-{memory_type}-{source_role}",
        memory_type=memory_type,
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=SemanticAssessment(
            "observation",
            "weak",
            temporal_scope,
            "unrelated",
            "assertion",
            "source_actor",
            "durable",
            "none",
            "atomic",
        ),
        epistemic_status=epistemic_status,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=(atomic_ref,),
        field_evidence_refs=_explicit_bindings(
            identity_fields,
            value_fields,
            (atomic_ref,),
            semantic_contract_version="v3",
        ),
        confidence=0.99,
        extractor_version="test-v3-typed",
        semantic_contract_version="v3",
        atomic_evidence_ref=atomic_ref,
        metadata={
            "source_role": source_role,
            "semantic_contract_validated": True,
            "atomic_evidence_validated": True,
            "transition_evidence_validated": True,
        },
    )
    return MemorySemanticNormalizer().normalize(proposal), scope


def test_slot_identity_ignores_body_and_claim_values_share_slot() -> None:
    episode, scope = _context()
    sqlite = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-sqlite")
    postgres = _proposal(episode, "PostgreSQL", "future_option", "exploratory", "p-postgres")
    resolver = StableMemoryIdentityResolver()
    sqlite_identity = resolver.resolve(sqlite, scope, tenant_id="t1", owner_user_id="u1")
    postgres_identity = resolver.resolve(postgres, scope, tenant_id="t1", owner_user_id="u1")
    body_changed = replace(sqlite, metadata={"body": "a completely different summary"})
    assert resolver.resolve(body_changed, scope, tenant_id="t1", owner_user_id="u1").slot_id == sqlite_identity.slot_id
    assert sqlite_identity.slot_id == postgres_identity.slot_id
    assert sqlite_identity.claim_id != postgres_identity.claim_id


def test_different_workspaces_do_not_share_slot_and_agents_in_one_workspace_do() -> None:
    episode, scope = _context()
    proposal = _proposal(episode, "SQLite", "confirmation", "confirmed", "p1")
    resolver = StableMemoryIdentityResolver()
    first = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    other_scope = replace(
        scope,
        applicability=ScopeSelector((ScopeRef("memoryos", "workspace", "other"),)),
    )
    other = resolver.resolve(proposal, other_scope, tenant_id="t1", owner_user_id="u1")
    agent_changed = replace(proposal, model_id="another-agent")
    assert first.slot_id != other.slot_id
    assert first.slot_id == resolver.resolve(agent_changed, scope, tenant_id="t1", owner_user_id="u1").slot_id


def test_tenant_and_canonical_subject_are_stable_slot_boundary_not_author() -> None:
    episode, scope = _context()
    proposal = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-owner")
    resolver = StableMemoryIdentityResolver()
    first = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    other_owner = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u2")
    other_tenant = resolver.resolve(proposal, scope, tenant_id="t2", owner_user_id="u1")
    assert first.slot_id == other_owner.slot_id
    assert first.slot_uri == other_owner.slot_uri
    assert first.slot_id != other_tenant.slot_id


def test_alias_registry_maps_reachy_names_to_one_stable_asset() -> None:
    aliases = AliasRegistry(
        {
            "scope:asset": {
                "Reachy Mini": "reachy_01",
                "reachy-mini": "reachy_01",
                "客厅机器人": "reachy_01",
            }
        }
    )
    assert {
        aliases.canonical_scope(ScopeRef("memoryos", "asset", name)).id
        for name in ("Reachy Mini", "reachy-mini", "客厅机器人")
    } == {"reachy_01"}


def test_confirming_an_existing_alternative_does_not_implicitly_supersede_active_claim() -> None:
    episode, scope = _context()
    sqlite = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-sqlite")
    _, first = _apply(sqlite, scope)
    assert first.claims[0].current.state == "ACTIVE"

    postgres = _proposal(episode, "PostgreSQL", "future_option", "exploratory", "p-postgres")
    with pytest.raises(PendingSemanticReconciliation, match="nonfinal_relation_requires_review"):
        _apply(postgres, scope, first.slot, first.claims)
    assert {claim.canonical_value: claim.current.state for claim in first.claims} == {"sqlite": "ACTIVE"}

    confirmed = replace(
        postgres,
        proposal_id="p-confirm",
        semantic=replace(
            postgres.semantic,
            speech_act=SpeechAct.CONFIRMATION,
            commitment=Commitment.CONFIRMED,
        ),
    )
    with pytest.raises(PendingSemanticReconciliation, match="relation_requires_confirmation"):
        _apply(confirmed, scope, first.slot, first.claims)
    assert {claim.canonical_value: claim.current.state for claim in first.claims} == {"sqlite": "ACTIVE"}


@pytest.mark.parametrize(
    ("text", "speech_act", "commitment"),
    [
        ("确认 MySQL 也可以作为备用", "confirmation", "confirmed"),
        ("PostgreSQL 保持不变，MySQL 作为备用", "proposal", "exploratory"),
    ],
)
def test_mysql_backup_never_replaces_active_postgresql(
    text: str,
    speech_act: str,
    commitment: str,
) -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-active",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _validated_decision_proposal(
        text,
        "MySQL",
        proposal_id=f"mysql-backup-{speech_act}",
        speech_act=speech_act,
        commitment=commitment,
        relation="alternative",
        related_claim_ids=(first.slot.active_claim_id,),
    )

    with pytest.raises(PendingSemanticReconciliation, match="nonfinal_relation_requires_review"):
        _apply(mysql, mysql_scope, first.slot, first.claims)

    assert {claim.canonical_value: claim.current.state for claim in first.claims} == {
        "postgresql": "ACTIVE",
    }


def test_explicit_mysql_switch_requires_target_and_replacement_evidence() -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-before-switch",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _validated_decision_proposal(
        "数据库正式改为 MySQL",
        "MySQL",
        proposal_id="mysql-explicit-switch",
        speech_act="confirmation",
        commitment="confirmed",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
    )
    assert mysql.metadata["relation_target_binding_validated"] is True
    assert mysql.metadata["semantic_relation_evidence_validated"] is False
    assert mysql.metadata["replacement_evidence_validated"] is False

    with pytest.raises(PendingSemanticReconciliation, match="destructive_effect_requires_structured_review"):
        _apply(mysql, mysql_scope, first.slot, first.claims)

    mysql_identity = StableMemoryIdentityResolver().resolve(
        mysql,
        mysql_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    mysql_reconciliation = MemorySemanticReconciler().reconcile(
        mysql,
        mysql_identity,
        slot=first.slot,
        claims=first.claims,
    )
    transition_policy = MemoryTransitionPolicy()
    assert not hasattr(canonical_api, "DestructiveEffectAuthorization")
    assert not hasattr(transition_policy, "_issue_effect_authorization")
    mismatched_pending = _confirmed_pending(postgres, mysql_scope)
    with pytest.raises(
        PendingSemanticReconciliation,
        match="destructive_effect_requires_confirmed_pending_record",
    ):
        transition_policy._apply_confirmed_pending_review(
            mismatched_pending,
            mysql,
            mysql_identity,
            mysql_reconciliation,
            authorization_id="mismatched-review",
            owner_user_id="u1",
            tenant_id="t1",
        )

    _, second = _apply(
        mysql,
        mysql_scope,
        first.slot,
        first.claims,
        destructive_effect_authorized=True,
    )

    assert {claim.canonical_value: claim.current.state for claim in second.claims} == {
        "postgresql": "SUPERSEDED",
        "mysql": "ACTIVE",
    }
    active = next(claim for claim in second.claims if claim.current.state == "ACTIVE")
    assert active.current.relation == SemanticRelation.SUPERSEDES.value


def test_model_mislabelled_backup_cannot_authorize_destructive_transition() -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-before-mislabel",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mislabeled, mislabeled_scope = _validated_decision_proposal(
        "确认 MySQL 只作为备用",
        "MySQL",
        proposal_id="mysql-backup-mislabeled-supersedes",
        speech_act="correction",
        commitment="confirmed",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
    )
    assert mislabeled.metadata["relation_target_binding_validated"] is True
    assert mislabeled.metadata["semantic_relation_evidence_validated"] is False
    assert mislabeled.metadata["replacement_evidence_validated"] is False
    mislabeled = replace(
        mislabeled,
        metadata={
            **dict(mislabeled.metadata),
            "relation_semantic_authority": "structured_confirmed_review",
            "effect_authority": "structured_explicit_command",
        },
    )
    mislabeled_identity = StableMemoryIdentityResolver().resolve(
        mislabeled,
        mislabeled_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    mislabeled_reconciliation = MemorySemanticReconciler().reconcile(
        mislabeled,
        mislabeled_identity,
        slot=first.slot,
        claims=first.claims,
    )
    assert mislabeled_reconciliation.relation_authority == RelationAuthority.MODEL_REPORTED

    with pytest.raises(PendingSemanticReconciliation, match="destructive_effect_requires_structured_review"):
        _apply(mislabeled, mislabeled_scope, first.slot, first.claims)

    assert {claim.canonical_value: claim.current.state for claim in first.claims} == {
        "postgresql": "ACTIVE",
    }


def test_transition_rejects_relation_target_tuple_that_disagrees_with_repository_state() -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-relation-state",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _validated_decision_proposal(
        "数据库正式改为 MySQL",
        "MySQL",
        proposal_id="mysql-relation-state",
        speech_act="correction",
        commitment="confirmed",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
    )
    inconsistent = replace(mysql, related_slot_ids=("another-slot",))

    with pytest.raises(PendingSemanticReconciliation, match="relation_slot_target_mismatch"):
        _apply(inconsistent, mysql_scope, first.slot, first.claims)


def test_v3_atomic_source_durable_current_assertion_can_become_active() -> None:
    proposal, scope = _v3_decision_proposal(
        "The current primary storage backend is PostgreSQL.",
        "PostgreSQL",
        proposal_id="v3-current-source",
    )

    _, transition = _apply(proposal, scope)

    assert transition.claims[0].current.state == "ACTIVE"


def test_transition_rejects_legacy_contract_even_when_semantic_fields_look_safe() -> None:
    proposal, scope = _v3_decision_proposal(
        "The current primary storage backend is PostgreSQL.",
        "PostgreSQL",
        proposal_id="legacy-contract-bypass",
    )
    legacy = replace(proposal, semantic_contract_version="v2")

    with pytest.raises(PendingSemanticReconciliation, match="semantic_contract_v3_required"):
        _apply(legacy, scope)


@pytest.mark.parametrize(
    ("memory_type", "identity_fields", "value_fields", "reason"),
    [
        (
            "preference",
            {"subject": "user", "dimension": "response_style"},
            {"canonical_value": "concise"},
            "preference_modal_force_inconsistent",
        ),
        (
            "project_rule",
            {"rule_topic": "storage_backend"},
            {"canonical_value": "REQUIRED", "constraint_polarity": "REQUIRED"},
            "project_rule_semantic_inconsistent",
        ),
    ],
)
def test_transition_repeats_type_specific_semantic_guards(
    memory_type: str,
    identity_fields: Mapping[str, object],
    value_fields: Mapping[str, object],
    reason: str,
) -> None:
    proposal, scope = _v3_decision_proposal(
        "This is an intentionally inconsistent semantic proposal.",
        "placeholder",
        proposal_id=f"transition-type-guard-{memory_type}",
    )
    inconsistent = replace(
        proposal,
        memory_type=memory_type,
        identity_fields=identity_fields,
        value_fields=value_fields,
    )

    with pytest.raises(PendingSemanticReconciliation, match=reason):
        _apply(inconsistent, scope)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"temporal_scope": "future"}, "semantic_v3_authoritative_temporality_pending"),
        (
            {"utterance_mode": "question"},
            "semantic_v3_question_or_hypothetical_or_quoted_or_transient",
        ),
        (
            {"attribution": "third_party"},
            "semantic_v3_unknown_or_mixed_or_compound_or_third_party",
        ),
        (
            {"durability": "transient"},
            "semantic_v3_question_or_hypothetical_or_quoted_or_transient",
        ),
        (
            {"atomicity": "compound"},
            "semantic_v3_unknown_or_mixed_or_compound_or_third_party",
        ),
        ({"include_atomic_ref": False}, "atomic_evidence_invalid_or_missing"),
    ],
)
def test_v3_critical_gate_rejects_non_authoritative_active_effects(
    overrides: dict[str, Any],
    reason: str,
) -> None:
    proposal, scope = _v3_decision_proposal(
        "The primary storage backend is PostgreSQL.",
        "PostgreSQL",
        proposal_id=f"v3-gate-{reason}",
        **overrides,
    )

    with pytest.raises(PendingSemanticReconciliation, match=reason):
        _apply(proposal, scope)


def test_v3_valid_replacement_keeps_existing_target_and_evidence_guards() -> None:
    postgres, scope = _v3_decision_proposal(
        "The current primary storage backend is PostgreSQL.",
        "PostgreSQL",
        proposal_id="v3-postgres-active",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _v3_decision_proposal(
        "The primary storage backend is now replaced by MySQL.",
        "MySQL",
        proposal_id="v3-mysql-replacement",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
    )

    with pytest.raises(PendingSemanticReconciliation, match="destructive_effect_requires_structured_review"):
        _apply(mysql, mysql_scope, first.slot, first.claims)

    _, second = _apply(
        mysql,
        mysql_scope,
        first.slot,
        first.claims,
        destructive_effect_authorized=True,
    )

    assert {claim.canonical_value: claim.current.state for claim in second.claims} == {
        "postgresql": "SUPERSEDED",
        "mysql": "ACTIVE",
    }


@pytest.mark.parametrize(
    ("memory_type", "source_role", "epistemic_status"),
    [
        ("event", "user", EpistemicStatus.OBSERVED),
        ("event", "assistant", EpistemicStatus.INFERRED),
        ("agent_experience", "assistant", EpistemicStatus.INFERRED),
    ],
)
def test_transition_allows_schema_authorized_past_event_and_experience(
    memory_type: str,
    source_role: str,
    epistemic_status: EpistemicStatus,
) -> None:
    proposal, scope = _v3_non_authoritative_proposal(
        memory_type=memory_type,
        source_role=source_role,
        temporal_scope="past",
        epistemic_status=epistemic_status,
    )

    _, transition = _apply(proposal, scope)

    assert transition.claims[0].current.state == "ACTIVE"


@pytest.mark.parametrize(
    ("temporal_scope", "epistemic_status", "reason"),
    [
        ("future", EpistemicStatus.OBSERVED, "semantic_v3_non_authoritative_temporality_pending"),
        ("past", EpistemicStatus.HYPOTHESIZED, "hypothesis_requires_confirmation"),
    ],
)
def test_transition_blocks_nonfinal_event_effects(
    temporal_scope: str,
    epistemic_status: EpistemicStatus,
    reason: str,
) -> None:
    proposal, scope = _v3_non_authoritative_proposal(
        memory_type="event",
        source_role="user",
        temporal_scope=temporal_scope,
        epistemic_status=epistemic_status,
    )

    with pytest.raises(PendingSemanticReconciliation, match=reason):
        _apply(proposal, scope)


def test_mysql_available_confirmation_without_replacement_relation_is_pending() -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-still-active",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    mysql, mysql_scope = _validated_decision_proposal(
        "确认 MySQL 可用",
        "MySQL",
        proposal_id="mysql-available",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    identity = StableMemoryIdentityResolver().resolve(
        mysql,
        mysql_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    reconciliation = MemorySemanticReconciler().reconcile(
        mysql,
        identity,
        slot=first.slot,
        claims=first.claims,
    )

    assert reconciliation.relation == SemanticRelation.AMBIGUOUS
    with pytest.raises(PendingSemanticReconciliation):
        MemoryTransitionPolicy().apply(mysql, identity, reconciliation)
    assert first.claims[0].current.state == "ACTIVE"


def test_transition_rejects_forged_supersedes_from_alternative() -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-forged-guard",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _validated_decision_proposal(
        "确认 MySQL 也可以作为备用",
        "MySQL",
        proposal_id="mysql-forged-supersedes",
        speech_act="confirmation",
        commitment="confirmed",
        relation="alternative",
        related_claim_ids=(first.slot.active_claim_id,),
    )
    identity = StableMemoryIdentityResolver().resolve(
        mysql,
        mysql_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    reconciled = MemorySemanticReconciler().reconcile(mysql, identity, slot=first.slot, claims=first.claims)
    forged = replace(reconciled, relation=SemanticRelation.SUPERSEDES)

    transition_policy = MemoryTransitionPolicy()
    with pytest.raises(PendingSemanticReconciliation, match="cannot_upgrade_from_non_replacement"):
        transition_policy._apply_confirmed_pending_review(
            _confirmed_pending(mysql, mysql_scope),
            mysql,
            identity,
            forged,
            authorization_id="forged-alternative-review",
            owner_user_id="u1",
            tenant_id="t1",
        )


def test_reconciler_and_transition_reject_self_target_supersedes() -> None:
    postgres, scope = _validated_decision_proposal(
        "数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-self-target-base",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    repeated, repeated_scope = _validated_decision_proposal(
        "数据库正式改为 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-self-target-switch",
        speech_act="correction",
        commitment="confirmed",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
    )
    identity = StableMemoryIdentityResolver().resolve(
        repeated,
        repeated_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    reconciled = MemorySemanticReconciler().reconcile(
        repeated,
        identity,
        slot=first.slot,
        claims=first.claims,
    )

    assert reconciled.relation == SemanticRelation.DUPLICATE
    forged = replace(reconciled, relation=SemanticRelation.SUPERSEDES)
    transition_policy = MemoryTransitionPolicy()
    with pytest.raises(PendingSemanticReconciliation, match="replacement_cannot_target_same_claim"):
        transition_policy._apply_confirmed_pending_review(
            _confirmed_pending(repeated, repeated_scope),
            repeated,
            identity,
            forged,
            authorization_id="self-target-review",
            owner_user_id="u1",
            tenant_id="t1",
        )


def test_replacement_with_conflicting_applicability_is_pending() -> None:
    postgres, scope = _validated_decision_proposal(
        "生产环境数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-production",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
        value_fields={"environment": "生产环境"},
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _validated_decision_proposal(
        "测试环境数据库正式改为 MySQL",
        "MySQL",
        proposal_id="mysql-test-only",
        speech_act="confirmation",
        commitment="confirmed",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
        value_fields={"environment": "测试环境"},
    )
    identity = StableMemoryIdentityResolver().resolve(
        mysql,
        mysql_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    reconciled = MemorySemanticReconciler().reconcile(mysql, identity, slot=first.slot, claims=first.claims)

    assert reconciled.relation == SemanticRelation.AMBIGUOUS
    with pytest.raises(PendingSemanticReconciliation):
        MemoryTransitionPolicy().apply(mysql, identity, reconciled)


def test_replacement_cannot_drop_active_applicability_qualifier() -> None:
    postgres, scope = _validated_decision_proposal(
        "生产环境数据库继续使用 PostgreSQL",
        "PostgreSQL",
        proposal_id="postgres-production-qualified",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
        value_fields={"environment": "生产环境"},
    )
    _, first = _apply(postgres, scope)
    assert first.slot.active_claim_id is not None
    mysql, mysql_scope = _validated_decision_proposal(
        "数据库正式改为 MySQL",
        "MySQL",
        proposal_id="mysql-missing-environment",
        speech_act="correction",
        commitment="confirmed",
        relation="supersedes",
        related_claim_ids=(first.slot.active_claim_id,),
    )
    identity = StableMemoryIdentityResolver().resolve(
        mysql,
        mysql_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    reconciled = MemorySemanticReconciler().reconcile(mysql, identity, slot=first.slot, claims=first.claims)

    assert reconciled.relation == SemanticRelation.AMBIGUOUS
    with pytest.raises(PendingSemanticReconciliation):
        MemoryTransitionPolicy().apply(mysql, identity, reconciled)


def test_unconfirmed_supplement_cannot_revise_active_claim_but_confirmed_details_can() -> None:
    postgres, scope = _validated_decision_proposal(
        "确认 PostgreSQL 作为当前数据库",
        "PostgreSQL",
        proposal_id="postgres-before-supplement",
        speech_act="confirmation",
        commitment="confirmed",
        relation="unrelated",
    )
    _, first = _apply(postgres, scope)
    active = next(claim for claim in first.claims if claim.current.state == "ACTIVE")
    weak, weak_scope = _validated_decision_proposal(
        "PostgreSQL 可以考虑补充理由：稳定",
        "PostgreSQL",
        proposal_id="postgres-weak-supplement",
        speech_act="proposal",
        commitment="weak",
        relation="supplements",
        related_claim_ids=(active.claim_id,),
        value_fields={"rationale": "稳定"},
    )
    weak_identity = StableMemoryIdentityResolver().resolve(
        weak,
        weak_scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    weak_reconciled = MemorySemanticReconciler().reconcile(
        weak,
        weak_identity,
        slot=first.slot,
        claims=first.claims,
    )
    assert weak_reconciled.relation == SemanticRelation.SUPPLEMENTS
    assert weak_reconciled.relation_authority == RelationAuthority.STATE_DERIVED
    with pytest.raises(
        PendingSemanticReconciliation,
        match="nonfinal_relation_requires_review|unconfirmed_supplement",
    ):
        MemoryTransitionPolicy().apply(weak, weak_identity, weak_reconciled)
    assert len(active.revisions) == 1
    assert dict(active.current.value_fields) == {"canonical_value": "PostgreSQL"}

    confirmed, confirmed_scope = _validated_decision_proposal(
        "确认 PostgreSQL 补充理由为稳定",
        "PostgreSQL",
        proposal_id="postgres-confirmed-supplement",
        speech_act="confirmation",
        commitment="confirmed",
        relation="supplements",
        related_claim_ids=(active.claim_id,),
        value_fields={"rationale": "稳定"},
    )
    _, second = _apply(confirmed, confirmed_scope, first.slot, first.claims)
    updated = next(claim for claim in second.claims if claim.claim_id == active.claim_id)
    assert len(updated.revisions) == 2
    assert updated.current.state == "ACTIVE"
    assert updated.current.value_fields["rationale"] == "稳定"


def test_authoritative_explicit_observation_cannot_bypass_confirmation_transition() -> None:
    episode, scope = _context()
    observation = _proposal(episode, "SQLite", "observation", "weak", "p-observation")

    with pytest.raises(
        PendingSemanticReconciliation,
        match="semantic_v3_authoritative_commitment_pending",
    ):
        _apply(observation, scope)


def test_claim_revision_history_links_previous_and_validity_interval() -> None:
    episode, scope = _context()
    first = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-first")
    _, initial = _apply(first, scope)
    corrected = replace(
        first,
        proposal_id="p-corrected",
        value_fields={"canonical_value": "SQLite", "reason": "verified"},
        field_evidence_refs={
            **dict(first.field_evidence_refs),
            "value.reason": first.evidence_refs,
        },
        semantic=replace(first.semantic, speech_act=SpeechAct.CORRECTION),
    )
    _, updated = _apply(corrected, scope, initial.slot, initial.claims)
    revisions = updated.claims[0].revisions
    assert len(revisions) == 2
    assert revisions[1].previous_revision == 1
    assert revisions[0] == initial.claims[0].revisions[0]
    assert revisions[0].valid_to is None
    assert revisions[1].valid_from


def test_formation_display_mirror_uses_effective_current_revision_not_history_tail() -> None:
    episode, _scope_ref = _context()
    proposal = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-display")
    current = MemoryRevision(
        revision=1,
        state="ACTIVE",
        value_fields={"canonical_value": "SQLite"},
        evidence_refs=proposal.evidence_refs,
        proposal_id="p-current",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        qualifiers={"display_fields": {"summary": "effective current display"}},
    )
    historical = MemoryRevision(
        revision=2,
        state="PROPOSED",
        value_fields={"canonical_value": "SQLite in the past"},
        evidence_refs=proposal.evidence_refs,
        proposal_id="p-history",
        relation="SUPPLEMENTS",
        epistemic_status="EXPLICIT",
        qualifiers={
            "non_current_historical": True,
            "display_fields": {"summary": "historical-only display"},
        },
        previous_revision=1,
    )
    claim = MemoryClaim(
        "claim-display",
        "memoryos://user/u1/memories/canonical/slots/slot-display/claims/claim-display",
        "slot-display",
        "sqlite",
        TransitionProfile.AUTHORITATIVE_STATE,
        (current, historical),
    )
    obj = claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_decision",
        scope={},
    )
    operation = ContextOperation(
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        target_uri=obj.uri,
        user_id="u1",
        payload={"context_object": obj.to_dict(), "expected_revision": 1},
    )

    CanonicalMemoryFormationService(None)._decorate_operations([operation], proposal, [])

    metadata = operation.payload["context_object"]["metadata"]
    assert metadata["current_revision"] == 1
    assert metadata["revision"] == 2
    assert metadata["display_fields"] == {"summary": "effective current display"}
