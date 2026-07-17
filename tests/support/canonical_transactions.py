from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TypedDict

from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical import (
    Atomicity,
    Attribution,
    CanonicalMemoryFormationService,
    CanonicalMemoryRepository,
    Commitment,
    Durability,
    EpistemicStatus,
    EvidenceRef,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemoryTransactionPlanner,
    MemoryTransitionPolicy,
    ModalForce,
    PendingMemoryProposal,
    ScopeSelector,
    SemanticAssessment,
    SemanticRelation,
    SessionArchiveEpisodeAdapter,
    SpeechAct,
    StableMemoryIdentityResolver,
    UtteranceMode,
    VisibilityPolicy,
    bind_field_evidence,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler
from memoryos.memory.canonical.review_command import PendingReviewCommandStore
from memoryos.operations.commit.operation_committer import OperationCommitter


class _ReviewCommandBinding(TypedDict):
    review_command_id: str
    review_decision: str
    review_request_digest: str


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
        "semantic.utterance_mode": evidence_refs,
        "semantic.attribution": evidence_refs,
        "semantic.durability": evidence_refs,
        "semantic.modal_force": evidence_refs,
        "semantic.atomicity": evidence_refs,
        "transition": evidence_refs,
    }
    return bind_field_evidence(
        identity_fields,
        value_fields,
        evidence_refs,
        bindings=bindings,
        semantic_contract_version="v3",
    )


def _artifact_root(root, tenant_id: str = "t1"):  # noqa: ANN001, ANN202
    return root if tenant_id == "default" else root / "tenants" / tenant_id


def _setup(tmp_path):  # noqa: ANN001, ANN202
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    queue = InMemoryQueueStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(
        source,
        index,
        str(tmp_path),
        relation_store=relations,
        queue_store=queue,
    )
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "The primary storage backend is SQLite. PostgreSQL can be evaluated later.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    SessionArchiveStore(tmp_path, tenant_id="t1").write_sync_archive(archive)
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    return source, index, queue, relations, committer, episode, scope


def _persisted_episode(tmp_path, archive: SessionArchive):  # noqa: ANN001, ANN202
    tenant_id = str(archive.metadata.get("tenant_id") or "default")
    SessionArchiveStore(tmp_path, tenant_id=tenant_id).write_sync_archive(archive)
    return SessionArchiveEpisodeAdapter().adapt(archive)


def _proposal(episode, proposal_id: str, value: str, speech: str, commitment: str):  # noqa: ANN001, ANN202
    assert episode.origin.primary_scope is not None
    identity_fields = {"decision_topic": "primary storage backend"}
    value_fields = {"canonical_value": value}
    text = episode.events[0].text()
    folded = text.casefold()
    value_start = folded.find(value.casefold())
    if value_start < 0:
        value_start = 0
    left = max(text.rfind(mark, 0, value_start) for mark in (".", "。", "!", "！", "?", "？", "\n")) + 1
    right_positions = [
        position + 1
        for mark in (".", "。", "!", "！", "?", "？", "\n")
        if (position := text.find(mark, value_start)) >= 0
    ]
    right = min(right_positions) if right_positions else len(text)
    evidence_refs = (
        EvidenceRef.from_event(
            episode.events[0],
            source_uri=episode.source_uris[0],
            span_start=left,
            span_end=right,
        ),
    )
    return MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(
                speech,
                commitment,
                "current",
                "alternative",
                UtteranceMode.ASSERTION.value,
                Attribution.SOURCE_ACTOR.value,
                Durability.DURABLE.value,
                ModalForce.NONE.value,
                Atomicity.ATOMIC.value,
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=_explicit_bindings(identity_fields, value_fields, evidence_refs),
            confidence=0.95,
            extractor_version="fake_v3",
            prompt_version="fake_v3",
            semantic_contract_version="v3",
            atomic_evidence_ref=evidence_refs[0],
            metadata={
                "source_role": "user",
                "transition_evidence_validated": True,
                "semantic_contract_validated": True,
                "atomic_evidence_validated": True,
            },
        )
    )


def _replacement_proposal(episode, proposal_id: str, value: str, target_claim):  # noqa: ANN001, ANN202
    proposal = _proposal(episode, proposal_id, value, "correction", "confirmed")
    return replace(
        proposal,
        semantic=replace(
            proposal.semantic,
            speech_act=SpeechAct.CORRECTION,
            commitment=Commitment.CONFIRMED,
            relation_to_existing=SemanticRelation.SUPERSEDES,
        ),
        related_claim_ids=(target_claim.claim_id,),
        metadata={
            **dict(proposal.metadata),
            "transition_evidence_validated": True,
            "semantic_relation_evidence_validated": True,
            "replacement_evidence_validated": True,
        },
    )


def _supplement_proposal(
    episode,
    proposal_id: str,
    target_claim,
    *,
    speech_act: SpeechAct,
    commitment: Commitment,
):  # noqa: ANN001, ANN202
    proposal = _proposal(episode, proposal_id, "SQLite", speech_act.value, commitment.value)
    values = {"canonical_value": "SQLite", "rationale": "stable under load"}
    return replace(
        proposal,
        value_fields=values,
        semantic=replace(
            proposal.semantic,
            speech_act=speech_act,
            commitment=commitment,
            relation_to_existing=SemanticRelation.SUPPLEMENTS,
        ),
        related_claim_ids=(target_claim.claim_id,),
        field_evidence_refs=_explicit_bindings(
            dict(proposal.identity_fields),
            values,
            proposal.evidence_refs,
        ),
    )


def _entity_aliases_proposal(
    episode,
    proposal_id: str,
    aliases: list[str],
    *,
    target_claim=None,
):  # noqa: ANN001, ANN202
    base = _proposal(episode, proposal_id, "SQLite", "confirmation", "confirmed")
    identity_fields = {
        "entity_type": "database",
        "canonical_entity_id": "sqlite",
    }
    value_fields = {
        "canonical_value": "SQLite",
        "name": "SQLite",
        "aliases": aliases,
    }
    return replace(
        base,
        memory_type="entity",
        identity_fields=identity_fields,
        value_fields=value_fields,
        semantic=replace(
            base.semantic,
            relation_to_existing=(
                SemanticRelation.SUPPLEMENTS
                if target_claim is not None
                else SemanticRelation.UNRELATED
            ),
        ),
        related_claim_ids=(target_claim.claim_id,) if target_claim is not None else (),
        metadata={
            **dict(base.metadata),
            "system_identity_fields": sorted(identity_fields),
            "system_value_fields": ["aliases"],
        },
        field_evidence_refs=_explicit_bindings(
            identity_fields,
            value_fields,
            base.evidence_refs,
        ),
    )


def _plan(  # noqa: ANN001, ANN202
    source,
    episode,
    scope,
    proposal,
    *,
    destructive_effect_authorized: bool = False,
    commit_group_id: str = "",
):
    identity = StableMemoryIdentityResolver().resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    reconciled = MemorySemanticReconciler().reconcile(proposal, identity, slot=slot, claims=claims)
    transition_policy = MemoryTransitionPolicy()
    if destructive_effect_authorized:
        pending = PendingMemoryProposal.create(
            proposal,
            scope,
            tenant_id="t1",
            owner_user_id="u1",
            source_role="user",
            pending_reason_code="test_review",
            request_identity=proposal.proposal_id,
        )
        pending = replace(
            pending,
            lifecycle_state=LifecycleState.CONFIRMED,
            lifecycle_revision=2,
        )
        transition = transition_policy._apply_confirmed_pending_review(
            pending,
            proposal,
            identity,
            reconciled,
            authorization_id=f"test-transaction:{proposal.proposal_id}",
            owner_user_id="u1",
            tenant_id="t1",
        )
    else:
        transition = transition_policy.apply(proposal, identity, reconciled)
    plan = MemoryTransactionPlanner().build(
        proposal,
        scope,
        transition,
        tenant_id="t1",
        owner_user_id="u1",
        episode_id=episode.episode_id,
        commit_group_id=commit_group_id,
    )
    return identity, transition, plan


def _reviewed_resolution_plan(  # noqa: ANN001, ANN202
    source,
    committer,
    episode,
    proposal,
    *,
    command_suffix: str = "review",
):
    """Build a real receipt-backed CONFIRM_AND_APPLY plan for commit tests."""

    formation = CanonicalMemoryFormationService(
        source,
        relation_store=committer.relation_store,
    )
    archive = SessionArchiveStore(committer.root, tenant_id="t1").read_archive(
        episode.source_uris[0],
        tenant_id="t1",
    )
    pending_result = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="destructive_effect_requires_structured_review",
        commit_group_id=f"pending-create:{command_suffix}",
    )
    if pending_result.operations:
        committer.commit("u1", list(pending_result.operations))
    pending_uri = str(pending_result.pending_uri or pending_result.operations[0].target_uri)
    pending = CanonicalMemoryRepository(
        source,
        committer.relation_store,
    ).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    )
    command_id = f"command-{stable_hash([pending_uri, command_suffix], length=32)}"
    command = PendingReviewCommandStore(committer.root, tenant_id="t1").begin(
        command_id,
        owner_user_id="u1",
        pending_uri=pending_uri,
        decision="CONFIRM_AND_APPLY",
        expected_lifecycle_revision=pending.lifecycle_revision,
        expected_proposal_fingerprint=pending.proposal.fingerprint,
        reason="test structured review",
    )
    confirmation = formation.plan_pending_lifecycle_transition(
        pending_uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id=f"pending-confirm:{command_suffix}",
        reason=f"structured_review:{command_id}",
        review_command_id=command_id,
        review_decision="CONFIRM_AND_APPLY",
        review_request_digest=str(command["request_digest"]),
    )
    committer.commit("u1", [confirmation])
    confirmed = CanonicalMemoryRepository(
        source,
        committer.relation_store,
    ).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    )
    return formation.plan_confirmed_pending_resolution(
        pending_uri,
        confirmed.proposal,
        archive=archive,
        episode=episode,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id=f"pending-resolution:{command_suffix}",
        reason=f"structured_review:{command_id}",
        review_command_id=command_id,
        review_decision="CONFIRM_AND_APPLY",
        review_request_digest=str(command["request_digest"]),
    )


def _review_command_binding(  # noqa: ANN001, ANN202
    committer,
    pending,
    *,
    decision: str,
    suffix: str,
    reason: str = "test review",
) -> _ReviewCommandBinding:
    command_id = f"command-{stable_hash([pending.uri, decision, suffix], length=32)}"
    command = PendingReviewCommandStore(committer.root, tenant_id="t1").begin(
        command_id,
        owner_user_id="u1",
        pending_uri=pending.uri,
        decision=decision,
        expected_lifecycle_revision=pending.lifecycle_revision,
        expected_proposal_fingerprint=pending.proposal.fingerprint,
        reason=reason,
    )
    return {
        "review_command_id": command_id,
        "review_decision": decision,
        "review_request_digest": str(command["request_digest"]),
    }
