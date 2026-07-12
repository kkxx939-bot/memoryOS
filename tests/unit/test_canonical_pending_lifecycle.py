from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

import pytest

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.memory.canonical import (
    CandidateProposalAdapter,
    CanonicalMemoryFormationService,
    CanonicalMemoryRepository,
    MemorySemanticProposal,
    ProposalAdmissionDecision,
    RevisionConflictError,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.extraction import MemoryExtractionBatchResult, RejectedMemoryCandidate
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def _archive(*, task_id: str = "pending-task") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="pending-session",
        archive_uri="memoryos://user/u1/sessions/history/pending-session",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "Project rule: Redis must perhaps be used after review.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id=task_id,
        created_at="2026-07-11T01:00:00Z",
    )


def _pending_draft(*, missing_source: bool = False) -> MemoryCandidateDraft:
    return MemoryCandidateDraft(
        memory_type=MemoryType.PROJECT_RULE,
        title="Redis review",
        content="Project rule: Redis must perhaps be used after review.",
        fields={"rule_topic": "redis_usage", "rule": "Redis", "project_id": "memoryos"},
        confidence=0.68,
        source_role="user",
        source_adapter_id="codex",
        source_session_id="pending-session",
        source_message_ids=["missing" if missing_source else "m1"],
        merge_key="project_rule:redis_usage",
        reason="needs review",
    )


def _stores(tmp_path):  # noqa: ANN001, ANN202
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    queue = InMemoryQueueStore()
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relations, queue_store=queue)
    return source, index, relations, queue, committer


def _persist_pending(tmp_path):  # noqa: ANN001, ANN202
    source, index, relations, queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formation = CanonicalMemoryFormationService(source)
    formed = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id="pending-hardening-create",
    )
    committer.commit("u1", list(formed.operations))
    return source, index, relations, queue, committer, formation, formed


def test_admission_pending_is_stable_durable_and_review_query_only(tmp_path) -> None:  # noqa: ANN001
    source, index, relations, queue, committer = _stores(tmp_path)
    archive = _archive()
    SessionArchiveStore(tmp_path, tenant_id="t1").write_sync_archive(archive)
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    service = CanonicalMemoryFormationService(source)

    first = service.plan(
        proposal,
        archive=archive,
        episode=episode,
        retrieval_views=["project:memoryos:rules"],
        commit_group_id="commit_group_pending-task",
    )
    repeated = service.plan(
        proposal,
        archive=archive,
        episode=episode,
        retrieval_views=["project:memoryos:rules"],
        commit_group_id="commit_group_pending-task",
    )

    assert first.decision == ProposalAdmissionDecision.PENDING
    assert len(first.operations) == 1
    assert first.operations[0].operation_id == repeated.operations[0].operation_id
    assert first.operations[0].target_uri == repeated.operations[0].target_uri
    assert first.operations[0].payload["context_object"] == repeated.operations[0].payload["context_object"]

    committed = committer.commit("u1", list(first.operations))
    retried = committer.commit("u1", list(repeated.operations))
    assert [item.operation_id for item in committed.operations] == [item.operation_id for item in retried.operations]

    uri = str(first.operations[0].target_uri)
    record = CanonicalMemoryRepository(source).load_pending(uri, tenant_id="t1", owner_user_id="u1")
    payload = record.to_payload()
    assert record.lifecycle_state == LifecycleState.PENDING
    assert record.pending_reason_code == "PENDING_FALLBACK_REQUIRES_SEMANTIC_REVIEW"
    assert payload["proposal_id"] == proposal.proposal_id
    assert payload["memory_type"] == "project_rule"
    assert payload["identity_fields"]
    assert payload["value_fields"]
    assert payload["scope"]["canonical_subject"]
    assert payload["source_role"] == "user"
    assert payload["semantic_assessment"]
    assert payload["field_evidence_refs"]
    assert payload["extractor_name"]
    assert payload["created_at"] == payload["updated_at"]
    assert payload["retry_count"] == 0
    assert CanonicalMemoryRepository(source).list_pending(tenant_id="t1", owner_user_id="u1") == (record,)

    db = ContextDB(source, index, relations, queue_store=queue, committer=committer)
    assembler = ContextAssembler(db)
    assert (
        assembler.search(
            "Redis",
            user_id="u1",
            tenant_id="t1",
            project_id="memoryos",
            context_type="memory",
        )
        == []
    )
    review = assembler.search(
        "Redis",
        user_id="u1",
        tenant_id="t1",
        project_id="memoryos",
        context_type="memory",
        search_scope="candidates",
    )
    assert [item["uri"] for item in review] == [uri]


def test_pending_lifecycle_transitions_use_operation_committer_and_terminal_states_do_not_reopen(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formation = CanonicalMemoryFormationService(source)
    formed = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        retrieval_views=["project:memoryos:rules"],
        commit_group_id="pending-create",
    )
    committer.commit("u1", list(formed.operations))
    uri = str(formed.operations[0].target_uri)
    repository = CanonicalMemoryRepository(source)
    pending_record = repository.load_pending(uri, tenant_id="t1", owner_user_id="u1")

    retryable_record = pending_record.with_lifecycle(
        LifecycleState.RETRYABLE,
        reason="transient_review_failure",
        retry_increment=True,
    )
    assert retryable_record.retry_count == 1
    repeated_retryable = retryable_record.with_lifecycle(
        LifecycleState.RETRYABLE,
        reason="second_retry_attempt",
        retry_increment=True,
    )
    assert repeated_retryable.retry_count == 2
    assert repeated_retryable.lifecycle_revision == retryable_record.lifecycle_revision + 1
    assert retryable_record.with_lifecycle(LifecycleState.PENDING).lifecycle_state == LifecycleState.PENDING
    for terminal_state in (LifecycleState.REJECTED, LifecycleState.EXPIRED):
        terminal_record = pending_record.with_lifecycle(terminal_state, reason="review_terminal")
        with pytest.raises(ValueError, match="illegal pending proposal lifecycle transition"):
            terminal_record.with_lifecycle(LifecycleState.PENDING)

    confirmed = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="pending-review",
        reason="human_confirmed",
        updated_at="2026-07-11T02:00:00Z",
    )
    committer.commit("u1", [confirmed])
    confirmed_record = repository.load_pending(uri, tenant_id="t1", owner_user_id="u1")
    assert confirmed_record.lifecycle_state == LifecycleState.CONFIRMED
    assert confirmed_record.lifecycle_history[-1]["from"] == "pending"

    repeated = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="pending-review",
        reason="human_confirmed",
    )
    assert repeated.operation_id == confirmed.operation_id
    committer.commit("u1", [repeated])

    with pytest.raises(ValueError, match="linked canonical ACTIVE Claim"):
        formation.plan_pending_lifecycle_transition(
            uri,
            LifecycleState.RESOLVED,
            tenant_id="t1",
            owner_user_id="u1",
            commit_group_id="unlinked-resolve",
        )


def test_pending_identity_is_stable_across_extractor_proposal_ids_and_reason_changes(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formation = CanonicalMemoryFormationService(source)
    first = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="first_review_reason",
        retrieval_views=["project:memoryos:rules"],
        commit_group_id="stable-pending",
    )
    committer.commit("u1", list(first.operations))

    regenerated = replace(proposal, proposal_id="model-generated-different-id")
    repeated = formation.plan_pending(
        regenerated,
        archive=archive,
        episode=episode,
        reason="changed_review_wording",
        retrieval_views=["project:memoryos:rules"],
        commit_group_id="stable-pending",
    )

    assert repeated.operations == ()
    assert repeated.pending_existing is True
    assert repeated.pending_uri == first.operations[0].target_uri
    assert repeated.pending_lifecycle_state == LifecycleState.PENDING.value
    assert repeated.pending_lifecycle_revision == 1
    records = CanonicalMemoryRepository(source).list_pending(tenant_id="t1", owner_user_id="u1")
    assert len(records) == 1
    assert records[0].proposal_id == proposal.proposal_id


def test_pending_lifecycle_commit_uses_revision_cas(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formation = CanonicalMemoryFormationService(source)
    formed = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id="cas-create",
    )
    committer.commit("u1", list(formed.operations))
    uri = str(formed.operations[0].target_uri)
    confirm = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="reviewer-a",
    )
    reject = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.REJECTED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="reviewer-b",
    )

    committer.commit("u1", [confirm])
    with pytest.raises(RevisionConflictError, match="pending proposal lifecycle conflict"):
        committer.commit("u1", [reject])
    current = CanonicalMemoryRepository(source).load_pending(uri, tenant_id="t1", owner_user_id="u1")
    assert current.lifecycle_state == LifecycleState.CONFIRMED
    assert current.lifecycle_revision == 2


def test_pending_lifecycle_commit_rejects_forged_state_history_content_and_immutable_proposal(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer, formation, formed = _persist_pending(tmp_path)
    uri = str(formed.operations[0].target_uri)
    current = CanonicalMemoryRepository(source).load_pending(uri, tenant_id="t1", owner_user_id="u1")
    legal_rejection = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.REJECTED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="forged-state-base",
        reason="reviewed",
        updated_at="2026-07-11T03:00:00Z",
    )
    forged_state = deepcopy(legal_rejection)
    forged_state.operation_id = "op_forged_pending_resolved"
    forged_state.payload["idempotency_key"] = "forged-pending-resolved"
    forged_desired = replace(
        current,
        lifecycle_state=LifecycleState.RESOLVED,
        lifecycle_revision=current.lifecycle_revision + 1,
        lifecycle_history=(
            *current.lifecycle_history,
            {
                "from": current.lifecycle_state.value,
                "to": LifecycleState.RESOLVED.value,
                "from_revision": current.lifecycle_revision,
                "to_revision": current.lifecycle_revision + 1,
                "reason": "reviewed",
                "updated_at": "2026-07-11T03:00:00Z",
            },
        ),
        updated_at="2026-07-11T03:00:00Z",
    )
    forged_state.payload.update(
        {
            "pending_lifecycle_state": LifecycleState.RESOLVED.value,
            "pending_lifecycle_revision": forged_desired.lifecycle_revision,
            "pending_lifecycle_resolution": True,
            "resolution_idempotency_keys": ["forged-key"],
            "resolved_claim_uris": ["memoryos://user/u1/memories/canonical/slots/x/claims/y"],
            "context_object": forged_desired.to_context_object(tenant_id="t1", owner_user_id="u1").to_dict(),
            "content": forged_desired.content(),
        }
    )
    with pytest.raises(ValueError, match="illegal pending proposal lifecycle transition"):
        committer.commit("u1", [forged_state])

    legal_confirmation = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="forged-content-base",
        reason="reviewed",
        updated_at="2026-07-11T04:00:00Z",
    )
    forged_content = deepcopy(legal_confirmation)
    forged_content.operation_id = "op_forged_pending_content"
    forged_content.payload["content"] = "{}"
    with pytest.raises(ValueError, match="content does not match"):
        committer.commit("u1", [forged_content])

    desired = (
        CanonicalMemoryRepository(source)
        .load_pending(uri, tenant_id="t1", owner_user_id="u1")
        .with_lifecycle(
            LifecycleState.CONFIRMED,
            reason="reviewed",
            updated_at="2026-07-11T05:00:00Z",
        )
    )
    rewritten = replace(
        desired,
        proposal=replace(
            desired.proposal,
            value_fields={**dict(desired.proposal.value_fields), "rule": "forged rewrite"},
        ),
    )
    forged_proposal = deepcopy(legal_confirmation)
    forged_proposal.operation_id = "op_forged_pending_proposal"
    forged_proposal.payload["pending_lifecycle_revision"] = rewritten.lifecycle_revision
    forged_proposal.payload["context_object"] = rewritten.to_context_object(
        tenant_id="t1", owner_user_id="u1"
    ).to_dict()
    forged_proposal.payload["content"] = rewritten.content()
    with pytest.raises(ValueError, match="cannot rewrite proposal content or scope"):
        committer.commit("u1", [forged_proposal])

    unchanged = CanonicalMemoryRepository(source).load_pending(uri, tenant_id="t1", owner_user_id="u1")
    assert unchanged.lifecycle_state == LifecycleState.PENDING
    assert unchanged.lifecycle_revision == 1


def test_pending_object_cannot_bypass_lifecycle_with_regular_mutations(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer, formation, formed = _persist_pending(tmp_path)
    uri = str(formed.operations[0].target_uri)
    confirmation = formation.plan_pending_lifecycle_transition(
        uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="bypass-base",
    )
    direct_update = deepcopy(confirmation)
    direct_update.operation_id = "op_direct_pending_update"
    direct_update.payload.pop("pending_lifecycle_transition")
    direct_update.payload.pop("canonical_pending_proposal")
    with pytest.raises(ValueError, match="require a legal lifecycle UPDATE"):
        committer.commit("u1", [direct_update])

    current = CanonicalMemoryRepository(source).load_pending(uri, tenant_id="t1", owner_user_id="u1")
    direct_delete = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.DELETE,
        target_uri=uri,
        operation_id="op_direct_pending_delete",
        payload={
            "reason": "bypass",
            "tenant_id": "t1",
            "memory_type": current.proposal.memory_type,
            "scope": current.scope.to_dict(),
        },
    )
    with pytest.raises(ValueError, match="require a legal lifecycle UPDATE"):
        committer.commit("u1", [direct_delete])
    current = CanonicalMemoryRepository(source).load_pending(uri, tenant_id="t1", owner_user_id="u1")
    assert current.lifecycle_state == LifecycleState.PENDING


def test_pending_add_requires_stable_uri_and_cannot_overwrite_any_existing_object(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formed = CanonicalMemoryFormationService(source).plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id="stable-uri-create",
    )
    original = formed.operations[0]
    forged = deepcopy(original)
    forged.operation_id = "op_arbitrary_pending_uri"
    arbitrary_uri = "memoryos://user/u1/memories/pending/arbitrary"
    forged.target_uri = arbitrary_uri
    forged.payload["context_object"]["uri"] = arbitrary_uri
    forged.payload["context_object"]["layers"] = {
        "l0_uri": f"{arbitrary_uri}/.abstract.md",
        "l1_uri": f"{arbitrary_uri}/.overview.md",
        "l2_uri": f"{arbitrary_uri}/content.md",
    }
    with pytest.raises(ValueError, match="identity or content is invalid"):
        committer.commit("u1", [forged])

    target_uri = str(original.target_uri)
    source.write_object(
        ContextObject(
            uri=target_uri,
            context_type=ContextType.MEMORY,
            title="ordinary existing memory",
            owner_user_id="u1",
            tenant_id="t1",
        ),
        content="ordinary",
    )
    with pytest.raises(ValueError, match="cannot overwrite an existing object"):
        committer.commit("u1", [original])
    assert source.read_object(target_uri).title == "ordinary existing memory"


def test_pending_add_source_written_redo_resumes_without_being_treated_as_overwrite(tmp_path) -> None:  # noqa: ANN001
    source, index, _relations, _queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formed = CanonicalMemoryFormationService(source).plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id="pending-redo-create",
    )
    operation = formed.operations[0]
    fresh_retry = ContextOperation.from_dict(deepcopy(operation.to_dict()))
    relation_manifest = committer._build_regular_relation_manifest(operation)
    committer.redo.begin(
        operation,
        phase="started",
        relation_manifest=relation_manifest,
    )
    committer._apply_source(operation)
    committer._apply_regular_relation_manifest(operation, relation_manifest)
    committer.redo.advance(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation, relation_manifest),
        relation_manifest=relation_manifest,
    )

    diff = committer.commit("u1", [fresh_retry])

    assert [item.operation_id for item in diff.operations] == [operation.operation_id]
    assert str(operation.target_uri) in index.indexed_uris()
    assert len(CanonicalMemoryRepository(source).list_pending(tenant_id="t1", owner_user_id="u1")) == 1
    assert not committer.redo.pending_entries()


def test_pending_add_source_written_redo_rejects_tampered_source_effect(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer = _stores(tmp_path)
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formed = CanonicalMemoryFormationService(source).plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id="pending-redo-tamper",
    )
    operation = formed.operations[0]
    fresh_retry = ContextOperation.from_dict(deepcopy(operation.to_dict()))
    relation_manifest = committer._build_regular_relation_manifest(operation)
    committer.redo.begin(
        operation,
        phase="started",
        relation_manifest=relation_manifest,
    )
    committer._apply_source(operation)
    committer._apply_regular_relation_manifest(operation, relation_manifest)
    committer.redo.advance(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation, relation_manifest),
        relation_manifest=relation_manifest,
    )
    tampered = source.read_object(str(operation.target_uri))
    tampered.title = "tampered pending source"
    source.write_object(tampered, content="tampered content")

    with pytest.raises((ValueError, RedoIntegrityError), match="SourceStore effect does not match"):
        committer.commit("u1", [fresh_retry])

    assert not committer._operation_marker(operation.operation_id).exists()
    assert committer.redo.pending_entries()


def test_regular_marker_binds_effect_and_normalizes_derived_layers_on_fresh_retry(tmp_path) -> None:  # noqa: ANN001
    source, _index, _relations, _queue, committer = _stores(tmp_path)
    obj = ContextObject(
        uri="memoryos://user/u1/memories/profile/retry",
        context_type=ContextType.MEMORY,
        title="retry memory",
        owner_user_id="u1",
        tenant_id="t1",
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        operation_id="op_regular_effect_retry",
        payload={"context_object": obj.to_dict(), "content": "stable content", "tenant_id": "t1"},
    )
    fresh_retry = ContextOperation.from_dict(deepcopy(operation.to_dict()))
    first = committer.commit("u1", [operation])
    repeated = committer.commit("u1", [fresh_retry])
    assert first.to_dict() == repeated.to_dict()

    forged = ContextOperation.from_dict(deepcopy(fresh_retry.to_dict()))
    forged.payload["context_object"]["title"] = "forged title"
    forged.payload["content"] = "forged content"
    with pytest.raises(ValueError, match="requested effect"):
        committer.commit("u1", [forged])
    assert source.read_object(obj.uri).title == "retry memory"
    assert source.read_content(f"{obj.uri}/content.md") == "stable content"


def test_automatic_target_update_redo_adopts_persisted_target_and_preserves_unchanged_content(tmp_path) -> None:  # noqa: ANN001
    source, index, _relations, _queue, committer = _stores(tmp_path)
    current = ContextObject(
        uri="memoryos://user/u1/memories/preferences/automatic-redo",
        context_type=ContextType.MEMORY,
        title="automatic redo preference",
        owner_user_id="u1",
        tenant_id="t1",
    )
    source.write_object(current, content="existing L2 content")
    index.upsert_index(current, content="automatic redo preference existing L2 content")
    desired = ContextObject.from_dict(current.to_dict())
    desired.title = "automatic redo preference updated"
    raw = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        operation_id="op_automatic_redo",
        payload={
            "query": "automatic redo preference",
            "tenant_id": "t1",
            "context_object": desired.to_dict(),
            "content": "",
        },
    )
    fresh_retry = ContextOperation.from_dict(deepcopy(raw.to_dict()))
    resolved = committer.target_resolver.resolve(raw, user_id="u1")
    assert resolved.resolved and raw.target_uri == current.uri
    relation_manifest = committer._build_regular_relation_manifest(raw)
    committer.redo.begin(
        raw,
        phase="started",
        relation_manifest=relation_manifest,
    )
    committer._apply_source(raw)
    committer._apply_regular_relation_manifest(raw, relation_manifest)
    committer.redo.advance(
        raw,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(raw, relation_manifest),
        relation_manifest=relation_manifest,
    )

    diff = committer.commit("u1", [fresh_retry])

    assert diff.operations[0].target_uri == current.uri
    assert source.read_object(current.uri).title == "automatic redo preference updated"
    assert source.read_content(current.uri) == "existing L2 content"
    assert not committer.redo.pending_entries()


class _PendingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],  # noqa: ARG002
    ) -> list[MemorySemanticProposal]:
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return [CandidateProposalAdapter().adapt(_pending_draft(), episode, archive)]

    def extract_with_context(self, archive, schemas, *, existing_memories, episode):  # noqa: ANN001, ANN201, ARG002
        return [CandidateProposalAdapter().adapt(_pending_draft(), episode, archive)]


def test_all_pending_session_result_reports_durable_substate(tmp_path) -> None:  # noqa: ANN001
    source, index, relations, queue, committer = _stores(tmp_path)
    planner = MemoryCommitPlanner(
        extractor=_PendingExtractor(),
        source_store=source,
        index_store=index,
        relation_store=relations,
    )
    service = SessionCommitService(
        SessionArchiveStore(tmp_path, tenant_id="t1"),
        queue,
        committer=committer,
        memory_planner=planner,
    )

    result = service.async_commit(_archive())

    assert result.status == "done_with_pending"
    assert result.archive_committed is True
    assert result.canonical_active_operation_count == 0
    assert result.pending_count == 1
    assert result.pending_persisted is True
    memory_diff = json.loads(
        (tmp_path / "tenants/t1/users/u1/sessions/history/pending-session/memory_diff.json").read_text(encoding="utf-8")
    )
    assert memory_diff["archive_committed"] is True
    assert memory_diff["canonical_active_operation_count"] == 0
    assert memory_diff["pending_count"] == 1
    assert memory_diff["pending_persisted"] is True
    assert len(CanonicalMemoryRepository(source).list_pending(tenant_id="t1", owner_user_id="u1")) == 1


@pytest.mark.parametrize(
    ("lifecycle_state", "expected_count", "expected_persisted"),
    [
        (LifecycleState.RETRYABLE, 1, True),
        (LifecycleState.CONFIRMED, 1, True),
        (LifecycleState.REJECTED, 0, False),
        (LifecycleState.EXPIRED, 0, False),
    ],
)
def test_session_pending_count_tracks_only_outstanding_lifecycle_states(
    tmp_path,  # noqa: ANN001
    lifecycle_state: LifecycleState,
    expected_count: int,
    expected_persisted: bool,
) -> None:
    source, index, relations, queue, committer = _stores(tmp_path)
    archive = _archive(task_id=f"pending-state-{lifecycle_state.value}")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(_pending_draft(missing_source=True), episode, archive)
    formation = CanonicalMemoryFormationService(source)
    created = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id=f"create-{lifecycle_state.value}",
    )
    committer.commit("u1", list(created.operations))
    pending_uri = str(created.operations[0].target_uri)
    transition = formation.plan_pending_lifecycle_transition(
        pending_uri,
        lifecycle_state,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id=f"review-{lifecycle_state.value}",
        retry_increment=lifecycle_state == LifecycleState.RETRYABLE,
    )
    service = SessionCommitService(
        SessionArchiveStore(tmp_path, tenant_id="t1"),
        queue,
        committer=committer,
    )

    result = service._commit_memory_with_reconcile_retry(archive, [transition])

    assert result["pending_count"] == expected_count
    assert result["pending_persisted"] is expected_persisted
    record = CanonicalMemoryRepository(source).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    )
    assert record.lifecycle_state == lifecycle_state


class _BatchExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def extract(
        self,
        archive: SessionArchive,  # noqa: ARG002
        schemas: Sequence[MemoryTypeSchema],  # noqa: ARG002
    ) -> tuple[MemorySemanticProposal, ...]:
        return ()

    def extract_batch_with_context(self, archive, schemas, *, existing_memories, episode):  # noqa: ANN001, ANN201, ARG002
        return MemoryExtractionBatchResult(
            accepted=(),
            rejected=(
                RejectedMemoryCandidate(
                    index=4,
                    proposal_id="bad-4",
                    reason="candidate[4] evidence does not exist",
                    security_flags=("fabricated_evidence",),
                ),
            ),
            security_flags=("candidate_rejected",),
        )


def test_batch_rejections_and_security_flags_are_preserved_in_planning_context() -> None:
    planner = MemoryCommitPlanner(extractor=_BatchExtractor())
    result = planner.plan(_archive(task_id="batch-task"))

    assert result.operations == ()
    assert result.context.extraction_security_flags == ("candidate_rejected",)
    assert result.context.proposal_outcomes[0].proposal_id == "bad-4"
    assert result.context.proposal_outcomes[0].decision == "REJECT"
    assert result.context.proposal_outcomes[0].candidate_index == 4
    assert result.context.proposal_outcomes[0].security_flags == ("fabricated_evidence",)
