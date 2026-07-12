from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import cast

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.planning import MemoryPlanningResult, PlanningContext, ProposalPlanningInput
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.memory.canonical import (
    Atomicity,
    Attribution,
    CanonicalMemoryFormationService,
    CanonicalMemoryProjector,
    CanonicalMemoryRepository,
    Commitment,
    Durability,
    EpistemicStatus,
    EvidenceRef,
    MemoryProjectionWorker,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemoryTransactionPlanner,
    MemoryTransitionPolicy,
    ModalForce,
    PendingMemoryProposal,
    RevisionConflictError,
    ScopeRef,
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
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


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


class FailingSecondWriteSource(FileSystemSourceStore):
    def __init__(self, root) -> None:  # noqa: ANN001
        super().__init__(root)
        self.write_count = 0
        self.armed = True

    def write_object(self, obj, content="") -> None:  # noqa: ANN001
        self.write_count += 1
        if self.armed and self.write_count == 2:
            self.armed = False
            raise OSError("injected batch write failure")
        super().write_object(obj, content)


class FailOnceQueue(InMemoryQueueStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = True

    def enqueue(self, job) -> None:  # noqa: ANN001
        if self.fail_next:
            self.fail_next = False
            raise OSError("injected queue outage")
        super().enqueue(job)


class CrashSecondWriteSource(FileSystemSourceStore):
    def __init__(self, root) -> None:  # noqa: ANN001
        super().__init__(root)
        self.write_count = 0

    def write_object(self, obj, content="") -> None:  # noqa: ANN001
        self.write_count += 1
        if self.write_count == 2:
            raise SystemExit("simulated process crash")
        super().write_object(obj, content)


class ArmableCrashSource(FileSystemSourceStore):
    def __init__(self, root) -> None:  # noqa: ANN001
        super().__init__(root)
        self.write_count = 0
        self.crash_at: int | None = None

    def write_object(self, obj, content="") -> None:  # noqa: ANN001
        self.write_count += 1
        if self.crash_at == self.write_count:
            self.crash_at = None
            raise SystemExit("simulated switch crash")
        super().write_object(obj, content)


def _setup(tmp_path):  # noqa: ANN001, ANN202
    source = FileSystemSourceStore(tmp_path)
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


def _plan(  # noqa: ANN001, ANN202
    source,
    episode,
    scope,
    proposal,
    *,
    destructive_effect_authorized: bool = False,
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
    )
    return identity, transition, plan


def test_expected_revision_idempotency_atomic_claim_switch_and_outbox(tmp_path) -> None:  # noqa: ANN001
    source, index, queue, relations, committer, episode, scope = _setup(tmp_path)
    sqlite = _proposal(episode, "p-sqlite", "SQLite", "confirmation", "confirmed")
    identity, _, first_plan = _plan(source, episode, scope, sqlite)
    first_ops = first_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    first_diff = committer.commit("u1", first_ops)
    repeated = committer.commit("u1", first_ops)
    assert repeated.diff_id == first_diff.diff_id
    assert (tmp_path / "system" / "diffs" / f"{first_diff.diff_id}.json").exists()
    assert [operation.operation_id for operation in repeated.operations] == [
        operation.operation_id for operation in first_diff.operations
    ]
    assert len(queue.jobs) == 1
    assert not index.indexed_uris(), "canonical projection must be outbox-driven"

    stale = [deepcopy(operation) for operation in first_ops]
    for operation in stale:
        operation.payload["transaction_id"] += "_stale"
        operation.payload["idempotency_key"] += "_stale"
    with pytest.raises(RevisionConflictError):
        committer.commit("u1", stale)

    _, claims = CanonicalMemoryRepository(source).load(identity)
    assert {claim.canonical_value: claim.current.state for claim in claims} == {"sqlite": "ACTIVE"}

    sqlite_claim = next(claim for claim in claims if claim.canonical_value == "sqlite")
    replacement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="s-replace",
            archive_uri="memoryos://user/u1/sessions/history/s-replace",
            messages=[
                {
                    "id": "replace-storage",
                    "role": "user",
                    "content": "The primary storage backend is now changed from SQLite to PostgreSQL.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    confirmed = _replacement_proposal(replacement_episode, "p-confirm", "PostgreSQL", sqlite_claim)
    _, transition, third_plan = _plan(
        source,
        replacement_episode,
        scope,
        confirmed,
        destructive_effect_authorized=True,
    )
    assert len(third_plan.operations) == 3, "slot and both claim changes must share one transaction"
    committer.commit(
        "u1",
        third_plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=replacement_episode.episode_id,
        ),
    )
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot and slot.revision == transition.slot.revision
    assert {claim.canonical_value: claim.current.state for claim in claims} == {
        "sqlite": "SUPERSEDED",
        "postgresql": "ACTIVE",
    }
    assert len([claim for claim in claims if claim.current.state == "ACTIVE"]) == 1
    assert len(list((tmp_path / "system" / "outbox").glob("*.json"))) == 2
    assert relations.relations_of(identity.slot_uri, tenant_id="t1", owner_user_id="u1")


def test_canonical_transaction_marker_rejects_same_key_with_changed_request_and_preserves_outbox(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "marker-sqlite", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    committer.commit("u1", operations)
    transaction_id = str(operations[0].payload["transaction_id"])
    idempotency_key = str(operations[0].payload["idempotency_key"])
    outbox_path = tmp_path / "system" / "outbox" / f"{transaction_id}.json"
    marker_path = tmp_path / "system" / "transactions" / f"{idempotency_key}.json"
    before_outbox = outbox_path.read_text(encoding="utf-8")
    before_marker = marker_path.read_text(encoding="utf-8")
    before_objects = {obj.uri: obj.to_dict() for obj in source.list_objects()}

    def unexpected_outbox_rewrite(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("committed outbox must not be rewritten on idempotent retry")

    monkeypatch.setattr(committer, "_write_outbox_event", unexpected_outbox_rewrite)
    repeated = committer.commit("u1", operations)
    assert {item.operation_id for item in repeated.operations} == {item.operation_id for item in operations}

    forged = deepcopy(operations)
    forged[0].payload["context_object"]["title"] = "forged same-key transaction"
    forged[0].payload["content"] = "forged same-key transaction"
    with pytest.raises(ValueError, match="idempotency marker conflicts"):
        committer.commit("u1", forged)

    assert outbox_path.read_text(encoding="utf-8") == before_outbox
    assert marker_path.read_text(encoding="utf-8") == before_marker
    assert {obj.uri: obj.to_dict() for obj in source.list_objects()} == before_objects


def test_noncanonical_operations_cannot_mutate_canonical_slot_or_claim(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "formal-only", "SQLite", "confirmation", "confirmed")
    principal = ScopeRef(namespace="memoryos", kind="principal", id="u1")
    principal_scope = MemoryScope(
        ScopeSelector((principal,)),
        VisibilityPolicy("t1"),
        (principal,),
    )
    identity, _, plan = _plan(source, episode, principal_scope, proposal)
    committer.commit(
        "u1",
        plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot is not None
    claim = claims[0]
    stored_claim = source.read_object(claim.uri)
    scope_payload = dict(stored_claim.metadata["scope"])
    forged_claim = ContextObject.from_dict(stored_claim.to_dict())
    forged_claim.title = "forged direct claim update"
    direct_update = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        target_uri=claim.uri,
        operation_id="op_direct_canonical_claim_update",
        payload={
            "tenant_id": "t1",
            "memory_type": proposal.memory_type,
            "scope": scope_payload,
            "context_object": forged_claim.to_dict(),
            "content": "forged",
        },
    )
    with pytest.raises(ValueError, match="require a canonical transaction"):
        committer.commit("u1", [direct_update])

    stored_slot = source.read_object(identity.slot_uri)
    direct_delete = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.DELETE,
        target_uri=identity.slot_uri,
        operation_id="op_direct_canonical_slot_delete",
        payload={
            "tenant_id": "t1",
            "memory_type": proposal.memory_type,
            "scope": dict(stored_slot.metadata["scope"]),
            "reason": "bypass canonical transaction",
        },
    )
    with pytest.raises(ValueError, match="require a canonical transaction"):
        committer.commit("u1", [direct_delete])

    unchanged_slot, unchanged_claims = CanonicalMemoryRepository(source).load(identity)
    assert unchanged_slot == slot
    assert unchanged_claims == claims


def test_all_canonical_groups_preflight_before_a_later_invalid_group_can_write(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    primary = _proposal(episode, "preflight-primary", "SQLite", "confirmation", "confirmed")
    _identity, _, primary_plan = _plan(source, episode, scope, primary)
    secondary_identity = {"decision_topic": "secondary storage backend"}
    secondary = replace(
        _proposal(episode, "preflight-secondary", "DuckDB", "confirmation", "confirmed"),
        identity_fields=secondary_identity,
        field_evidence_refs=_explicit_bindings(
            secondary_identity,
            {"canonical_value": "DuckDB"},
            primary.evidence_refs,
        ),
    )
    _secondary_identity, _, secondary_plan = _plan(source, episode, scope, secondary)
    primary_operations = primary_plan.to_context_operations(
        user_id="u1", tenant_id="t1", episode_id=episode.episode_id
    )
    invalid_operations = secondary_plan.to_context_operations(
        user_id="u1", tenant_id="t1", episode_id=episode.episode_id
    )
    invalid_operations[0].payload["context_object"] = "malformed"

    with pytest.raises(ValueError, match="requires context_object"):
        committer.commit("u1", [*primary_operations, *invalid_operations])

    assert not [obj for obj in source.list_objects() if dict(obj.metadata or {}).get("canonical_kind")]
    assert not list((tmp_path / "system" / "transactions").glob("*.json"))
    assert not list((tmp_path / "system" / "outbox").glob("*.json"))


def test_mixed_retry_accepts_fresh_automatic_target_operation_and_returns_persisted_effect(tmp_path) -> None:  # noqa: ANN001
    source, index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    target = ContextObject(
        uri="memoryos://user/u1/memories/preferences/temperature",
        context_type=ContextType.MEMORY,
        title="temperature preference",
        owner_user_id="u1",
        tenant_id="t1",
    )
    source.write_object(target, content="temperature preference 26 degrees")
    index.upsert_index(target, content="temperature preference 26 degrees")
    desired = ContextObject.from_dict(target.to_dict())
    desired.title = "temperature preference updated"
    raw_update = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        operation_id="op_automatic_target_retry",
        payload={
            "query": "temperature preference",
            "tenant_id": "t1",
            "context_object": desired.to_dict(),
            "content": "temperature preference 27 degrees",
        },
    )
    fresh_retry = ContextOperation.from_dict(deepcopy(raw_update.to_dict()))
    initial_diff = committer.commit("u1", [raw_update])
    assert initial_diff.operations[0].target_uri == target.uri

    canonical = _proposal(episode, "mixed-auto-target", "SQLite", "confirmation", "confirmed")
    _identity, _, canonical_plan = _plan(source, episode, scope, canonical)
    canonical_operations = canonical_plan.to_context_operations(
        user_id="u1", tenant_id="t1", episode_id=episode.episode_id
    )
    mixed = committer.commit("u1", [*canonical_operations, fresh_retry])

    assert {item.operation_id for item in mixed.operations} == {
        *(item.operation_id for item in canonical_operations),
        raw_update.operation_id,
    }
    persisted_retry = next(item for item in mixed.operations if item.operation_id == raw_update.operation_id)
    assert persisted_retry.target_uri == target.uri
    assert source.read_object(target.uri).title == "temperature preference updated"
    assert source.read_content(f"{target.uri}/content.md") == "temperature preference 27 degrees"


def test_all_regular_effects_preflight_before_a_later_malformed_object_can_write(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, _episode, _scope = _setup(tmp_path)
    first_obj = ContextObject(
        uri="memoryos://user/u1/memories/profile/preflight-first",
        context_type=ContextType.MEMORY,
        title="first",
        owner_user_id="u1",
        tenant_id="t1",
    )
    first = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=first_obj.uri,
        operation_id="op_regular_preflight_first",
        payload={"tenant_id": "t1", "context_object": first_obj.to_dict(), "content": "first"},
    )
    malformed_uri = "memoryos://user/u1/memories/profile/preflight-malformed"
    malformed = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=malformed_uri,
        operation_id="op_regular_preflight_malformed",
        payload={
            "tenant_id": "t1",
            "context_object": {
                "uri": malformed_uri,
                "owner_user_id": "u1",
                "tenant_id": "t1",
            },
            "content": "malformed",
        },
    )

    with pytest.raises(ValueError, match="context_object is invalid"):
        committer.commit("u1", [first, malformed])

    with pytest.raises(FileNotFoundError):
        source.read_object(first_obj.uri)
    assert not list((tmp_path / "system" / "diffs").glob("*.json"))
    assert not list((tmp_path / "system" / "audit").glob("*.jsonl"))
    assert not committer.redo.pending_entries()


def test_canonical_batch_rolls_back_all_source_and_relations_on_mid_batch_failure(tmp_path) -> None:  # noqa: ANN001
    _source, index, queue, relations, _committer, episode, scope = _setup(tmp_path)
    source = FailingSecondWriteSource(tmp_path)
    committer = OperationCommitter(
        source,
        index,
        str(tmp_path),
        relation_store=relations,
        queue_store=queue,
    )
    proposal = _proposal(episode, "p-sqlite", "SQLite", "confirmation", "confirmed")
    identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)

    with pytest.raises(OSError, match="injected batch write failure"):
        committer.commit("u1", operations)

    assert not [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind")]
    assert relations.relations_of(identity.slot_uri, tenant_id="t1", owner_user_id="u1") == []
    outbox = next((tmp_path / "system" / "outbox").glob("*.json"))
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "aborted"
    assert not list((tmp_path / "system" / "transactions").glob("*.json"))


def test_canonical_preflight_rejects_evidence_archive_tampering_before_any_write(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "p-evidence", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    event_path = next(
        (tmp_path / "tenants" / "t1" / "users" / "u1" / "sessions" / "history" / "s1" / "evidence" / "events").glob(
            "*.json"
        )
    )
    event_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="immutable event digest mismatch"):
        committer.commit(
            "u1",
            plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
        )
    assert not [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind")]


def test_canonical_preflight_rejects_owner_mismatch_and_outbox_prepare_failure_writes_nothing(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "p-owner", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    forged = deepcopy(operations)
    forged[0].payload["context_object"]["owner_user_id"] = "u2"
    with pytest.raises(ValueError, match="tenant or owner"):
        committer.commit("u1", forged)
    assert not [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind")]

    def fail_outbox(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("injected outbox prepare failure")

    monkeypatch.setattr(committer, "_write_outbox_event", fail_outbox)
    with pytest.raises(OSError, match="outbox prepare"):
        committer.commit("u1", operations)
    assert not [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind")]


def test_canonical_envelope_validates_commit_user_target_uri_and_immutable_update_scope(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "p-envelope", "SQLite", "confirmation", "confirmed")
    _identity, _, initial_plan = _plan(source, episode, scope, proposal)
    initial = initial_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)

    with pytest.raises(ValueError, match="operation user does not match commit user"):
        committer.commit("u2", initial)
    mismatched_target = deepcopy(initial)
    mismatched_target[0].target_uri = f"{mismatched_target[0].target_uri}-forged"
    with pytest.raises(ValueError, match="target_uri does not match"):
        committer.commit("u1", mismatched_target)

    committer.commit("u1", initial)
    revised_values = {"canonical_value": "SQLite", "rationale": "verified"}
    revised = replace(
        proposal,
        proposal_id="p-envelope-revision",
        value_fields=revised_values,
        semantic=replace(proposal.semantic, speech_act=SpeechAct.CORRECTION),
        field_evidence_refs=_explicit_bindings(
            dict(proposal.identity_fields),
            revised_values,
            proposal.evidence_refs,
        ),
    )
    _identity, _, revision_plan = _plan(source, episode, scope, revised)
    revision_ops = revision_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    forged_scope = deepcopy(revision_ops)
    forged_scope[0].payload["context_object"]["metadata"]["scope"]["visibility"][
        "allowed_principal_ids"
    ] = ["u1", "u2"]
    with pytest.raises(ValueError, match="cannot weaken or change its scope"):
        committer.commit("u1", forged_scope)

def test_canonical_preflight_rejects_v1_identity_and_redirect_payload(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "p-v2-only", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)

    wrong_version = deepcopy(operations)
    wrong_version[0].payload["identity_algorithm_version"] = "identity_v1"
    with pytest.raises(ValueError, match="requires Identity V2"):
        committer.commit("u1", wrong_version)

    redirects = deepcopy(operations)
    for operation in redirects:
        operation.payload["identity_alias_operations"] = []
    with pytest.raises(ValueError, match="cannot contain redirects"):
        committer.commit("u1", redirects)
    assert not [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind")]


def test_committed_outbox_is_dispatched_after_initial_queue_outage(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    queue = FailOnceQueue()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relations, queue_store=queue)
    episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "The storage backend is confirmed as SQLite."}],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    proposal = _proposal(episode, "p-sqlite", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    committer.commit(
        "u1",
        plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    assert queue.jobs == {}

    worker = MemoryProjectionWorker(CanonicalMemoryProjector(source, index, tmp_path), queue)
    result = worker.process_pending()
    assert result["processed"]
    assert index.indexed_uris()


def test_source_committed_transaction_recovers_when_final_outbox_write_fails(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    source, _index, queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "p-final-outbox", "SQLite", "confirmation", "confirmed")
    identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    original_finalize = committer._finalize_canonical_outbox
    calls = 0

    def fail_once(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected final outbox failure")
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(committer, "_finalize_canonical_outbox", fail_once)
    with pytest.raises(OSError, match="final outbox"):
        committer.commit("u1", operations)
    outbox = next((tmp_path / "system" / "outbox").glob("*.json"))
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "source_committed"
    assert CanonicalMemoryRepository(source).load(identity)[0] is not None

    monkeypatch.setattr(committer, "_finalize_canonical_outbox", original_finalize)
    assert set(committer.recover_pending_canonical("u1")) == {operation.operation_id for operation in operations}
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "committed"
    assert queue.jobs


def test_prepared_outbox_and_redo_recover_complete_transaction_after_crash(tmp_path) -> None:  # noqa: ANN001
    source = CrashSecondWriteSource(tmp_path)
    index = InMemoryIndexStore()
    queue = InMemoryQueueStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relations, queue_store=queue)
    episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "The primary storage backend is SQLite."}],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    proposal = _proposal(episode, "p-sqlite", "SQLite", "confirmation", "confirmed")
    identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    with pytest.raises(SystemExit, match="simulated process crash"):
        committer.commit("u1", operations)

    recovery = RecoveryService(committer.redo, committer).recover("u1")
    assert set(recovery.operation_ids) == {operation.operation_id for operation in operations}
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot is not None and {claim.canonical_value: claim.current.state for claim in claims} == {"sqlite": "ACTIVE"}
    outbox = next((tmp_path / "system" / "outbox").glob("*.json"))
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "committed"


def test_canonical_recovery_rejects_same_revision_with_divergent_source_effect(tmp_path) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "recovery-integrity", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    transaction_id = str(operations[0].payload["transaction_id"])
    idempotency_key = str(operations[0].payload["idempotency_key"])
    committer._write_outbox_event(
        transaction_id,
        idempotency_key,
        operations,
        status="prepared",
        before_images=committer._capture_canonical_state(operations),
    )
    for operation in operations:
        committer.redo.begin(operation, phase="started")
    first = operations[0]
    committer._apply_canonical_source(first)
    committer.redo.advance(first, phase="source_written")
    first_payload = first.payload["context_object"]
    tampered = source.read_object(str(first_payload["uri"]))
    tampered.title = "tampered at the same revision"
    source.write_object(tampered, content=str(first.payload.get("content", "")))

    with pytest.raises(RevisionConflictError, match="divergent object at desired revision"):
        committer.resume_canonical_batch("u1", committer.redo.pending_entries())

    assert not (tmp_path / "system" / "transactions" / f"{idempotency_key}.json").exists()
    assert committer.redo.pending_entries()


def test_session_commit_revision_conflict_rereads_and_reconciles_same_proposal(tmp_path) -> None:  # noqa: ANN001
    source, index, queue, relations, committer, sqlite_episode, scope = _setup(tmp_path)
    postgres_archive = SessionArchive(
        user_id="u1",
        session_id="s2",
        archive_uri="memoryos://user/u1/sessions/history/s2",
        messages=[{"id": "m1", "role": "user", "content": "PostgreSQL is a future primary storage backend option."}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    postgres_episode = _persisted_episode(tmp_path, postgres_archive)
    postgres = _proposal(postgres_episode, "p-postgres", "PostgreSQL", "confirmation", "confirmed")
    _identity, _, stale_plan = _plan(source, postgres_episode, scope, postgres)
    stale_operations = stale_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=postgres_episode.episode_id,
    )

    sqlite = _proposal(sqlite_episode, "p-sqlite", "SQLite", "confirmation", "confirmed")
    identity, _, sqlite_plan = _plan(source, sqlite_episode, scope, sqlite)
    committer.commit(
        "u1",
        sqlite_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=sqlite_episode.episode_id),
    )

    planner = MemoryCommitPlanner(source_store=source, index_store=index, relation_store=relations)
    planning_context = PlanningContext(
        planning_id="postgres-stale-plan",
        task_id=postgres_archive.task_id,
        archive_digest=postgres_archive.archive_digest,
        manifest_digest=postgres_archive.manifest_digest,
        episode_id=postgres_episode.episode_id,
        session_id=postgres_archive.session_id,
        tenant_id="t1",
        proposal_inputs=(ProposalPlanningInput(postgres),),
        prefetch_snapshot=(),
        planned_against_revisions=tuple(sorted(stale_plan.expected_revisions.items())),
        staged_objects=(),
        scope_candidates=tuple(scope_ref.key for scope_ref in postgres_episode.legal_scope_candidates()),
        evidence_references=postgres.evidence_refs,
        operation_group_identity=f"commit_group_{postgres_archive.task_id}",
    )
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        queue,
        committer=committer,
        memory_planner=planner,
    )
    service._commit_memory_with_reconcile_retry(postgres_archive, stale_operations, planning_context)
    _slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert {claim.canonical_value: claim.current.state for claim in claims} == {"sqlite": "ACTIVE"}
    pending = [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind") == "pending_proposal"]
    assert len(pending) == 1
    assert pending[0].metadata["proposal"]["value_fields"]["canonical_value"] == "PostgreSQL"


def test_uncommitted_partial_switch_is_invisible_until_transaction_recovery(tmp_path) -> None:  # noqa: ANN001
    source = ArmableCrashSource(tmp_path)
    index = InMemoryIndexStore()
    queue = InMemoryQueueStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relations, queue_store=queue)
    episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": "SQLite is confirmed. PostgreSQL is a future primary storage backend option.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    sqlite = _proposal(episode, "sqlite-initial", "SQLite", "confirmation", "confirmed")
    identity, _, sqlite_plan = _plan(source, episode, scope, sqlite)
    committer.commit(
        "u1",
        sqlite_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    sqlite_claim = next(
        claim
        for claim in CanonicalMemoryRepository(source).load(identity)[1]
        if claim.canonical_value == "sqlite"
    )
    replacement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="s-replace",
            archive_uri="memoryos://user/u1/sessions/history/s-replace",
            messages=[
                {
                    "id": "replace-storage",
                    "role": "user",
                    "content": "The primary storage backend is now changed from SQLite to PostgreSQL.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    confirmed = _replacement_proposal(replacement_episode, "postgres-confirm", "PostgreSQL", sqlite_claim)
    _, _, switch_plan = _plan(
        source,
        replacement_episode,
        scope,
        confirmed,
        destructive_effect_authorized=True,
    )
    switch_operations = switch_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=replacement_episode.episode_id,
    )
    source.crash_at = source.write_count + 2
    with pytest.raises(SystemExit, match="simulated switch crash"):
        committer.commit("u1", switch_operations)

    before_recovery = CanonicalMemoryRepository(source).load(identity)[1]
    assert {claim.canonical_value: claim.current.state for claim in before_recovery} == {"sqlite": "ACTIVE"}
    RecoveryService(committer.redo, committer).recover("u1")
    after_recovery = CanonicalMemoryRepository(source).load(identity)[1]
    assert {claim.canonical_value: claim.current.state for claim in after_recovery} == {
        "sqlite": "SUPERSEDED",
        "postgresql": "ACTIVE",
    }


def test_mixed_pending_is_not_written_when_canonical_preflight_conflicts(tmp_path) -> None:  # noqa: ANN001
    source, index, _queue, relations, committer, episode, scope = _setup(tmp_path)
    archive = SessionArchiveStore(tmp_path, tenant_id="t1").read_archive(
        "memoryos://user/u1/sessions/history/s1",
        tenant_id="t1",
    )
    sqlite = _proposal(episode, "mixed-sqlite", "SQLite", "confirmation", "confirmed")
    _identity, _, initial_plan = _plan(source, episode, scope, sqlite)
    initial = initial_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    committer.commit("u1", initial)
    stale = [deepcopy(operation) for operation in initial]
    for operation in stale:
        operation.payload["transaction_id"] += "_stale"
        operation.payload["idempotency_key"] += "_stale"

    pending = CanonicalMemoryFormationService(source).plan_pending(
        sqlite,
        archive=archive,
        episode=episode,
        reason="manual_review",
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id=f"commit_group_{archive.task_id}",
    )
    pending_uri = str(pending.operations[0].target_uri)

    with pytest.raises(RevisionConflictError) as raised:
        committer.commit("u1", [*stale, *pending.operations])

    assert raised.value.committed_diff is None
    with pytest.raises(FileNotFoundError):
        source.read_object(pending_uri)
    assert pending_uri not in index.indexed_uris()
    assert not any(
        pending.operations[0].operation_id in path.read_text(encoding="utf-8")
        for path in (tmp_path / "system" / "diffs").glob("*.json")
    )
    assert relations.relations_of(pending_uri, tenant_id="t1", owner_user_id="u1") == []


def test_partial_canonical_conflict_replan_merges_diff_then_commits_pending_once(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    source, index, queue, relations, committer, episode, scope = _setup(tmp_path)
    archive_store = SessionArchiveStore(tmp_path, tenant_id="t1")
    archive = archive_store.read_archive(
        "memoryos://user/u1/sessions/history/s1",
        tenant_id="t1",
    )
    primary = _proposal(episode, "primary-sqlite", "SQLite", "confirmation", "confirmed")
    secondary_identity = {"decision_topic": "secondary storage backend"}
    secondary = _proposal(episode, "secondary-duckdb", "DuckDB", "confirmation", "confirmed")
    secondary = replace(
        secondary,
        identity_fields=secondary_identity,
        field_evidence_refs=_explicit_bindings(
            secondary_identity,
            dict(secondary.value_fields),
            secondary.evidence_refs,
        ),
    )
    _primary_identity, _, primary_plan = _plan(source, episode, scope, primary)
    _secondary_identity, _, secondary_plan = _plan(source, episode, scope, secondary)
    primary_ops = primary_plan.to_context_operations(
        user_id="u1", tenant_id="t1", episode_id=episode.episode_id
    )
    secondary_ops = secondary_plan.to_context_operations(
        user_id="u1", tenant_id="t1", episode_id=episode.episode_id
    )
    pending = CanonicalMemoryFormationService(source).plan_pending(
        primary,
        archive=archive,
        episode=episode,
        reason="manual_review",
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id=f"commit_group_{archive.task_id}",
    )

    original_batch = committer._commit_canonical_batch
    batch_calls = 0

    def conflict_second_batch(user_id, operations):  # noqa: ANN001, ANN202
        nonlocal batch_calls
        batch_calls += 1
        if batch_calls == 2:
            raise RevisionConflictError("injected concurrent revision change")
        return original_batch(user_id, operations)

    monkeypatch.setattr(committer, "_commit_canonical_batch", conflict_second_batch)
    planning_context = PlanningContext(
        planning_id="mixed-replan",
        task_id=archive.task_id,
        archive_digest=archive.archive_digest,
        manifest_digest=archive.manifest_digest,
        episode_id=episode.episode_id,
        session_id=archive.session_id,
        tenant_id="t1",
        proposal_inputs=(),
        prefetch_snapshot=(),
        planned_against_revisions=(),
        staged_objects=(),
        scope_candidates=tuple(item.key for item in episode.legal_scope_candidates()),
        evidence_references=primary.evidence_refs,
        operation_group_identity=f"commit_group_{archive.task_id}",
    )

    class StaticReplanner:
        def replan(self, context, requested_archive):  # noqa: ANN001, ANN201, ARG002
            return MemoryPlanningResult(tuple([*secondary_ops, *pending.operations]), context)

    service = SessionCommitService(
        archive_store,
        queue,
        committer=committer,
        memory_planner=cast(MemoryCommitPlanner, StaticReplanner()),
    )
    result = service._commit_memory_with_reconcile_retry(
        archive,
        [*primary_ops, *secondary_ops, *pending.operations],
        planning_context,
    )

    expected_ids = {
        *(operation.operation_id for operation in primary_ops),
        *(operation.operation_id for operation in secondary_ops),
        pending.operations[0].operation_id,
    }
    assert {item["operation_id"] for item in result["operations"]} == expected_ids
    assert result["operation_count"] == len(expected_ids)
    assert result["canonical_active_operation_count"] == 2
    assert result["pending_count"] == 1
    assert result["pending_persisted"] is True
    assert source.read_object(str(pending.operations[0].target_uri)).metadata["canonical_kind"] == "pending_proposal"

    before_audit = (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8")
    before_diffs = {path.name: path.read_text(encoding="utf-8") for path in (tmp_path / "system" / "diffs").glob("*.json")}
    before_objects = {obj.uri: obj.to_dict() for obj in source.list_objects()}
    repeated = service._commit_memory_with_reconcile_retry(
        archive,
        [*primary_ops, *secondary_ops, *pending.operations],
        planning_context,
    )
    assert {item["operation_id"] for item in repeated["operations"]} == expected_ids
    assert (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8") == before_audit
    assert {
        path.name: path.read_text(encoding="utf-8") for path in (tmp_path / "system" / "diffs").glob("*.json")
    } == before_diffs
    assert {obj.uri: obj.to_dict() for obj in source.list_objects()} == before_objects


def test_regular_lifecycle_conflict_preflights_all_groups_then_replan_commits_every_unwritten_effect(tmp_path) -> None:  # noqa: ANN001
    source, index, queue, relations, committer, episode, scope = _setup(tmp_path)
    archive_store = SessionArchiveStore(tmp_path, tenant_id="t1")
    archive = archive_store.read_archive(
        "memoryos://user/u1/sessions/history/s1",
        tenant_id="t1",
    )
    proposal = _proposal(episode, "canonical-before-regular-conflict", "SQLite", "confirmation", "confirmed")
    identity, _, canonical_plan = _plan(source, episode, scope, proposal)
    canonical_operations = canonical_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    formation = CanonicalMemoryFormationService(source)
    pending = formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="manual_review",
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id="regular-conflict-create",
    )
    committer.commit("u1", list(pending.operations))
    pending_uri = str(pending.operations[0].target_uri)
    stale_rejection = formation.plan_pending_lifecycle_transition(
        pending_uri,
        LifecycleState.REJECTED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="stale-reviewer",
    )
    confirmed = formation.plan_pending_lifecycle_transition(
        pending_uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="winning-reviewer",
    )
    committer.commit("u1", [confirmed])
    secondary_proposal = _proposal(
        episode,
        "regular-success-before-conflict",
        "PostgreSQL",
        "future_option",
        "exploratory",
    )
    secondary_pending = formation.plan_pending(
        secondary_proposal,
        archive=archive,
        episode=episode,
        reason="secondary_manual_review",
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id="regular-success-create",
    )
    secondary_pending_uri = str(secondary_pending.operations[0].target_uri)
    before_objects = {obj.uri: obj.to_dict() for obj in source.list_objects()}
    before_audit = (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8")
    before_diffs = {
        path.name: path.read_text(encoding="utf-8")
        for path in (tmp_path / "system" / "diffs").glob("*.json")
    }

    with pytest.raises(RevisionConflictError) as raised:
        committer.commit(
            "u1",
            [*canonical_operations, *secondary_pending.operations, stale_rejection],
        )

    assert raised.value.committed_diff is None
    canonical_ids = {operation.operation_id for operation in canonical_operations}
    secondary_add_id = secondary_pending.operations[0].operation_id
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot is None
    assert not claims
    assert CanonicalMemoryRepository(source).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    ).lifecycle_state == LifecycleState.CONFIRMED
    with pytest.raises(FileNotFoundError):
        source.read_object(secondary_pending_uri)
    assert secondary_pending_uri not in index.indexed_uris()
    assert {obj.uri: obj.to_dict() for obj in source.list_objects()} == before_objects
    assert (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8") == before_audit
    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in (tmp_path / "system" / "diffs").glob("*.json")
    } == before_diffs
    assert not any(
        (tmp_path / "system" / "transactions" / f"{operation.payload['idempotency_key']}.json").exists()
        for operation in canonical_operations
    )
    assert not committer.redo.pending_entries()

    planning_context = PlanningContext(
        planning_id="regular-conflict-replan",
        task_id=archive.task_id,
        archive_digest=archive.archive_digest,
        manifest_digest=archive.manifest_digest,
        episode_id=episode.episode_id,
        session_id=archive.session_id,
        tenant_id="t1",
        proposal_inputs=(),
        prefetch_snapshot=(),
        planned_against_revisions=(),
        staged_objects=(),
        scope_candidates=tuple(item.key for item in episode.legal_scope_candidates()),
        evidence_references=proposal.evidence_refs,
        operation_group_identity=f"commit_group_{archive.task_id}",
    )

    class LifecycleReplanner:
        operations = ()

        def replan(self, context, requested_archive):  # noqa: ANN001, ANN201, ARG002
            reject_primary = formation.plan_pending_lifecycle_transition(
                pending_uri,
                LifecycleState.REJECTED,
                tenant_id="t1",
                owner_user_id="u1",
                commit_group_id="fresh-reviewer",
                reason="review_reconciled_after_cas_conflict",
            )
            self.operations = (*canonical_operations, *secondary_pending.operations, reject_primary)
            return MemoryPlanningResult(self.operations, context)

    replanner = LifecycleReplanner()
    service = SessionCommitService(
        archive_store,
        queue,
        committer=committer,
        memory_planner=cast(MemoryCommitPlanner, replanner),
    )
    result = service._commit_memory_with_reconcile_retry(
        archive,
        [*canonical_operations, *secondary_pending.operations, stale_rejection],
        planning_context,
    )

    assert replanner.operations
    assert {item["operation_id"] for item in result["operations"]} == {
        *canonical_ids,
        secondary_add_id,
        replanner.operations[-1].operation_id,
    }
    assert result["pending_count"] == 1
    assert result["pending_persisted"] is True
    assert CanonicalMemoryRepository(source).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    ).lifecycle_state == LifecycleState.REJECTED
    assert CanonicalMemoryRepository(source).load_pending(
        secondary_pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    ).lifecycle_state == LifecycleState.PENDING
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot is not None
    assert {claim.canonical_value: claim.current.state for claim in claims} == {"sqlite": "ACTIVE"}
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    audited_operation_ids = [str(row.get("payload", {}).get("operation_id") or "") for row in audit_rows]
    actual_operation_ids = {
        *canonical_ids,
        secondary_add_id,
        replanner.operations[-1].operation_id,
    }
    assert all(audited_operation_ids.count(operation_id) == 1 for operation_id in actual_operation_ids)


def test_confirmed_pending_supplement_commits_revision_before_linked_resolution(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    source, index, _queue, relations, committer, initial_episode, scope = _setup(tmp_path)
    initial = _proposal(initial_episode, "supplement-base", "SQLite", "confirmation", "confirmed")
    identity, _, initial_plan = _plan(source, initial_episode, scope, initial)
    committer.commit(
        "u1",
        initial_plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=initial_episode.episode_id,
        ),
    )
    active = next(
        claim for claim in CanonicalMemoryRepository(source).load(identity)[1] if claim.current.state == "ACTIVE"
    )
    weak_archive = SessionArchive(
        user_id="u1",
        session_id="supplement-weak",
        archive_uri="memoryos://user/u1/sessions/history/supplement-weak",
        messages=[
            {
                "id": "weak-detail",
                "role": "user",
                "content": (
                    "For the primary storage backend SQLite, perhaps supplement the rationale: stable under load."
                ),
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id="supplement-weak-task",
    )
    weak_episode = _persisted_episode(tmp_path, weak_archive)
    weak = _supplement_proposal(
        weak_episode,
        "supplement-weak",
        active,
        speech_act=SpeechAct.PROPOSAL,
        commitment=Commitment.WEAK,
    )
    formation = CanonicalMemoryFormationService(source)
    pending = formation.plan(
        weak,
        archive=weak_archive,
        episode=weak_episode,
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id="supplement-weak-group",
    )
    assert pending.decision.value == "PENDING"
    assert pending.operations[0].payload["canonical_pending_proposal"] is True
    committer.commit("u1", list(pending.operations))
    pending_uri = str(pending.operations[0].target_uri)
    confirm_review = formation.plan_pending_lifecycle_transition(
        pending_uri,
        LifecycleState.CONFIRMED,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="supplement-human-review",
        reason="human_confirmed_with_followup_evidence",
    )
    committer.commit("u1", [confirm_review])

    confirmed_archive = SessionArchive(
        user_id="u1",
        session_id="supplement-confirmed",
        archive_uri="memoryos://user/u1/sessions/history/supplement-confirmed",
        messages=[
            {
                "id": "confirmed-detail",
                "role": "user",
                "content": (
                    "I confirm an additional supplement for the primary storage backend SQLite: "
                    "the rationale is stable under load."
                ),
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id="supplement-confirmed-task",
    )
    confirmed_episode = _persisted_episode(tmp_path, confirmed_archive)
    confirmed = _supplement_proposal(
        confirmed_episode,
        "supplement-confirmed",
        active,
        speech_act=SpeechAct.CONFIRMATION,
        commitment=Commitment.CONFIRMED,
    )
    rewritten = replace(
        confirmed,
        value_fields={"canonical_value": "SQLite", "rationale": "different reviewed content"},
    )
    with pytest.raises(ValueError, match="cannot rewrite pending identity or value fields"):
        formation.plan_confirmed_pending_resolution(
            pending_uri,
            rewritten,
            archive=confirmed_archive,
            episode=confirmed_episode,
            tenant_id="t1",
            owner_user_id="u1",
            commit_group_id="supplement-rewrite-attempt",
            retrieval_views=["project:memoryos:decisions"],
        )
    changed_relation = replace(
        confirmed,
        semantic=replace(confirmed.semantic, relation_to_existing=SemanticRelation.DUPLICATE),
    )
    with pytest.raises(ValueError, match="cannot change pending semantic relation"):
        formation.plan_confirmed_pending_resolution(
            pending_uri,
            changed_relation,
            archive=confirmed_archive,
            episode=confirmed_episode,
            tenant_id="t1",
            owner_user_id="u1",
            commit_group_id="supplement-relation-attempt",
            retrieval_views=["project:memoryos:decisions"],
        )
    changed_modal = replace(
        confirmed,
        semantic=replace(confirmed.semantic, modal_force=ModalForce.PREFER),
    )
    with pytest.raises(ValueError, match="cannot change pending proposition semantics"):
        formation.plan_confirmed_pending_resolution(
            pending_uri,
            changed_modal,
            archive=confirmed_archive,
            episode=confirmed_episode,
            tenant_id="t1",
            owner_user_id="u1",
            commit_group_id="supplement-modal-attempt",
            retrieval_views=["project:memoryos:decisions"],
        )
    resolved_plan = formation.plan_confirmed_pending_resolution(
        pending_uri,
        confirmed,
        archive=confirmed_archive,
        episode=confirmed_episode,
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="supplement-resolution-group",
        retrieval_views=["project:memoryos:decisions"],
    )
    resolution_operation = resolved_plan.operations[-1]
    assert resolution_operation.payload["pending_lifecycle_resolution"] is True
    assert resolution_operation.payload["canonical_pending_resolution"] is True
    assert resolution_operation.payload["canonical_memory"] is True
    assert resolution_operation.payload["transaction_id"] == resolved_plan.operations[0].payload["transaction_id"]
    assert resolution_operation.payload["resolved_claim_uris"] == [active.uri]
    original_write = source.write_object
    fail_resolution = True

    def write_with_resolution_failure(obj, content=""):  # noqa: ANN001, ANN202
        nonlocal fail_resolution
        if (
            fail_resolution
            and dict(obj.metadata or {}).get("canonical_kind") == "pending_proposal"
            and obj.lifecycle_state == LifecycleState.RESOLVED
        ):
            fail_resolution = False
            raise OSError("injected pending resolution write failure")
        return original_write(obj, content)

    monkeypatch.setattr(source, "write_object", write_with_resolution_failure)
    with pytest.raises(OSError, match="pending resolution write failure"):
        committer.commit("u1", list(resolved_plan.operations))
    rolled_back = CanonicalMemoryRepository(source).load(identity)[1]
    assert len(next(claim for claim in rolled_back if claim.claim_id == active.claim_id).revisions) == 1
    assert CanonicalMemoryRepository(source).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    ).lifecycle_state == LifecycleState.CONFIRMED

    monkeypatch.setattr(source, "write_object", original_write)
    committed = committer.commit("u1", list(resolved_plan.operations))
    assert {operation.operation_id for operation in committed.operations} == {
        operation.operation_id for operation in resolved_plan.operations
    }
    repeated = committer.commit("u1", list(resolved_plan.operations))
    assert {operation.operation_id for operation in repeated.operations} == {
        operation.operation_id for operation in resolved_plan.operations
    }

    updated = next(
        claim for claim in CanonicalMemoryRepository(source).load(identity)[1] if claim.claim_id == active.claim_id
    )
    assert len(updated.revisions) == 2
    assert updated.current.state == "ACTIVE"
    assert updated.current.qualifiers["display_fields"]["rationale"] == "stable under load"
    pending_record = CanonicalMemoryRepository(source).load_pending(
        pending_uri,
        tenant_id="t1",
        owner_user_id="u1",
    )
    assert pending_record.lifecycle_state == LifecycleState.RESOLVED
    assert pending_record.lifecycle_revision == 3
    assert pending_uri in index.indexed_uris()
    assert source.read_object(pending_uri).metadata["admission"]["decision"] == "pending"
    assert index.search(
        "stable under load",
        filters={"owner_user_id": "u1", "tenant_id": "t1"},
    ) == []
    assert resolution_operation.payload["resolution_idempotency_keys"]
    assert all(
        (tmp_path / "system" / "transactions" / f"{key}.json").exists()
        for key in resolution_operation.payload["resolution_idempotency_keys"]
    )
