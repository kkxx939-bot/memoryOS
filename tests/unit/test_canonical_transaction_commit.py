from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

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
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.memory.canonical import (
    CanonicalMemoryProjector,
    CanonicalMemoryRepository,
    Commitment,
    EpistemicStatus,
    EvidenceRef,
    MemoryProjectionWorker,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemoryTransactionPlanner,
    MemoryTransitionPolicy,
    RevisionConflictError,
    ScopeSelector,
    SemanticAssessment,
    SessionArchiveEpisodeAdapter,
    SpeechAct,
    StableMemoryIdentityResolver,
    VisibilityPolicy,
    bind_field_evidence,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler
from memoryos.operations.commit.operation_committer import OperationCommitter


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
    evidence_refs = (EvidenceRef.from_event(episode.events[0], source_uri=episode.source_uris[0]),)
    return MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(speech, commitment, "current", "alternative"),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=bind_field_evidence(identity_fields, value_fields, evidence_refs),
            confidence=0.95,
            extractor_version="fake",
        )
    )


def _plan(source, episode, scope, proposal):  # noqa: ANN001, ANN202
    identity = StableMemoryIdentityResolver().resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    reconciled = MemorySemanticReconciler().reconcile(proposal, identity, slot=slot, claims=claims)
    transition = MemoryTransitionPolicy().apply(proposal, identity, reconciled)
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

    postgres = _proposal(episode, "p-postgres", "PostgreSQL", "future_option", "exploratory")
    _, _, second_plan = _plan(source, episode, scope, postgres)
    committer.commit(
        "u1",
        second_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    _, claims = CanonicalMemoryRepository(source).load(identity)
    assert {claim.canonical_value: claim.current.state for claim in claims} == {
        "sqlite": "ACTIVE",
        "postgresql": "PROPOSED",
    }

    confirmed = replace(
        postgres,
        proposal_id="p-confirm",
        semantic=replace(
            postgres.semantic,
            speech_act=SpeechAct.CONFIRMATION,
            commitment=Commitment.CONFIRMED,
        ),
    )
    _, transition, third_plan = _plan(source, episode, scope, confirmed)
    assert len(third_plan.operations) == 3, "slot and both claim changes must share one transaction"
    committer.commit(
        "u1",
        third_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot and slot.revision == transition.slot.revision
    assert {claim.canonical_value: claim.current.state for claim in claims} == {
        "sqlite": "SUPERSEDED",
        "postgresql": "ACTIVE",
    }
    assert len([claim for claim in claims if claim.current.state == "ACTIVE"]) == 1
    assert len(list((tmp_path / "system" / "outbox").glob("*.json"))) == 3
    assert relations.relations_of(identity.slot_uri, tenant_id="t1", owner_user_id="u1")


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
    archive = SessionArchive(
        user_id="u1",
        session_id=episode.episode_id,
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"id": "m1", "role": "user", "content": episode.events[0].text()}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    archive_store = SessionArchiveStore(tmp_path)
    directory = archive_store.write_sync_archive(archive)
    proposal = _proposal(episode, "p-evidence", "SQLite", "confirmation", "confirmed")
    proposal = replace(
        proposal,
        evidence_refs=(EvidenceRef.from_event(episode.events[0], source_uri=archive.archive_uri),),
    )
    _identity, _, plan = _plan(source, episode, scope, proposal)
    (directory / "messages.jsonl").write_text(
        json.dumps({"id": "m1", "role": "user", "content": "tampered"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash no longer matches"):
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
        )
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
        )
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
    assert slot is not None and {claim.canonical_value: claim.current.state for claim in claims} == {
        "sqlite": "ACTIVE"
    }
    outbox = next((tmp_path / "system" / "outbox").glob("*.json"))
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "committed"


def test_session_commit_revision_conflict_rereads_and_reconciles_same_proposal(tmp_path) -> None:  # noqa: ANN001
    source, index, queue, relations, committer, sqlite_episode, scope = _setup(tmp_path)
    postgres_archive = SessionArchive(
        user_id="u1",
        session_id="s2",
        archive_uri="memoryos://user/u1/sessions/history/s2",
        messages=[
            {"id": "m1", "role": "user", "content": "PostgreSQL is a future primary storage backend option."}
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    postgres_episode = _persisted_episode(tmp_path, postgres_archive)
    postgres = _proposal(postgres_episode, "p-postgres", "PostgreSQL", "future_option", "exploratory")
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
    planner.last_canonical_inputs = [(postgres, [])]
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        queue,
        committer=committer,
        memory_planner=planner,
    )
    service._commit_memory_with_reconcile_retry(postgres_archive, stale_operations)
    _slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert {claim.canonical_value: claim.current.state for claim in claims} == {
        "sqlite": "ACTIVE",
        "postgresql": "PROPOSED",
    }


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
        )
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
    postgres = _proposal(episode, "postgres-option", "PostgreSQL", "future_option", "exploratory")
    _, _, option_plan = _plan(source, episode, scope, postgres)
    committer.commit(
        "u1",
        option_plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    confirmed = replace(
        postgres,
        proposal_id="postgres-confirm",
        semantic=replace(postgres.semantic, speech_act=SpeechAct.CONFIRMATION, commitment=Commitment.CONFIRMED),
    )
    _, _, switch_plan = _plan(source, episode, scope, confirmed)
    switch_operations = switch_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    source.crash_at = source.write_count + 2
    with pytest.raises(SystemExit, match="simulated switch crash"):
        committer.commit("u1", switch_operations)

    before_recovery = CanonicalMemoryRepository(source).load(identity)[1]
    assert {claim.canonical_value: claim.current.state for claim in before_recovery} == {
        "sqlite": "ACTIVE",
        "postgresql": "PROPOSED",
    }
    RecoveryService(committer.redo, committer).recover("u1")
    after_recovery = CanonicalMemoryRepository(source).load(identity)[1]
    assert {claim.canonical_value: claim.current.state for claim in after_recovery} == {
        "sqlite": "SUPERSEDED",
        "postgresql": "ACTIVE",
    }
