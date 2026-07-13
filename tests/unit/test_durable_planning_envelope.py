from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.planners.memory_commit_planner import (
    MemoryCommitPlanner,
    MemoryExtractionBackendError,
)
from memoryos.contextdb.session.planning import ProposalPlanningInput
from memoryos.contextdb.session.planning_envelope import (
    PlanningEnvelopeIntegrityError,
    PlanningEnvelopeStore,
)
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.proposal import MemorySemanticProposal, SemanticRelation
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.salience_ledger import SalienceLedgerIntegrityError
from memoryos.memory.schema import MemoryTypeSchema
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeReadinessState
from memoryos.workers.memory_proposal_worker import MemoryProposalWorker
from memoryos.workers.session_commit_worker import SessionCommitWorker
from tests.unit.test_canonical_transaction_commit import (
    _persisted_episode,
    _plan,
    _proposal,
    _setup,
)


class _BlockingCountingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemorySemanticProposal]:
        del archive, schemas
        with self._lock:
            self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=10)
        return []


class _FixedExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self, proposals: Sequence[MemorySemanticProposal]) -> None:
        self.proposals = list(proposals)
        self.calls = 0
        self.extractor_version = type(self).__name__
        self.model_id = ""
        self.prompt_version = ""
        self.semantic_contract_version = ""

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemorySemanticProposal]:
        del archive, schemas
        self.calls += 1
        return list(self.proposals)


class _FailingCountingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, archive, schemas):  # noqa: ANN001, ANN201
        del archive, schemas
        self.calls += 1
        raise OSError("injected extraction outage")


def _salient_archive() -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="same-task",
        archive_uri="memoryos://user/u1/sessions/history/same-task",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "Remember this durable project rule for all future sessions.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id="same-planning-task",
        created_at="2026-07-12T00:00:00+00:00",
    )


def _persist_archive(root: Path, archive: SessionArchive) -> SessionArchive:
    SessionArchiveStore(root, tenant_id="t1").write_sync_archive(archive)
    return archive


def test_durable_planner_rejects_uncommitted_archive_before_model_call(tmp_path: Path) -> None:
    extractor = _FixedExtractor(())
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )

    with pytest.raises(
        PlanningEnvelopeIntegrityError,
        match="integrity-checked immutable session archive",
    ):
        planner.plan(_salient_archive())

    assert extractor.calls == 0


def test_durable_planner_rejects_historical_archive_manifest_after_head_advance(
    tmp_path: Path,
) -> None:
    extractor = _FixedExtractor([])
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive_store = SessionArchiveStore(tmp_path, tenant_id="t1")
    first = _salient_archive()
    archive_store.write_sync_archive(first)
    historical = deepcopy(first)
    advanced = replace(
        first,
        messages=[
            {
                "id": "m2",
                "role": "user",
                "content": "Remember this newer durable project rule for all future sessions.",
            }
        ],
    )
    archive_store.write_sync_archive(advanced)

    with pytest.raises(
        PlanningEnvelopeIntegrityError,
        match="differs from its immutable session archive|digest binding is inconsistent",
    ):
        planner.plan(historical)

    assert extractor.calls == 0
    assert not list((tmp_path / "system" / "planning-envelopes").glob("*.json"))


def test_runtime_bound_custom_planner_also_requires_committed_archive_before_model_call(
    tmp_path: Path,
) -> None:
    extractor = _FixedExtractor(())
    planner = MemoryCommitPlanner(extractor=extractor)
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()

    planner.bind_runtime_stores(
        source,
        index,
        None,
        root=tmp_path,
        tenant_id="t1",
    )

    with pytest.raises(
        PlanningEnvelopeIntegrityError,
        match="integrity-checked immutable session archive",
    ):
        planner.plan(_salient_archive())

    assert planner.archive_store is not None
    assert extractor.calls == 0
    assert not list((tmp_path / "tenants" / "t1" / "system" / "planning-envelopes").glob("*.json"))


def test_concurrent_same_task_calls_model_once_and_return_same_envelope(tmp_path: Path) -> None:
    extractor = _BlockingCountingExtractor()
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(planner.plan, archive)
        assert extractor.entered.wait(timeout=10)
        second = pool.submit(planner.plan, archive)
        extractor.release.set()
        first_result = first.result(timeout=20)
        second_result = second.result(timeout=20)

    assert extractor.calls == 1
    assert first_result.context.planning_digest == second_result.context.planning_digest
    assert first_result.context.proposal_set_digest == second_result.context.proposal_set_digest
    assert first_result.operations == second_result.operations == ()


def test_planning_envelope_tamper_and_same_task_different_proposal_set_are_rejected(
    tmp_path: Path,
) -> None:
    source, index, _queue, _relations, _committer, episode, _scope = _setup(tmp_path)
    archive = SessionArchiveStore(tmp_path, tenant_id="t1").read_archive(episode.source_uris[0])
    proposal = _proposal(episode, "planning-original", "SQLite", "confirmation", "confirmed")
    planner = MemoryCommitPlanner(
        extractor=_FixedExtractor([proposal]),
        source_store=source,
        index_store=index,
    )
    result = planner.plan(archive)
    assert planner.planning_store is not None
    store = planner.planning_store
    different = replace(proposal, proposal_id="planning-different")
    changed = replace(
        result.context,
        proposal_inputs=(ProposalPlanningInput(different),),
        proposal_set_digest=canonical_digest([different.to_dict()]),
    )
    with pytest.raises(PlanningEnvelopeIntegrityError, match="another immutable proposal set"):
        store.create(changed, archive_uri=archive.archive_uri)

    path = store.path(archive.task_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["egress_decision"] = "DENY"
    atomic_write_json(path, payload, artifact_root=store.artifact_root)
    with pytest.raises(PlanningEnvelopeIntegrityError, match="digest is corrupt"):
        store.load(archive.task_id)


def test_canonical_commit_requires_exact_proposal_proof_from_planning_envelope(tmp_path: Path) -> None:
    source, index, _queue, _relations, committer, _episode, scope = _setup(tmp_path)
    episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="planning-proof-binding",
            archive_uri="memoryos://user/u1/sessions/history/planning-proof-binding",
            messages=[
                {
                    "id": "remember-proof",
                    "role": "user",
                    "content": "Remember that SQLite is the confirmed primary storage backend.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id="planning-proof-binding-task",
        ),
    )
    archive = SessionArchiveStore(tmp_path, tenant_id="t1").read_archive(episode.source_uris[0])
    proposal = _proposal(episode, "planning-proof-binding", "SQLite", "confirmation", "confirmed")
    planner = MemoryCommitPlanner(
        extractor=_FixedExtractor([proposal]),
        source_store=source,
        index_store=index,
    )
    result = planner.plan(archive)
    normalized = result.context.proposal_inputs[0].proposal
    _identity, _transition, transaction_plan = _plan(
        source,
        episode,
        scope,
        normalized,
        commit_group_id=result.context.operation_group_identity,
    )
    operations = transaction_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    for operation in operations:
        operation.payload["planning_task_id"] = result.context.task_id
        operation.payload["planning_digest"] = result.context.planning_digest
    committer._ensure_canonical_planning_digest(operations, publish=False)
    assert planner.planning_store is not None
    envelope_path = planner.planning_store.path(result.context.task_id)
    anchor_path = planner.planning_store.anchor_path(result.context.task_id)
    envelope_bytes = envelope_path.read_bytes()
    anchor_bytes = anchor_path.read_bytes()
    envelope_path.unlink()
    anchor_path.unlink()
    envelope_path.symlink_to(tmp_path / "missing-planning-envelope.json")
    anchor_path.symlink_to(tmp_path / "missing-planning-anchor.json")

    detached_operations = deepcopy(operations)
    for operation in detached_operations:
        operation.payload.pop("planning_digest", None)
    with pytest.raises(ValueError, match="planning envelope|symbolic link"):
        committer._ensure_canonical_planning_digest(detached_operations, publish=False)

    envelope_path.unlink()
    anchor_path.unlink()
    envelope_path.write_bytes(envelope_bytes)
    anchor_path.write_bytes(anchor_bytes)
    operations = deepcopy(operations)
    for operation in operations:
        operation.payload["proposal_proofs"][0]["proposal_id"] = "forged-envelope-proposal"

    with pytest.raises(ValueError, match="detached from its durable planning envelope"):
        committer._ensure_canonical_planning_digest(operations, publish=False)


def test_second_same_slot_proposal_reads_request_local_staged_revision(tmp_path: Path) -> None:
    source, index, _queue, relations, _committer, _episode, scope = _setup(tmp_path)
    episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="staged-same-slot",
            archive_uri="memoryos://user/u1/sessions/history/staged-same-slot",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": (
                        "I confirm SQLite is the primary storage backend because its rationale is stable under load."
                    ),
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id="staged-same-slot-task",
            created_at="2026-07-12T00:00:00+00:00",
        ),
    )
    archive = SessionArchiveStore(tmp_path, tenant_id="t1").read_archive(episode.source_uris[0])
    initial = _proposal(episode, "staged-initial", "SQLite", "confirmation", "confirmed")
    initial = replace(
        initial,
        semantic=replace(initial.semantic, relation_to_existing=SemanticRelation.UNRELATED),
    )
    second = replace(initial, proposal_id="staged-second-same-fact")
    extractor = _FixedExtractor([initial, second])
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=source,
        index_store=index,
        relation_store=relations,
    )

    result = planner.plan(archive)

    claim_operations = [
        operation
        for operation in result.operations
        if dict(operation.payload["context_object"]["metadata"]).get("canonical_kind") == "claim"
    ]
    assert extractor.calls == 1
    assert result.context.proposal_outcomes[1].decision == "ACCEPT_FOR_RECONCILE"
    # Without request-local overlay the second proposal would plan another
    # revision-1 ADD. Seeing the staged ACTIVE Claim makes it a deterministic
    # duplicate/no-op instead.
    assert len(claim_operations) == 1
    assert claim_operations[0].payload["expected_revision"] == 0
    assert claim_operations[0].payload["context_object"]["metadata"]["revision"] == 1
    assert result.context.staged_objects

    restarted = MemoryCommitPlanner(
        extractor=_FixedExtractor([]),
        source_store=source,
        index_store=index,
        relation_store=relations,
    )
    replay = restarted.plan(archive)
    assert replay.context.proposal_set_digest == result.context.proposal_set_digest
    replay_claims = [
        operation
        for operation in replay.operations
        if dict(operation.payload["context_object"]["metadata"]).get("canonical_kind") == "claim"
    ]
    assert len(replay_claims) == 1
    assert replay_claims[0].payload["expected_revision"] == 0


def test_second_identical_pending_proposal_reads_request_local_staging(
    tmp_path: Path,
) -> None:
    source, index, _queue, relations, _committer, _episode, scope = _setup(tmp_path)
    episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="staged-pending",
            archive_uri="memoryos://user/u1/sessions/history/staged-pending",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": "Remember this durable project decision, but keep it pending for review.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id="staged-pending-task",
            created_at="2026-07-12T00:00:00+00:00",
        ),
    )
    archive = SessionArchiveStore(tmp_path, tenant_id="t1").read_archive(episode.source_uris[0])
    first = _proposal(episode, "staged-pending-one", "SQLite", "confirmation", "confirmed")
    first = replace(
        first,
        metadata={**dict(first.metadata), "fallback_pending_only": True},
    )
    second = replace(first, proposal_id="staged-pending-two")
    extractor = _FixedExtractor([first, second])
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=source,
        index_store=index,
        relation_store=relations,
    )

    result = planner.plan(archive)

    pending_operations = [
        operation for operation in result.operations if operation.payload.get("canonical_pending_proposal") is True
    ]
    assert extractor.calls == 1
    assert [item.decision for item in result.context.proposal_outcomes] == ["PENDING", "PENDING"]
    assert len(pending_operations) == 1
    assert len(result.context.staged_objects) == 1


def test_envelope_path_is_tenant_scoped(tmp_path: Path) -> None:
    a = PlanningEnvelopeStore(tmp_path, tenant_id="a")
    b = PlanningEnvelopeStore(tmp_path, tenant_id="b")
    assert a.path("task") != b.path("task")


def test_non_default_tenant_receipt_history_validates_its_planning_envelope(
    tmp_path: Path,
) -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="tenant-history",
        archive_uri="memoryos://user/u1/sessions/history/tenant-history",
        messages=[
            {
                "id": "tenant-history-message",
                "role": "user",
                "content": "I confirm SQLite is the primary storage backend.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id="tenant-history-task",
        created_at="2026-07-12T00:00:00+00:00",
    )
    episode = _persisted_episode(tmp_path, archive)
    persisted = SessionArchiveStore(tmp_path, tenant_id="t1").read_archive(archive.archive_uri)
    proposal = _proposal(
        episode,
        "tenant-history-proposal",
        "SQLite",
        "confirmation",
        "confirmed",
    )
    runtime = build_runtime_container(
        RuntimeConfig(
            root=str(tmp_path),
            tenant_id="t1",
            memory_extractor=_FixedExtractor([proposal]),
        )
    )

    committed = runtime.session_commit_service.async_commit(persisted)

    assert committed.canonical_committed is True
    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons


def test_deleted_planning_envelope_is_detected_by_immutable_anchor_at_startup(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    extractor = _FixedExtractor([])
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=source,
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    planner.plan(archive)
    assert planner.planning_store is not None
    assert planner.planning_store.anchor_path(archive.task_id).exists()
    planner.planning_store.path(archive.task_id).unlink()
    with pytest.raises(PlanningEnvelopeIntegrityError, match="missing after its immutable identity anchor"):
        planner.plan(archive)
    assert extractor.calls == 1

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert any("planning envelope" in reason for reason in restarted.readiness.reasons)


def test_deleted_planning_anchor_is_not_silently_recreated_at_startup(tmp_path: Path) -> None:
    planner = MemoryCommitPlanner(
        extractor=_FixedExtractor([]),
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    planner.plan(archive)
    assert planner.planning_store is not None
    anchor = planner.planning_store.anchor_path(archive.task_id)
    anchor.unlink()

    with pytest.raises(PlanningEnvelopeIntegrityError, match="anchor is missing"):
        planner.planning_store.validate_all()
    assert not anchor.exists()

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert not anchor.exists()


def test_deleted_salience_anchor_forces_startup_not_ready(tmp_path: Path) -> None:
    planner = MemoryCommitPlanner(
        extractor=_FixedExtractor([]),
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    planner.plan(archive)
    assert planner.salience_ledger is not None
    anchor = planner.salience_ledger.anchor_path(archive.task_id)
    anchor.unlink()

    with pytest.raises(SalienceLedgerIntegrityError, match="sets disagree"):
        planner.salience_ledger.validate_all()
    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY


def test_malformed_normalized_proposal_raises_typed_integrity_error(tmp_path: Path) -> None:
    planner = MemoryCommitPlanner(
        extractor=_FixedExtractor([]),
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    planner.plan(archive)
    assert planner.planning_store is not None
    payload = planner.planning_store.load_payload(archive.task_id)
    payload["proposal_inputs"] = [{"proposal": "not-an-object"}]
    core = {key: value for key, value in payload.items() if key != "envelope_digest"}
    payload["envelope_digest"] = canonical_digest(core)
    atomic_write_json(
        planner.planning_store.path(archive.task_id),
        payload,
        artifact_root=planner.planning_store.artifact_root,
    )

    with pytest.raises(PlanningEnvelopeIntegrityError, match="invalid normalized proposal"):
        planner.planning_store.load_payload(archive.task_id)


def test_public_repository_cannot_accept_caller_injected_staged_state(tmp_path: Path) -> None:
    staged = ContextObject(
        uri="memoryos://user/u1/memories/canonical/slots/injected",
        context_type=ContextType.MEMORY,
        title="injected",
        owner_user_id="u1",
        tenant_id="t1",
        metadata={"canonical_kind": "slot"},
    )
    with pytest.raises(PermissionError, match="internal to one PlanningContext"):
        CanonicalMemoryRepository(
            FileSystemSourceStore(tmp_path, tenant_id="t1"),
            _request_staged_objects={staged.uri: staged},
        )


def test_durable_budget_limits_distinct_tasks_and_retry_does_not_consume_twice(tmp_path: Path) -> None:
    extractor = _FixedExtractor([])
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    results = []
    for index in range(9):
        archive = replace(
            _salient_archive(),
            session_id=f"budget-{index}",
            archive_uri=f"memoryos://user/u1/sessions/history/budget-{index}",
            task_id=f"budget-task-{index}",
            messages=[
                {
                    "id": f"m-{index}",
                    "role": "user",
                    "content": f"Remember this durable project rule number {index} for future sessions.",
                }
            ],
        )
        _persist_archive(tmp_path, archive)
        results.append(planner.plan(archive))

    assert extractor.calls == 8
    assert results[-1].context.proposal_outcomes[0].reason == "episode_budget_exhausted"
    repeated = planner.plan(
        replace(
            _salient_archive(),
            session_id="budget-8",
            archive_uri="memoryos://user/u1/sessions/history/budget-8",
            task_id="budget-task-8",
            messages=[
                {
                    "id": "m-8",
                    "role": "user",
                    "content": "Remember this durable project rule number 8 for future sessions.",
                }
            ],
        )
    )
    assert extractor.calls == 8
    assert repeated.context.planning_digest == results[-1].context.planning_digest


def test_existing_salient_reservation_without_envelope_never_recalls_model(tmp_path: Path) -> None:
    extractor = _FailingCountingExtractor()
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())

    with pytest.raises(MemoryExtractionBackendError, match="memory extraction backend failed"):
        planner.plan(archive)
    assert extractor.calls == 3
    with pytest.raises(PlanningEnvelopeIntegrityError, match="re-extraction requires a new commit group"):
        planner.plan(archive)
    assert extractor.calls == 3


def test_extraction_retry_exhaustion_never_reinvokes_model_for_same_commit_group(
    tmp_path: Path,
) -> None:
    extractor = _FailingCountingExtractor()
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1", memory_extractor=extractor))
    archive = _salient_archive()

    result = runtime.session_commit_service.async_commit(archive)

    status = runtime.session_commit_service.commit_group_store.load(result.commit_group_id)
    assert extractor.calls == 3
    assert result.state.value == "DEAD_LETTER"
    assert status is not None
    assert status.canonical_status == "dead_letter"
    assert status.canonical_retryable is False
    assert status.canonical_phase == "salience_reserved"
    assert status.canonical_attempt_id == ""
    assert status.canonical_owner_pid == 0
    assert status.canonical_lease_expires_at == ""
    assert runtime.queue_store.get(f"memory_proposal_{archive.task_id}") is None

    second = runtime.session_commit_service.async_commit(archive)
    third = runtime.session_commit_service.async_commit(archive)
    terminal = runtime.session_commit_service.commit_group_store.load(result.commit_group_id)
    assert second.state.value == "DEAD_LETTER"
    assert third.state.value == "DEAD_LETTER"
    assert extractor.calls == 3
    assert terminal is not None
    assert terminal.canonical_status == "dead_letter"
    assert terminal.canonical_attempt_id == ""
    assert terminal.canonical_owner_pid == 0
    assert terminal.canonical_lease_expires_at == ""

    settled = MemoryProposalWorker(runtime.session_commit_service).process_pending(max_retries=3)
    queued = runtime.queue_store.get(f"memory_proposal_{archive.task_id}")
    assert settled == {"claimed": 0, "committed": 0, "failed": 0, "dead_letter": 0}
    assert queued is None
    assert extractor.calls == 3


def test_session_worker_dead_letters_planning_integrity_failure_without_retry_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime_container(
        RuntimeConfig(root=str(tmp_path), tenant_id="t1", memory_extractor=_FixedExtractor([]))
    )
    archive = _salient_archive()
    runtime.session_commit_service.sync_archive(archive)

    def fail_closed(_archive: SessionArchive):  # noqa: ANN202
        raise PlanningEnvelopeIntegrityError("corrupt durable planning envelope")

    monkeypatch.setattr(runtime.session_commit_service, "async_commit", fail_closed)
    result = SessionCommitWorker(runtime.session_commit_service).process_pending(max_retries=3)

    assert result["failed"] == 1
    assert result["dead_letter"] == 1
    job = runtime.queue_store.get(archive.task_id)
    assert job is not None
    assert job.status == "dead_letter"
    assert job.retry_count == 1


class _UnexpectedSessionCommitFailure(Exception):
    pass


def test_unexpected_session_commit_failure_releases_canonical_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime_container(
        RuntimeConfig(root=str(tmp_path), tenant_id="t1", memory_extractor=_FixedExtractor([]))
    )
    archive = _salient_archive()

    def fail_unexpectedly(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise _UnexpectedSessionCommitFailure("unexpected internal failure")

    monkeypatch.setattr(
        runtime.session_commit_service.memory_planner,
        "plan_with_progress",
        fail_unexpectedly,
    )

    with pytest.raises(_UnexpectedSessionCommitFailure, match="unexpected internal failure"):
        runtime.session_commit_service.async_commit(archive)

    group = runtime.session_commit_service.commit_group_store.load(f"commit_group_{archive.task_id}")
    assert group is not None
    assert group.canonical_status == "dead_letter"
    assert group.canonical_retryable is False
    assert group.canonical_attempt_id == ""
    assert group.canonical_owner_pid == 0
    assert group.canonical_lease_expires_at == ""


def test_session_worker_terminally_settles_unexpected_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime_container(
        RuntimeConfig(root=str(tmp_path), tenant_id="t1", memory_extractor=_FixedExtractor([]))
    )
    archive = _salient_archive()
    runtime.session_commit_service.sync_archive(archive)

    def fail_unexpectedly(_archive: SessionArchive):  # noqa: ANN202
        raise _UnexpectedSessionCommitFailure("unexpected worker failure")

    monkeypatch.setattr(runtime.session_commit_service, "async_commit", fail_unexpectedly)

    result = SessionCommitWorker(runtime.session_commit_service).process_pending(max_retries=3)

    assert result["failed"] == 1
    assert result["dead_letter"] == 1
    job = runtime.queue_store.get(archive.task_id)
    assert job is not None
    assert job.status == "dead_letter"
    assert job.retry_count == 1


def test_memory_proposal_worker_terminally_settles_unexpected_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime_container(
        RuntimeConfig(root=str(tmp_path), tenant_id="t1", memory_extractor=_FixedExtractor([]))
    )
    archive = _salient_archive()
    runtime.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
    job_id = f"memory_proposal_unexpected_{archive.task_id}"
    runtime.queue_store.enqueue(
        QueueJob(
            job_id=job_id,
            queue_name="memory_proposal",
            action="extract_memory_proposals",
            target_uri=archive.archive_uri,
            payload={
                "tenant_id": "t1",
                "manifest_digest": archive.manifest_digest,
            },
        )
    )

    def fail_unexpectedly(_archive: SessionArchive):  # noqa: ANN202
        raise _UnexpectedSessionCommitFailure("unexpected proposal worker failure")

    monkeypatch.setattr(runtime.session_commit_service, "async_commit", fail_unexpectedly)

    result = MemoryProposalWorker(runtime.session_commit_service).process_pending(max_retries=3)

    assert result["failed"] == 1
    assert result["dead_letter"] == 1
    job = runtime.queue_store.get(job_id)
    assert job is not None
    assert job.status == "dead_letter"
    assert job.retry_count == 1


@pytest.mark.parametrize(
    ("worker_type", "queue_name"),
    (
        (MemoryProposalWorker, "memory_proposal"),
        (SessionCommitWorker, "session_commit"),
    ),
)
@pytest.mark.parametrize("declared_tenant", (None, "other-tenant"))
def test_tenant_bound_workers_reject_untrusted_job_tenant_before_archive_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_type: type[MemoryProposalWorker] | type[SessionCommitWorker],
    queue_name: str,
    declared_tenant: str | None,
) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    payload = {"manifest_digest": "untrusted-manifest"}
    if declared_tenant is not None:
        payload["tenant_id"] = declared_tenant
    job_id = f"tenant-boundary-{queue_name}-{declared_tenant or 'missing'}"
    runtime.queue_store.enqueue(
        QueueJob(
            job_id=job_id,
            queue_name=queue_name,
            action="untrusted_tenant_job",
            target_uri="memoryos://user/u1/sessions/history/untrusted-tenant-job",
            payload=payload,
        )
    )
    archive_read_attempted = False

    def reject_archive_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal archive_read_attempted
        archive_read_attempted = True
        raise AssertionError("an untrusted job tenant reached the archive store")

    monkeypatch.setattr(runtime.session_archive_store, "read_archive", reject_archive_read)

    result = worker_type(runtime.session_commit_service).process_pending(max_retries=3)

    assert archive_read_attempted is False
    assert result["claimed"] == result["failed"] == result["dead_letter"] == 1
    settled = runtime.queue_store.get(job_id)
    assert settled is not None
    assert settled.status == "dead_letter"
    assert settled.retry_count == 1


def test_commit_group_detects_dual_deletion_of_planning_and_salience_artifacts(tmp_path: Path) -> None:
    runtime = build_runtime_container(
        RuntimeConfig(
            root=str(tmp_path),
            tenant_id="t1",
            memory_extractor=_FixedExtractor([]),
        )
    )
    archive = _salient_archive()
    result = runtime.session_commit_service.async_commit(archive)
    assert result.canonical_committed
    planner = runtime.session_commit_service.memory_planner
    assert planner.planning_store is not None and planner.salience_ledger is not None
    planner.planning_store.path(archive.task_id).unlink()
    planner.planning_store.anchor_path(archive.task_id).unlink()
    planner.salience_ledger.path(archive.task_id).unlink()
    planner.salience_ledger.anchor_path(archive.task_id).unlink()

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "salience reservation" in " ".join(restarted.readiness.reasons).casefold()


def test_envelope_persists_only_digest_snapshots_and_explicit_empty_extractor_versions(tmp_path: Path) -> None:
    planner = MemoryCommitPlanner(
        extractor=_FixedExtractor([]),
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    result = planner.plan(archive)
    assert planner.planning_store is not None
    payload = planner.planning_store.load_payload(archive.task_id)

    assert payload["user_id"] == archive.user_id
    assert payload["extractor_version"] == "_FixedExtractor"
    assert payload["model_id"] == payload["prompt_version"] == payload["semantic_contract_version"] == ""
    assert "payload_json" not in json.dumps(payload["prefetch_snapshot"] + payload["staged_objects"])
    assert result.context.salience_reservation_digest == payload["salience_reservation_digest"]


def test_durable_replan_preserves_original_extractor_identity(tmp_path: Path) -> None:
    original = _FixedExtractor([])
    original.extractor_version = "extractor-original"
    original.model_id = "model-original"
    original.prompt_version = "prompt-original"
    original.semantic_contract_version = "contract-original"
    planner = MemoryCommitPlanner(
        extractor=original,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = _persist_archive(tmp_path, _salient_archive())
    first = planner.plan(archive)
    replacement = _FixedExtractor([])
    replacement.extractor_version = "extractor-replacement"
    replacement.model_id = "model-replacement"
    replacement.prompt_version = "prompt-replacement"
    replacement.semantic_contract_version = "contract-replacement"
    planner.extractor = replacement

    replay = planner.plan(archive)

    assert replacement.calls == 0
    assert (
        replay.context.extractor_version,
        replay.context.model_id,
        replay.context.prompt_version,
        replay.context.semantic_contract_version,
    ) == (
        first.context.extractor_version,
        first.context.model_id,
        first.context.prompt_version,
        first.context.semantic_contract_version,
    )
