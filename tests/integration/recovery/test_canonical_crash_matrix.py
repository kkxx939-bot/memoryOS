from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.memory.canonical import CanonicalMemoryRepository
from memoryos.memory.canonical.current_head import head_set_path, load_current_head
from memoryos.memory.canonical.history import validate_canonical_receipt_history
from memoryos.memory.canonical.review_command import PendingReviewCommandStore
from memoryos.memory.extraction import RuleFallbackExtractor
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeReadinessState
from tests.support.canonical_transactions import (
    _artifact_root,
    _persisted_episode,
    _plan,
    _proposal,
    _replacement_proposal,
    _reviewed_resolution_plan,
    _setup,
)

CRASH_STAGES = (
    "before_redo",
    "after_redo_begin",
    "after_source_effect",
    "after_relation_effect",
    "after_audit",
    "after_diff",
    "before_receipt",
    "after_receipt",
    "before_current_head",
    "after_current_head",
    "after_projection_enqueue",
    "after_committed_outbox",
    "before_redo_cleanup",
)


class _CrashOnce:
    def __init__(self, target: str) -> None:
        self.target = target
        self.crashed = False

    def __call__(self, stage: str, _transaction_id: str) -> None:
        if stage == self.target and not self.crashed:
            self.crashed = True
            raise SystemExit(f"process crash at {stage}")


def _crash_after_commit_group_effect(root: str) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=root, tenant_id="t1"))
    runtime.session_commit_service.memory_planner = MemoryCommitPlanner(
        extractor=RuleFallbackExtractor(),
        source_store=runtime.source_store,
        index_store=runtime.index_store,
        relation_store=runtime.relation_store,
    )

    def crash(stage: str, _group_id: str) -> None:
        if stage == "after_commit_group_effect_record":
            raise SystemExit(19)

    runtime.session_commit_service.commit_group_store.test_hook = crash
    archive = SessionArchive(
        user_id="u1",
        session_id="commit-group-crash",
        archive_uri="memoryos://user/u1/sessions/history/commit-group-crash",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "Project rule: MemoryOS must preserve crash-consistent commit groups.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id="commit-group-crash-task",
        created_at="2026-07-12T03:00:00Z",
    )
    runtime.session_commit_service.async_commit(archive)


@pytest.mark.parametrize("crash_stage", CRASH_STAGES)
def test_canonical_transaction_recovers_at_every_durable_publication_stage(
    tmp_path: Path,
    crash_stage: str,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert runtime.readiness.state == RuntimeReadinessState.READY
    proposal = _proposal(episode, f"crash-{crash_stage}", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    transaction_id = str(operations[0].payload["transaction_id"])
    idempotency_key = str(operations[0].payload["idempotency_key"])
    runtime.committer.test_hook = _CrashOnce(crash_stage)

    with pytest.raises(SystemExit, match=crash_stage):
        runtime.committer.commit("u1", operations)

    recovered = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    repository = CanonicalMemoryRepository(recovered.source_store, recovered.relation_store)
    if crash_stage == "before_redo":
        assert repository.load(identity) == (None, ())
        recovered.committer.commit("u1", operations)
    slot, claims = repository.load(identity)
    assert slot is not None
    assert len(claims) == 1
    assert claims[0].latest_revision.value_fields["canonical_value"] == "SQLite"

    artifact_root = _artifact_root(tmp_path)
    receipt_path = artifact_root / "system" / "transactions" / f"{idempotency_key}.json"
    assert receipt_path.exists()
    head, receipt, _snapshot = load_current_head(artifact_root, identity.slot_uri)
    assert head["current_transaction_id"] == transaction_id
    assert head["receipt_digest"] == receipt["receipt_digest"]
    assert (artifact_root / "system" / "outbox" / f"{transaction_id}.json").exists()
    assert not list((artifact_root / "system" / "redo").glob("*.json"))
    history = validate_canonical_receipt_history(artifact_root, tenant_id="t1")
    assert history["receipts"] == 1
    assert recovered.queue_store.stats().get("dead_letter", 0) == 0
    assert recovered.queue_store.stats().get("quarantine", 0) == 0

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons
    repeated_slot, repeated_claims = CanonicalMemoryRepository(
        restarted.source_store,
        restarted.relation_store,
    ).load(identity)
    assert repeated_slot == slot
    assert repeated_claims == claims
    assert not list((artifact_root / "system" / "redo").glob("*.json"))


def test_after_current_head_crash_deleted_first_head_fails_closed(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    proposal = _proposal(episode, "first-head-published", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    runtime.committer.test_hook = _CrashOnce("after_current_head")

    with pytest.raises(SystemExit, match="after_current_head"):
        runtime.committer.commit("u1", operations)

    assert {entry.phase for entry in runtime.committer.redo.pending_entries()} == {"head_published"}
    head_set_path(_artifact_root(tmp_path), identity.slot_uri).unlink()

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "head-published" in " ".join(restarted.readiness.reasons)


def test_idempotent_canonical_replay_cannot_recreate_a_deleted_committed_head(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    proposal = _proposal(episode, "deleted-head-replay", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    runtime.committer.commit("u1", operations)
    head_path = head_set_path(_artifact_root(tmp_path), identity.slot_uri)
    head_path.unlink()

    with pytest.raises(RedoIntegrityError, match="missing or has an invalid current head"):
        runtime.committer.commit("u1", operations)

    assert not head_path.exists()


def test_committed_outbox_resume_cannot_recreate_a_deleted_head_from_stale_redo(
    tmp_path: Path,
) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "committed-outbox-stale-redo", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    for operation in operations:
        committer.redo.begin(operation, phase="started")
    entries = committer.redo.pending_entries()
    head_path = head_set_path(_artifact_root(tmp_path), identity.slot_uri)
    head_path.unlink()

    with pytest.raises(RedoIntegrityError, match="missing or has an invalid current head"):
        committer.resume_canonical_batch("u1", entries)

    assert not head_path.exists()
    assert {entry.operation_id for entry in committer.redo.pending_entries()} == {
        operation.operation_id for operation in operations
    }


def test_committed_outbox_recovery_preserves_valid_outbox_when_head_is_missing(
    tmp_path: Path,
) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "committed-outbox-recovery-head", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    artifact_root = _artifact_root(tmp_path)
    transaction_id = str(operations[0].payload["transaction_id"])
    outbox_path = artifact_root / "system" / "outbox" / f"{transaction_id}.json"
    outbox_bytes = outbox_path.read_bytes()
    head_path = head_set_path(artifact_root, identity.slot_uri)
    head_path.unlink()

    result = RecoveryService(committer.redo, committer).recover_outboxes()

    assert result.failed_count == 1
    assert result.quarantine_count == 0
    assert "current head" in result.last_error
    assert outbox_path.read_bytes() == outbox_bytes
    assert not head_path.exists()


def test_first_start_migration_cannot_recreate_head_after_committed_outbox(
    tmp_path: Path,
) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "first-start-deleted-head", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    artifact_root = _artifact_root(tmp_path)
    assert not (artifact_root / "system" / "migrations" / "memory-closure-v1.json").exists()
    outbox = artifact_root / "system" / "outbox" / f"{operations[0].payload['transaction_id']}.json"
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "committed"
    head_path = head_set_path(artifact_root, identity.slot_uri)
    head_path.unlink()

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "missing its current head" in " ".join(restarted.readiness.reasons)
    assert not head_path.exists()


def test_after_current_head_crash_deleted_later_revision_head_fails_closed(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    initial = _proposal(episode, "head-revision-one", "SQLite", "confirmation", "confirmed")
    identity, _transition, initial_plan = _plan(runtime.source_store, episode, scope, initial)
    runtime.committer.commit(
        "u1",
        initial_plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=episode.episode_id,
        ),
    )
    sqlite_claim = CanonicalMemoryRepository(
        runtime.source_store,
        runtime.relation_store,
    ).load(identity)[1][0]
    replacement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="head-revision-two",
            archive_uri="memoryos://user/u1/sessions/history/head-revision-two",
            messages=[
                {
                    "id": "head-revision-two-message",
                    "role": "user",
                    "content": "The primary storage backend is now changed from SQLite to PostgreSQL.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    replacement = _replacement_proposal(
        replacement_episode,
        "head-revision-two",
        "PostgreSQL",
        sqlite_claim,
    )
    replacement_plan = _reviewed_resolution_plan(
        runtime.source_store,
        runtime.committer,
        replacement_episode,
        replacement,
        command_suffix="head-revision-two",
    )
    runtime.committer.test_hook = _CrashOnce("after_current_head")

    with pytest.raises(SystemExit, match="after_current_head"):
        runtime.committer.commit("u1", list(replacement_plan.operations))

    assert {entry.phase for entry in runtime.committer.redo.pending_entries()} == {"head_published"}
    head_set_path(_artifact_root(tmp_path), identity.slot_uri).unlink()

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "head-published" in " ".join(restarted.readiness.reasons)


@pytest.mark.parametrize(
    ("crash_stage", "queue_job_exists"),
    [("after_committed_outbox", False), ("after_projection_enqueue", True)],
)
def test_committed_outbox_and_projection_enqueue_have_distinct_crash_windows(
    tmp_path: Path,
    crash_stage: str,
    queue_job_exists: bool,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    proposal = _proposal(episode, f"outbox-{crash_stage}", "SQLite", "confirmation", "confirmed")
    _identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    transaction_id = str(operations[0].payload["transaction_id"])
    runtime.committer.test_hook = _CrashOnce(crash_stage)

    with pytest.raises(SystemExit, match=crash_stage):
        runtime.committer.commit("u1", operations)

    outbox = runtime.committer._outbox_path(transaction_id)
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "committed"
    assert (runtime.queue_store.get(f"outbox_{transaction_id}") is not None) is queue_job_exists
    assert {entry.phase for entry in runtime.committer.redo.pending_entries()} == {"head_published"}

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons
    assert restarted.queue_store.get(f"outbox_{transaction_id}") is not None


def test_startup_recovers_crash_after_commit_group_effect_record(tmp_path: Path) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_commit_group_effect,
        args=(str(tmp_path),),
    )
    process.start()
    process.join(timeout=30)
    assert not process.is_alive()
    assert process.exitcode == 19

    recovered = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    group_id = "commit_group_commit-group-crash-task"
    status = recovered.session_commit_service.commit_group_store.load(group_id)
    assert status is not None and status.complete
    assert len(status.canonical_effects) == 1
    pending = CanonicalMemoryRepository(
        recovered.source_store,
        recovered.relation_store,
    ).list_pending(tenant_id="t1", owner_user_id="u1")
    assert len(pending) == 1
    assert pending[0].request_identity == "commit-group-crash-task"

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons
    repeated = restarted.session_commit_service.commit_group_store.load(group_id)
    assert repeated is not None and repeated.complete
    assert repeated.canonical_effects == status.canonical_effects


def test_startup_does_not_steal_a_live_commit_group_lease(tmp_path: Path) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    store = runtime.session_commit_service.commit_group_store
    group_id = "commit_group_live-owner"
    store.create(
        group_id,
        task_id="live-owner",
        archive_uri="memoryos://user/u1/sessions/history/live-owner",
        user_id="u1",
        tenant_id="t1",
    )
    assert store.claim_canonical(group_id, attempt_id="live-attempt", lease_seconds=300)

    competing = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert competing.readiness.state == RuntimeReadinessState.NOT_READY
    assert "live lease" in " ".join(competing.readiness.reasons)


def test_existing_revision_source_ahead_returns_old_snapshot_then_startup_finishes_switch(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    initial = _proposal(episode, "revision-one", "SQLite", "confirmation", "confirmed")
    identity, _transition, initial_plan = _plan(runtime.source_store, episode, scope, initial)
    runtime.committer.commit(
        "u1",
        initial_plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=episode.episode_id,
        ),
    )
    sqlite_claim = CanonicalMemoryRepository(runtime.source_store, runtime.relation_store).load(identity)[1][0]
    replacement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="revision-two",
            archive_uri="memoryos://user/u1/sessions/history/revision-two",
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
    replacement = _replacement_proposal(
        replacement_episode,
        "revision-two",
        "PostgreSQL",
        sqlite_claim,
    )
    replacement_plan = _reviewed_resolution_plan(
        runtime.source_store,
        runtime.committer,
        replacement_episode,
        replacement,
        command_suffix="revision-two-crash",
    )
    operations = list(replacement_plan.operations)
    runtime.committer.test_hook = _CrashOnce("after_source_effect")

    with pytest.raises(SystemExit, match="after_source_effect"):
        runtime.committer.commit("u1", operations)

    before_recovery = CanonicalMemoryRepository(
        runtime.source_store,
        runtime.relation_store,
    ).load(identity)[1]
    assert {claim.canonical_value: claim.current.state for claim in before_recovery} == {"sqlite": "ACTIVE"}

    recovered = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    after_recovery = CanonicalMemoryRepository(
        recovered.source_store,
        recovered.relation_store,
    ).load(identity)[1]
    assert {claim.canonical_value: claim.current.state for claim in after_recovery} == {
        "sqlite": "SUPERSEDED",
        "postgresql": "ACTIVE",
    }
    assert validate_canonical_receipt_history(_artifact_root(tmp_path), tenant_id="t1")["transaction_receipts"] == 2
    assert not list((_artifact_root(tmp_path) / "system" / "redo").glob("*.json"))
    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons


def test_pending_confirm_source_ahead_is_not_authority_until_recovered_receipt_and_head(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    identity_fields = {"decision_topic": "primary storage backend"}
    assert (
        client.remember(
            user_id="u1",
            content="PostgreSQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields=identity_fields,
        )["status"]
        == "COMMITTED"
    )
    assert (
        client.remember(
            user_id="u1",
            content="MySQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields=identity_fields,
        )["status"]
        == "PENDING"
    )
    record = client.list_pending(user_id="u1", lifecycle_states=["PENDING"])[0]

    def crash_after_pending_bundle_pointer(stage: str, _uri: str, _generation_id: str) -> None:
        if stage == "after_current_pointer":
            raise SystemExit("pending source_written crash")

    assert isinstance(client.source_store, FileSystemSourceStore)
    client.source_store.test_hook = crash_after_pending_bundle_pointer

    with pytest.raises(SystemExit, match="source_written"):
        client.review_pending(
            user_id="u1",
            pending_uri=record["uri"],
            decision="CONFIRM",
            expected_lifecycle_revision=record["lifecycle_revision"],
            expected_proposal_fingerprint=record["proposal_fingerprint"],
            command_id="confirm-source-crash",
        )
    client.source_store.test_hook = None

    still_pending = client.list_pending(user_id="u1", lifecycle_states=["PENDING"])
    assert [item["uri"] for item in still_pending] == [record["uri"]]
    assert client.list_pending(user_id="u1", lifecycle_states=["CONFIRMED"]) == []

    recovered = MemoryOSClient(str(tmp_path))
    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    confirmed = recovered.list_pending(user_id="u1", lifecycle_states=["CONFIRMED"])
    assert len(confirmed) == 1 and confirmed[0]["lifecycle_revision"] == 2
    recovered_command = PendingReviewCommandStore(tmp_path, tenant_id="default").load("confirm-source-crash")
    assert recovered_command["status"] == "completed"
    assert recovered_command["result"]["status"] == "confirmed"
    assert recovered_command["result"]["lifecycle_revision"] == 2
    applied = recovered.review_pending(
        user_id="u1",
        pending_uri=record["uri"],
        decision="CONFIRM_AND_APPLY",
        expected_lifecycle_revision=2,
        expected_proposal_fingerprint=record["proposal_fingerprint"],
        command_id="apply-after-confirm-recovery",
    )
    assert applied["status"] == "resolved"
    assert MemoryOSClient(str(tmp_path)).readiness.state == RuntimeReadinessState.READY
