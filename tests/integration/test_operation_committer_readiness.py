from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

import memoryos.runtime.container as runtime_container_module
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.memory.canonical import (
    CandidateProposalAdapter,
    CanonicalMemoryFormationService,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.schema import MemoryTypeSchema
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeNotReadyError, RuntimeReadinessState
from tests.unit.test_canonical_pending_lifecycle import _archive, _pending_draft
from tests.unit.test_canonical_transaction_commit import _plan, _proposal, _setup


class _CountingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemorySemanticProposal]:
        del archive, schemas
        self.calls += 1
        return ()


def _assert_no_memory_commit_artifacts(root: Path, runtime) -> None:  # noqa: ANN001
    artifact_root = root / "tenants" / "t1"
    for name in ("transactions", "operations", "current-heads", "outbox", "redo"):
        assert not list((artifact_root / "system" / name).glob("*.json"))
    assert not any(dict(obj.metadata or {}).get("canonical_kind") for obj in runtime.source_store.list_objects())


@pytest.mark.parametrize(
    "state",
    [RuntimeReadinessState.NOT_READY, RuntimeReadinessState.RECOVERING],
)
def test_direct_committer_rejects_canonical_and_contextdb_before_artifacts(
    tmp_path: Path,
    state: RuntimeReadinessState,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    proposal = _proposal(episode, f"blocked-{state.value}", "SQLite", "confirmation", "confirmed")
    _identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    runtime.readiness.transition(state, reasons=("startup proof incomplete",))

    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.committer.commit("u1", operations)
    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.context_db.commit_operations(operations)

    _assert_no_memory_commit_artifacts(tmp_path, runtime)


@pytest.mark.parametrize(
    "state",
    [RuntimeReadinessState.NOT_READY, RuntimeReadinessState.RECOVERING],
)
def test_direct_committer_rejects_pending_before_artifacts(
    tmp_path: Path,
    state: RuntimeReadinessState,
) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    archive = _archive(task_id=f"blocked-pending-{state.value.lower()}")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(
        _pending_draft(missing_source=True),
        episode,
        archive,
    )
    formed = CanonicalMemoryFormationService(runtime.source_store).plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        commit_group_id=f"blocked-pending-{state.value.lower()}",
    )
    runtime.readiness.transition(state, reasons=("startup proof incomplete",))

    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.committer.commit("u1", list(formed.operations))

    _assert_no_memory_commit_artifacts(tmp_path, runtime)


@pytest.mark.parametrize(
    "state",
    [RuntimeReadinessState.NOT_READY, RuntimeReadinessState.RECOVERING],
)
def test_direct_session_service_rejects_before_archive_model_or_group_mutation(
    tmp_path: Path,
    state: RuntimeReadinessState,
) -> None:
    extractor = _CountingExtractor()
    runtime = build_runtime_container(
        RuntimeConfig(
            root=str(tmp_path),
            tenant_id="t1",
            memory_extractor=extractor,
        )
    )
    archive = SessionArchive(
        user_id="u1",
        session_id=f"blocked-{state.value.lower()}",
        archive_uri=(f"memoryos://user/u1/sessions/history/blocked-{state.value.lower()}"),
        messages=[{"id": "m1", "role": "user", "content": "Remember this durable rule."}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id=f"blocked-{state.value.lower()}-task",
    )
    runtime.readiness.transition(state, reasons=("startup proof incomplete",))

    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.session_commit_service.sync_archive(archive)
    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.session_commit_service.async_commit(archive)

    assert extractor.calls == 0
    assert not runtime.session_archive_store.archive_exists(
        archive.archive_uri,
        tenant_id="t1",
    )
    artifact_root = tmp_path / "tenants" / "t1"
    assert not list((artifact_root / "system" / "commit_groups").glob("*.json"))
    assert not list((artifact_root / "system" / "planning-envelopes").glob("*.json"))
    assert runtime.queue_store.stats().get("pending", 0) == 0
    _assert_no_memory_commit_artifacts(tmp_path, runtime)


def test_startup_validates_receipt_history_before_resuming_commit_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_receipt_history(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        del args, kwargs
        calls.append("receipt_history")
        raise RuntimeError("historical receipt digest mismatch")

    def forbidden_group_recovery(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        del args, kwargs
        calls.append("commit_groups")
        raise AssertionError("commit groups must not resume before receipt validation")

    monkeypatch.setattr(
        runtime_container_module,
        "validate_canonical_receipt_history",
        fail_receipt_history,
    )
    monkeypatch.setattr(
        runtime_container_module,
        "_recover_startup_commit_groups",
        forbidden_group_recovery,
    )

    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert runtime.readiness.state == RuntimeReadinessState.NOT_READY
    assert calls == ["receipt_history"]
    assert "historical receipt digest mismatch" in " ".join(runtime.readiness.reasons)


def test_startup_rejects_live_queue_lease_before_resuming_commit_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    runtime.queue_store.enqueue(
        QueueJob(
            job_id="live-startup-lease",
            queue_name="session_commit",
            action="commit_session",
            target_uri="memoryos://user/u1/sessions/history/live-startup-lease",
        )
    )
    lease = runtime.queue_store.lease(
        "session_commit",
        lease_owner="still-running-worker",
        lease_seconds=300,
        job_ids=("live-startup-lease",),
    )[0]
    recovered_groups = False

    def forbidden_group_recovery(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal recovered_groups
        del args, kwargs
        recovered_groups = True
        raise AssertionError("commit groups must not race a live queue lease")

    monkeypatch.setattr(
        runtime_container_module,
        "_recover_startup_commit_groups",
        forbidden_group_recovery,
    )

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "active lease" in " ".join(restarted.readiness.reasons)
    assert recovered_groups is False
    current = restarted.queue_store.get(lease.job_id)
    assert current is not None
    assert current.status == "leased"
    assert current.lease_generation == lease.lease_generation


@pytest.mark.parametrize("queue_name", ["session_commit", "memory_proposal", "behavior_projection"])
def test_startup_does_not_treat_nonprojection_dead_letter_as_projection_failure(
    tmp_path: Path,
    queue_name: str,
) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    queued = runtime.queue_store.enqueue(
        QueueJob(
            job_id=f"terminal-{queue_name}",
            queue_name=queue_name,
            action="terminal_work",
            target_uri=f"memoryos://user/u1/work/{queue_name}",
        )
    )
    leased = runtime.queue_store.lease(
        queue_name,
        lease_owner="terminal-worker",
        job_ids=(queued.job_id,),
    )[0]
    runtime.queue_store.fail(leased, "explicit terminal outcome")

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons
    terminal = restarted.queue_store.get(queued.job_id)
    assert terminal is not None and terminal.status == "dead_letter"
    assert restarted.readiness.details["projection_queue_preflight"].get("dead_letter", 0) == 0
    assert restarted.readiness.details["projection_queue_final"].get("dead_letter", 0) == 0
    assert restarted.readiness.details["queue"].get("dead_letter", 0) == 1


def test_direct_sdk_pending_review_reports_not_ready_before_argument_validation(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    client.readiness.transition(RuntimeReadinessState.NOT_READY, reasons=("startup proof incomplete",))

    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        client.review_pending(
            user_id="u1",
            pending_uri="",
            decision="",
            expected_lifecycle_revision=0,
            expected_proposal_fingerprint="",
            command_id="",
        )
