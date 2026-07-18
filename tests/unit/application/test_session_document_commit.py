from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import (
    FileSystemMemoryDocumentStore,
)
from memoryos.adapters.persistence.filesystem.session_archive import SessionArchiveStore
from memoryos.adapters.persistence.in_memory.queue_store import InMemoryQueueStore
from memoryos.application.session.commit_group import CommitGroupIntegrityError, CommitGroupStore
from memoryos.application.session.commit_service import DerivedConsumerError, SessionCommitService
from memoryos.application.session.planners.context_commit_planner import ContextCommitPlanner
from memoryos.application.session.planners.memory_commit_planner import (
    MemoryDocumentPlanningResult,
    PlannedMemoryEdit,
)
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.evidence_encoder import (
    SessionEvidenceEvent,
    register_session_evidence_encoder,
)
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.integrity import canonical_digest
from memoryos.memory.documents import (
    DocumentCommitResult,
    DocumentConflictError,
    DocumentEditPlan,
    MemoryCandidateKind,
    MemoryDocumentCommitter,
    MemoryDocumentControlStore,
    MemoryDocumentPlanner,
    MemoryDocumentRevisionStore,
    MemoryEditProposal,
)
from memoryos.memory.evidence import SessionArchiveEpisodeAdapter
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.workers.session_commit_worker import SessionCommitWorker


class _EvidenceEncoder:
    def encode(self, archive: SessionArchive) -> tuple[SessionEvidenceEvent, ...]:
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return tuple(
            SessionEvidenceEvent(
                payload=event.to_dict(),
                event_id=event.event_id,
                event_digest=event.digest,
                event_type=event.event_type,
                category=str(event.metadata.get("category", "")),
                occurred_at=event.occurred_at,
                ingested_at=event.ingested_at,
                sequence=event.sequence,
            )
            for event in episode.events
        )


class _SessionMemoryPlanner:
    def __init__(self, document_planner: MemoryDocumentPlanner, *, fail_once: bool = False) -> None:
        self.document_planner = document_planner
        self.fail_once = fail_once
        self.calls = 0

    def plan_session(
        self,
        archive: SessionArchive,
        *,
        tenant_id: str,
        owner_user_id: str,
        commit_group_id: str,
    ) -> MemoryDocumentPlanningResult:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise OSError("transient extraction transport")
        proposal = MemoryEditProposal(
            candidate_kind=MemoryCandidateKind.PREFERENCE,
            title="Preferred editor",
            body="SESSION-DOCUMENT-SECRET uses Vim.",
            evidence_refs=("message:0",),
        )
        proposal_digest = canonical_digest([proposal.to_dict()])
        plan = self.document_planner.plan(
            proposal,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            idempotency_key=f"{commit_group_id}:memory:0:{proposal_digest}",
            evidence_digest=archive.archive_digest,
        )
        return MemoryDocumentPlanningResult(
            edits=(PlannedMemoryEdit(proposal=proposal, plan=plan),),
            proposal_set_digest=proposal_digest,
            edit_proposal_count=0,
            candidate_count=1,
        )


class _CountingDocumentPlanner(MemoryDocumentPlanner):
    def __init__(self, store: FileSystemMemoryDocumentStore) -> None:
        super().__init__(store)
        self.replan_calls = 0

    def replan(
        self,
        sealed_proposal: MemoryEditProposal,
        *,
        tenant_id: str,
        owner_user_id: str,
        idempotency_key: str,
        evidence_digest: str,
    ) -> DocumentEditPlan:
        self.replan_calls += 1
        return super().replan(
            sealed_proposal,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
            evidence_digest=evidence_digest,
        )


class _ConflictOnceCommitter:
    def __init__(
        self,
        delegate: MemoryDocumentCommitter,
        planner: MemoryDocumentPlanner,
    ) -> None:
        self.delegate = delegate
        self.planner = planner
        self.control_store = delegate.control_store
        self.projection_queue = delegate.projection_queue
        self.calls = 0

    def commit(
        self,
        plan: DocumentEditPlan,
        *,
        actor_binding: str,
        evidence_reference: str,
    ) -> DocumentCommitResult:
        self.calls += 1
        if self.calls == 1:
            competing = MemoryEditProposal(
                candidate_kind=MemoryCandidateKind.PREFERENCE,
                title="Existing preference",
                body="A competing writer arrived first.",
                evidence_refs=("external:1",),
            )
            competing_plan = self.planner.plan(
                competing,
                tenant_id=plan.tenant_id,
                owner_user_id=plan.owner_user_id,
                idempotency_key="external-competing-create",
                evidence_digest="f" * 64,
            )
            self.delegate.commit(
                competing_plan,
                actor_binding="external:user-a",
                evidence_reference="external:1",
            )
            raise DocumentConflictError("simulated compare-and-swap conflict")
        return self.delegate.commit(
            plan,
            actor_binding=actor_binding,
            evidence_reference=evidence_reference,
        )

    def recover_intent(
        self,
        tenant_id: str,
        owner_user_id: str,
        intent_id: str,
    ) -> DocumentCommitResult:
        return self.delegate.recover_intent(tenant_id, owner_user_id, intent_id)


class _ProjectionJournalStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    def set_session_projection_frontier(self, **payload) -> None:  # noqa: ANN003
        self.rows[str(payload["archive_uri"])] = {
            "tenant_id": payload["tenant_id"],
            "source_uri": payload["archive_uri"],
            "owner_user_id": payload["owner_user_id"],
            "workspace_id": payload["workspace_id"],
            "source_id": payload["session_id"],
            "source_digest": payload["manifest_digest"],
            "status": payload["status"],
            "last_error": payload["error"],
        }

    def list_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        statuses: tuple[str, ...],
        after_archive_uri: str,
        limit: int,
    ) -> list[dict[str, object]]:
        return [
            dict(row)
            for uri, row in sorted(self.rows.items())
            if uri > after_archive_uri and row.get("tenant_id") == tenant_id and row.get("status") in statuses
        ][:limit]


class _FailingSessionProjector:
    def __init__(self, store: _ProjectionJournalStore) -> None:
        self.catalog_store = store
        self.calls = 0

    def project(self, _archive: SessionArchive) -> SimpleNamespace:
        self.calls += 1
        if self.calls == 1:
            raise OSError("transient Session projection failure")
        return SimpleNamespace(projected=1)


class _OrdinaryContextPlanner(ContextCommitPlanner):
    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        return [
            ContextOperation(
                context_type=ContextType.RESOURCE,
                action=OperationAction.REFRESH_LAYERS,
                payload={"reason": "ordinary-session-consumer"},
                user_id=archive.user_id,
                target_uri="memoryos://resources/editor",
                source_session_id=archive.session_id,
            )
        ]


class _RecordingOperationCommitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ContextOperation]]] = []

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        self.calls.append((user_id, list(operations)))
        return ContextDiff(user_id=user_id, operations=list(operations))


def _archive(session_id: str = "session-a") -> SessionArchive:
    return SessionArchive(
        user_id="user-a",
        session_id=session_id,
        archive_uri=f"memoryos://user/user-a/sessions/history/{session_id}",
        messages=[{"role": "user", "content": "Please remember my editor preference."}],
    )


def _service(
    root: Path,
    *,
    fail_once: bool = False,
    document_hook=None,  # noqa: ANN001
    archive_hook=None,  # noqa: ANN001
):  # noqa: ANN202
    register_session_evidence_encoder(_EvidenceEncoder())
    archive_store = SessionArchiveStore(root, test_hook=archive_hook)
    queue = InMemoryQueueStore()
    document_store = FileSystemMemoryDocumentStore(root)
    document_planner = MemoryDocumentPlanner(document_store)
    memory_planner = _SessionMemoryPlanner(document_planner, fail_once=fail_once)
    memory_committer = MemoryDocumentCommitter(
        document_store,
        MemoryDocumentControlStore(root),
        MemoryDocumentRevisionStore(root),
        queue,
        test_hook=document_hook,
    )
    service = SessionCommitService(
        archive_store,
        queue,
        memory_planner=memory_planner,
        memory_committer=memory_committer,
        document_planner=document_planner,
    )
    return service, queue, document_store, memory_committer, memory_planner


def test_inline_commit_uses_document_committer_and_content_free_group(tmp_path: Path) -> None:
    service, queue, document_store, committer, planner = _service(tmp_path)
    archive = _archive()

    result = service.commit_session(archive, async_commit=True)

    assert result.done is True
    assert result.status == "done"
    assert result.memory_committed is True
    assert result.memory_document_change_count == 1
    assert result.edit_proposal_count == 0
    assert result.edit_proposal_ids == ()
    assert not hasattr(result, "canonical_committed")
    assert planner.calls == 1
    assert document_store.read_raw("default", "user-a", relative_path="preferences.md").endswith(
        b"SESSION-DOCUMENT-SECRET uses Vim.\n"
    )
    assert queue.get(archive.task_id) is None

    group = service.commit_group_store.load(result.commit_group_id)
    assert group is not None and group.complete
    assert len(group.memory_effects) == 1
    binding = committer.control_store.load_event_binding(
        "default",
        "user-a",
        group.memory_effects[0].document_id,
        group.memory_effects[0].change_event_id,
    )
    assert binding is not None
    intent, _event = binding
    assert intent is not None
    projection = queue.get(intent.projection_job_id)
    assert projection is not None and projection.queue_name == "memory_projection"

    encoded_group = service.commit_group_store.path(result.commit_group_id).read_bytes()
    assert b"SESSION-DOCUMENT-SECRET" not in encoded_group
    assert b"canonical" not in encoded_group.lower()
    payload = json.loads(encoded_group)
    assert set(payload["memory_effects"][0]) == {
        "document_id",
        "change_event_id",
        "change_digest",
    }

    outputs = service.archive_store.read_async_outputs(archive)
    assert outputs["memory_diff"]["memory_document_change_count"] == 1
    assert "operations" not in outputs["memory_diff"]
    assert "SESSION-DOCUMENT-SECRET" not in json.dumps(outputs["memory_diff"])


def test_failure_before_archive_does_not_publish_retry_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, queue, _, _, _ = _service(tmp_path)
    archive = _archive("before-archive")

    def fail_write(_archive: SessionArchive) -> Path:
        raise OSError("disk unavailable before durable archive")

    monkeypatch.setattr(service.archive_store, "write_sync_archive", fail_write)

    with pytest.raises(OSError):
        service.commit_session(archive, async_commit=True)

    assert service.archive_store.archive_exists(archive.archive_uri, tenant_id="default") is False
    assert queue.get(archive.task_id) is None


def test_inline_failure_queues_same_task_and_worker_replay_is_idempotent(tmp_path: Path) -> None:
    service, queue, document_store, committer, planner = _service(tmp_path, fail_once=True)
    archive = _archive("after-archive")

    with pytest.raises(DerivedConsumerError):
        service.commit_session(archive, async_commit=True)

    job = queue.get(archive.task_id)
    assert job is not None
    assert job.queue_name == "session_commit"
    assert job.target_uri == archive.archive_uri
    assert job.payload["manifest_digest"] == archive.manifest_digest
    group_id = f"commit_group_{archive.task_id}"
    failed_group = service.commit_group_store.load(group_id)
    assert failed_group is not None
    assert failed_group.consumers["memory"].status == "failed"
    assert all(failed_group.consumers[name].status == "completed" for name in ("behavior", "action_policy", "context"))

    worker_result = SessionCommitWorker(service, worker_id="test-worker").process_pending()

    assert worker_result["recovered"] == 1
    assert worker_result["committed"] == 1
    assert worker_result["failed"] == 0
    settled_after_worker = queue.get(archive.task_id)
    assert settled_after_worker is not None and settled_after_worker.status == "done"
    persisted = service.archive_store.read_archive_at_manifest(
        archive.archive_uri,
        archive.manifest_digest,
        tenant_id="default",
    )
    recovered = service.async_commit(persisted)

    assert recovered.done is True
    assert recovered.commit_group_id == group_id
    assert recovered.memory_document_change_count == 1
    assert planner.calls == 2
    assert SessionCommitWorker(service).process_archive(persisted) == {
        "task_id": archive.task_id,
        "status": "done",
        "done": True,
        "memory_committed": True,
    }
    effect = recovered.commit_group_status["memory_effects"][0]
    binding = committer.control_store.load_event_binding(
        "default", "user-a", effect["document_id"], effect["change_event_id"]
    )
    assert binding is not None
    intent, _event = binding
    assert intent is not None and intent.logical_revision == 1
    assert (
        document_store.read_raw("default", "user-a", relative_path="preferences.md").count(
            b"SESSION-DOCUMENT-SECRET uses Vim."
        )
        == 1
    )

    replayed = service.async_commit(persisted)
    assert replayed.done is True
    assert replayed.commit_group_id == group_id
    assert planner.calls == 2
    replayed_binding = committer.control_store.load_event_binding(
        "default", "user-a", effect["document_id"], effect["change_event_id"]
    )
    assert replayed_binding is not None
    replayed_intent, _event = replayed_binding
    assert replayed_intent is not None and replayed_intent.logical_revision == 1
    settled_job = queue.get(archive.task_id)
    assert settled_job is not None and settled_job.status == "done"


def test_session_worker_retries_derived_failure_then_acks_same_task(tmp_path: Path) -> None:
    service, queue, _, _, planner = _service(tmp_path, fail_once=True)
    archive = _archive("worker-retry")
    service.sync_archive(archive)

    first = SessionCommitWorker(service, worker_id="retry-worker-a").process_pending()

    assert first["claimed"] == 1
    assert first["failed"] == 1
    queued = queue.get(archive.task_id)
    assert queued is not None and queued.status == "pending" and queued.retry_count == 1

    second = SessionCommitWorker(service, worker_id="retry-worker-b").process_pending()

    assert second["recovered"] == 1
    assert second["committed"] == 1
    settled = queue.get(archive.task_id)
    assert settled is not None and settled.status == "done"
    assert planner.calls == 2


def test_crash_after_document_completion_resumes_intent_without_second_revision(tmp_path: Path) -> None:
    crashed = False

    def crash_once(stage, _intent) -> None:  # noqa: ANN001
        nonlocal crashed
        if stage == "completed" and not crashed:
            crashed = True
            raise OSError("crash after source and projection job became durable")

    service, queue, document_store, committer, planner = _service(
        tmp_path,
        document_hook=crash_once,
    )
    archive = _archive("document-tail-crash")

    with pytest.raises(DerivedConsumerError):
        service.commit_session(archive, async_commit=True)

    failed_group = service.commit_group_store.load(f"commit_group_{archive.task_id}")
    assert failed_group is not None and failed_group.memory_effects == []
    assert queue.get(archive.task_id) is not None
    raw_after_crash = document_store.read_raw("default", "user-a", relative_path="preferences.md")

    persisted = service.archive_store.read_archive_at_manifest(
        archive.archive_uri,
        archive.manifest_digest,
        tenant_id="default",
    )
    recovered = service.async_commit(persisted)

    assert recovered.done is True
    assert recovered.memory_document_change_count == 1
    assert planner.calls == 2
    assert document_store.read_raw("default", "user-a", relative_path="preferences.md") == raw_after_crash
    effect = recovered.commit_group_status["memory_effects"][0]
    binding = committer.control_store.load_event_binding(
        "default", "user-a", effect["document_id"], effect["change_event_id"]
    )
    assert binding is not None
    intent, _event = binding
    assert intent is not None and intent.logical_revision == 1


def test_cas_conflict_replans_only_from_same_sealed_proposal(tmp_path: Path) -> None:
    register_session_evidence_encoder(_EvidenceEncoder())
    archive_store = SessionArchiveStore(tmp_path)
    queue = InMemoryQueueStore()
    document_store = FileSystemMemoryDocumentStore(tmp_path)
    document_planner = _CountingDocumentPlanner(document_store)
    session_planner = _SessionMemoryPlanner(document_planner)
    delegate = MemoryDocumentCommitter(
        document_store,
        MemoryDocumentControlStore(tmp_path),
        MemoryDocumentRevisionStore(tmp_path),
        queue,
    )
    conflict_committer = _ConflictOnceCommitter(delegate, document_planner)
    service = SessionCommitService(
        archive_store,
        queue,
        memory_planner=session_planner,
        memory_committer=cast(MemoryDocumentCommitter, conflict_committer),
        document_planner=document_planner,
    )

    result = service.commit_session(_archive("cas-replan"), async_commit=True)

    assert result.done is True
    assert result.memory_document_change_count == 1
    assert session_planner.calls == 1
    assert document_planner.replan_calls == 1
    assert conflict_committer.calls == 2
    raw = document_store.read_raw("default", "user-a", relative_path="preferences.md")
    assert b"A competing writer arrived first." in raw
    assert raw.count(b"SESSION-DOCUMENT-SECRET uses Vim.") == 1


def test_projection_failure_after_archive_is_journaled_and_same_task_is_replayed(tmp_path: Path) -> None:
    register_session_evidence_encoder(_EvidenceEncoder())
    archive_store = SessionArchiveStore(tmp_path)
    queue = InMemoryQueueStore()
    journal_store = _ProjectionJournalStore()
    projector = _FailingSessionProjector(journal_store)
    service = SessionCommitService(
        archive_store,
        queue,
        session_projector=projector,
    )
    archive = _archive("projection-window")

    with pytest.raises(OSError):
        service.commit_session(archive, async_commit=True)

    assert archive_store.archive_exists(archive.archive_uri, tenant_id="default")
    job = queue.get(archive.task_id)
    assert job is not None and job.payload["manifest_digest"] == archive.manifest_digest
    assert journal_store.rows[archive.archive_uri]["status"] == "FAILED"
    assert journal_store.rows[archive.archive_uri]["source_digest"] == archive.manifest_digest

    recovered = service.recover_session_projection_frontier()

    assert recovered == {"projected": 1, "abandoned": 0, "failed": 0}
    assert projector.calls == 2
    assert journal_store.rows[archive.archive_uri]["status"] == "PROJECTED"
    replayed_job = queue.get(archive.task_id)
    assert replayed_job is not None and replayed_job.job_id == job.job_id


def test_startup_resume_publishes_outputs_after_process_crash_with_complete_group(tmp_path: Path) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    crashed = False

    def crash_before_output_head(stage: str, _task_id: str) -> None:
        nonlocal crashed
        if stage == "before_current" and not crashed:
            crashed = True
            raise SimulatedProcessCrash

    service, queue, _, committer, planner = _service(
        tmp_path,
        archive_hook=crash_before_output_head,
    )
    archive = _archive("output-head-crash")

    with pytest.raises(SimulatedProcessCrash):
        service.commit_session(archive, async_commit=True)

    group_id = f"commit_group_{archive.task_id}"
    group = service.commit_group_store.load(group_id)
    assert group is not None and group.complete
    assert queue.get(archive.task_id) is None
    assert service.archive_store.async_outputs_done_for_task(archive) is False

    restarted = SessionCommitService(
        SessionArchiveStore(tmp_path),
        queue,
        memory_planner=planner,
        memory_committer=committer,
        document_planner=planner.document_planner,
    )
    persisted = restarted.archive_store.read_archive_at_manifest(
        archive.archive_uri,
        archive.manifest_digest,
        tenant_id="default",
    )
    resumable = restarted.resumable_commit_groups()
    assert [item.group_id for item in resumable] == [group_id]

    worker_result = SessionCommitWorker(restarted, worker_id="startup-recovery").process_pending()

    assert worker_result["recovered"] == 1
    assert worker_result["failed"] == 0
    assert restarted.archive_store.async_outputs_done_for_task(persisted) is True
    assert restarted.resumable_commit_groups() == ()
    assert planner.calls == 1


def test_commit_group_rejects_symlinked_control_directory(tmp_path: Path) -> None:
    (tmp_path / "system").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "system" / "commit_groups").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CommitGroupIntegrityError):
        CommitGroupStore(tmp_path).load("commit_group_untrusted")

    assert list(outside.iterdir()) == []


def test_ordinary_session_operations_stay_on_operation_committer(tmp_path: Path) -> None:
    register_session_evidence_encoder(_EvidenceEncoder())
    queue = InMemoryQueueStore()
    recorder = _RecordingOperationCommitter()
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        queue,
        committer=cast(OperationCommitter, recorder),
        context_planner=_OrdinaryContextPlanner(),
    )
    archive = _archive("ordinary-operation")

    first = service.commit_session(archive, async_commit=True)
    call_count = len(recorder.calls)
    repeated = service.commit_session(archive, async_commit=True)

    committed = [operation for _, operations in recorder.calls for operation in operations]
    assert first.done is True and repeated.done is True
    assert first.memory_committed is True
    assert first.memory_document_change_count == 0
    assert len(committed) == 1
    assert committed[0].context_type is ContextType.RESOURCE
    assert committed[0].payload["reason"] == "ordinary-session-consumer"
    assert committed[0].payload["commit_group_id"] == first.commit_group_id
    assert committed[0].payload["commit_consumer"] == "context"
    assert committed[0].operation_id.startswith("op_")
    assert len(recorder.calls) == call_count


def test_commit_group_control_root_is_physically_tenant_scoped(tmp_path: Path) -> None:
    register_session_evidence_encoder(_EvidenceEncoder())
    service = SessionCommitService(
        SessionArchiveStore(tmp_path, tenant_id="tenant-a"),
        InMemoryQueueStore(),
    )
    archive = _archive("tenant-scoped-group")
    archive.metadata = {"tenant_id": "tenant-a"}

    result = service.commit_session(archive, async_commit=True)

    expected_root = (tmp_path / "tenants" / "tenant-a").resolve()
    assert result.done is True
    assert service.commit_group_store.artifact_root == expected_root
    assert service.commit_group_store.path(result.commit_group_id).is_file()
    assert not (tmp_path / "system" / "commit_groups").exists()
