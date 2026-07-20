from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from infrastructure.store.contracts.queue import QueueIdempotencyConflictError, QueueJob, QueueStore
from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory import (
    DocumentCommitIntent,
    DocumentControlIntegrityError,
    DocumentDeletionStatus,
    DocumentIntentStatus,
    MemoryDocumentControlStore,
    MemoryDocumentRevisionStore,
)
from infrastructure.store.memory.erasure_store import MemoryDocumentEraseStore
from memory.commit import DocumentCommitConflict, MemoryDocumentCommitter
from memory.core import (
    ABSENT,
    DocumentEditKind,
    DocumentEditPlan,
    PresentPath,
    new_document_id,
    render_new_document,
)


class _Queue:
    def __init__(self) -> None:
        self.jobs: dict[str, QueueJob] = {}

    def enqueue(self, job: QueueJob) -> QueueJob:
        current = self.jobs.get(job.job_id)
        if current is not None and current != job:
            raise QueueIdempotencyConflictError(job.job_id)
        self.jobs[job.job_id] = current or job
        return self.jobs[job.job_id]


def _components(root: Path):  # noqa: ANN202 - compact test fixture factory.
    source = FileSystemMemoryDocumentStore(root)
    controls = MemoryDocumentControlStore(root)
    revisions = MemoryDocumentRevisionStore(root)
    queue = _Queue()
    committer = MemoryDocumentCommitter(
        source,
        controls,
        revisions,
        cast(QueueStore, queue),
        erasure_store=MemoryDocumentEraseStore(controls.root),
    )
    return source, controls, revisions, queue, committer


def _plan(
    *,
    key: str,
    document_id: str,
    kind: DocumentEditKind,
    expected,
    path: str,
    after: bytes | None = None,
    new_path: str = "",
) -> DocumentEditPlan:
    return DocumentEditPlan(
        idempotency_key=key,
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=kind,
        expected_state=expected,
        evidence_digest="a" * 64,
        edit_summary=f"{kind.value} bounded summary",
        document_id=document_id,
        relative_path=path,
        after_bytes=after,
        new_relative_path=new_path,
        expected_registration_document_id=document_id,
    )


def _commit(committer: MemoryDocumentCommitter, plan: DocumentEditPlan):  # noqa: ANN202
    return committer.commit(
        plan,
        actor_binding="trusted-runtime:user-a",
        evidence_reference="explicit-command:command-1",
    )


def test_create_persists_content_free_control_and_exact_revision_blob(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, revisions, queue, committer = _components(root)
    document_id = new_document_id()
    raw = render_new_document(document_id, "super-secret-markdown-body")

    result = _commit(
        committer,
        _plan(
            key="create-1",
            document_id=document_id,
            kind=DocumentEditKind.CREATE,
            expected=ABSENT,
            path="knowledge/topics/one.md",
            after=raw,
        ),
    )

    assert result.status == DocumentIntentStatus.COMPLETED
    assert result.event is not None and result.event.logical_revision == 1
    assert source.read_raw("default", "user-a", document_id=document_id) == raw
    root_identity = controls.load_root_identity("default", "user-a")
    assert root_identity is not None
    assert root_identity.root_identity == source.full_scan("default", "user-a").root_identity
    assert revisions.read_revision_blob("default", "user-a", document_id, 1) == raw
    intent = controls.load_intent("default", "user-a", result.intent_id)
    assert isinstance(intent, DocumentCommitIntent)
    assert intent.status == DocumentIntentStatus.COMPLETED
    assert len(intent.effects) == 1
    assert intent.after_blob_digest == result.event.after_raw_digest
    assert len(queue.jobs) == 1
    job = next(iter(queue.jobs.values()))
    assert (job.queue_name, job.action) == ("memory_projection", "memory_committed")
    assert job.payload["tenant_id"] == "default"
    assert job.payload["owner_user_id"] == "user-a"
    assert job.payload["document_id"] == document_id

    control_json = [path for path in root.rglob("*.json") if "memory-documents" in path.as_posix()]
    assert control_json
    for path in control_json:
        encoded = path.read_bytes()
        assert b"super-secret-markdown-body" not in encoded
        json.loads(encoded.decode("utf-8"))


def test_identical_update_is_no_op_without_revision_event_or_queue_job(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, _, revisions, queue, committer = _components(root)
    document_id = new_document_id()
    path = "knowledge/topics/no-op.md"
    raw = render_new_document(document_id, "unchanged")
    _commit(
        committer,
        _plan(
            key="create",
            document_id=document_id,
            kind=DocumentEditKind.CREATE,
            expected=ABSENT,
            path=path,
            after=raw,
        ),
    )
    expected = source.read_state("default", "user-a", path)
    assert isinstance(expected, PresentPath)

    result = _commit(
        committer,
        _plan(
            key="same-update",
            document_id=document_id,
            kind=DocumentEditKind.UPDATE,
            expected=expected,
            path=path,
            after=raw,
        ),
    )

    assert result.no_op is True
    assert result.event is None
    assert revisions.latest_revision("default", "user-a", document_id) == 1
    assert len(queue.jobs) == 1


def test_rename_delete_and_restore_are_new_exact_cas_revisions(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, revisions, _, committer = _components(root)
    document_id = new_document_id()
    first_path = "knowledge/topics/first.md"
    second_path = "knowledge/topics/second.md"
    raw = render_new_document(document_id, "restorable content")
    _commit(
        committer,
        _plan(
            key="create",
            document_id=document_id,
            kind=DocumentEditKind.CREATE,
            expected=ABSENT,
            path=first_path,
            after=raw,
        ),
    )

    expected = source.read_state("default", "user-a", first_path)
    rename_plan = _plan(
        key="rename",
        document_id=document_id,
        kind=DocumentEditKind.RENAME,
        expected=expected,
        path=first_path,
        new_path=second_path,
    )
    rename_result = _commit(committer, rename_plan)
    assert _commit(committer, rename_plan).event == rename_result.event
    rename_intent = controls.load_intent("default", "user-a", rename_result.intent_id)
    assert rename_intent is not None and len(rename_intent.effects) == 2
    assert rename_intent.effects[0].after == ABSENT
    assert rename_intent.effects[1].before == ABSENT
    assert source.read_state("default", "user-a", first_path) == ABSENT

    expected = source.read_state("default", "user-a", second_path)
    delete_plan = _plan(
        key="delete",
        document_id=document_id,
        kind=DocumentEditKind.DELETE,
        expected=expected,
        path=second_path,
    )
    delete_result = _commit(committer, delete_plan)
    assert _commit(committer, delete_plan).event == delete_result.event
    delete_intent = controls.load_intent("default", "user-a", delete_result.intent_id)
    assert delete_intent is not None
    assert delete_intent.after_blob_digest == ""
    assert delete_intent.revision_blob_role == "before_delete"
    assert source.read_state("default", "user-a", second_path) == ABSENT
    assert revisions.read_revision_blob("default", "user-a", document_id, 3) == raw
    barrier = controls.load_publication_barrier("default", "user-a", document_id)
    assert barrier is not None
    assert barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
    assert barrier.deletion_generation == 3
    assert barrier.deletion_event_digest

    with pytest.raises(DocumentCommitConflict, match="explicit revision restore"):
        _commit(
            committer,
            _plan(
                key="direct-recreate",
                document_id=document_id,
                kind=DocumentEditKind.CREATE,
                expected=ABSENT,
                path=second_path,
                after=raw,
            ),
        )

    restore_plan = _plan(
        key="restore",
        document_id=document_id,
        kind=DocumentEditKind.CREATE,
        expected=ABSENT,
        path=second_path,
    )
    restored = committer.restore_revision(
        restore_plan,
        revision=1,
        actor_binding="trusted-runtime:user-a",
        evidence_reference="explicit-command:restore-1",
    )
    assert restored.event is not None and restored.event.logical_revision == 4
    assert source.read_raw("default", "user-a", relative_path=second_path) == raw
    restored_control = controls.load_control("default", "user-a", document_id)
    assert restored_control is not None
    assert restored_control.projection_generation > barrier.deletion_generation
    assert restored_control.restored_from_deletion_generation == barrier.deletion_generation
    assert controls.load_publication_barrier("default", "user-a", document_id) == barrier


def test_soft_forget_barrier_is_durable_before_live_unlink(tmp_path: Path) -> None:
    source = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    revisions = MemoryDocumentRevisionStore(tmp_path)
    queue = _Queue()
    document_id = new_document_id()
    relative_path = "knowledge/topics/fenced.md"
    raw = render_new_document(document_id, "fenced delete")
    observed: list[str] = []

    def hook(stage: str, intent: DocumentCommitIntent) -> None:
        if stage != "intent_prepared" or intent.edit_kind is not DocumentEditKind.DELETE:
            return
        barrier = controls.load_publication_barrier("default", "user-a", document_id)
        assert barrier is not None and barrier.deletion_generation == intent.projection_generation
        assert isinstance(source.read_state("default", "user-a", relative_path), PresentPath)
        observed.append(stage)

    committer = MemoryDocumentCommitter(
        source,
        controls,
        revisions,
        cast(QueueStore, queue),
        test_hook=hook,
        erasure_store=MemoryDocumentEraseStore(controls.root),
    )
    _commit(
        committer,
        _plan(
            key="create-fenced",
            document_id=document_id,
            kind=DocumentEditKind.CREATE,
            expected=ABSENT,
            path=relative_path,
            after=raw,
        ),
    )
    expected = source.read_state("default", "user-a", relative_path)
    assert isinstance(expected, PresentPath)
    _commit(
        committer,
        _plan(
            key="delete-fenced",
            document_id=document_id,
            kind=DocumentEditKind.DELETE,
            expected=expected,
            path=relative_path,
        ),
    )
    assert observed == ["intent_prepared"]


def test_idempotency_retry_cannot_change_effect_or_lineage(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, _, _, _, committer = _components(root)
    document_id = new_document_id()
    path = "knowledge/topics/idempotent.md"
    raw = render_new_document(document_id, "first")
    plan = _plan(
        key="one-key",
        document_id=document_id,
        kind=DocumentEditKind.CREATE,
        expected=ABSENT,
        path=path,
        after=raw,
    )
    first = _commit(committer, plan)
    repeated = _commit(committer, plan)
    assert repeated.intent_id == first.intent_id
    assert repeated.event == first.event

    changed = replace_plan_after(plan, render_new_document(document_id, "changed"))
    with pytest.raises(DocumentControlIntegrityError):
        _commit(committer, changed)
    assert source.read_raw("default", "user-a", relative_path=path) == raw


def replace_plan_after(plan: DocumentEditPlan, after: bytes) -> DocumentEditPlan:
    return DocumentEditPlan(
        idempotency_key=plan.idempotency_key,
        tenant_id=plan.tenant_id,
        owner_user_id=plan.owner_user_id,
        edit_kind=plan.edit_kind,
        expected_state=plan.expected_state,
        evidence_digest=plan.evidence_digest,
        edit_summary=plan.edit_summary,
        document_id=plan.document_id,
        relative_path=plan.relative_path,
        after_bytes=after,
        new_relative_path=plan.new_relative_path,
        expected_new_state=plan.expected_new_state,
        expected_registration_document_id=plan.expected_registration_document_id,
    )
