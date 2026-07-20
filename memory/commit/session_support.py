"""Session 提交的持久结果校验、排队和租户绑定。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, cast

from foundation.ids import stable_hash
from foundation.integrity import canonical_digest
from infrastructure.store.contracts.queue import QueueJob
from infrastructure.store.memory.control_store import (
    DocumentDeletionStatus,
    DocumentIntentStatus,
)
from infrastructure.store.session.commit_group import (
    CommitGroupStatus,
    MemoryDocumentEffect,
)
from memory.commit.document_commit import (
    DocumentCommitResult,
)
from memory.commit.errors import RevisionConflictError
from memory.commit.session_consumers import _SessionCommitConsumers
from memory.core.model import DocumentEditPlan
from memory.ports.document_store import DocumentConflictError
from pre.session import SessionArchive
from transaction.model.context_operation import ContextOperation


class _SessionCommitSupport(_SessionCommitConsumers):
    def _validate_persisted_memory_effects(self, group: CommitGroupStatus) -> None:
        if not group.memory_effects:
            return
        if self.memory_committer is None:
            raise RuntimeError("durable memory effects require MemoryDocumentCommitter")
        for effect in group.memory_effects:
            binding = self.memory_committer.control_store.load_event_binding(
                group.tenant_id,
                group.user_id,
                effect.document_id,
                effect.change_event_id,
            )
            if binding is None:
                barrier = self.memory_committer.control_store.load_publication_barrier(
                    group.tenant_id,
                    group.user_id,
                    effect.document_id,
                )
                if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
                    continue
                raise RuntimeError("commit-group memory effect is detached from its change event")
            intent, event = binding
            if intent.status is not DocumentIntentStatus.COMPLETED:
                raise RuntimeError("commit-group memory effect is detached from its completed intent")
            if canonical_digest(event.to_dict()) != effect.change_digest:
                raise RuntimeError("commit-group memory effect is detached from its change event")
            job = self.memory_committer.projection_queue.get(intent.projection_job_id)
            if (
                job is None
                or job.queue_name != "memory_projection"
                or job.action != "memory_committed"
                or job.payload.get("intent_id") != intent.intent_id
                or job.payload.get("document_id") != intent.document_id
                or job.payload.get("event_id") != intent.event_id
            ):
                raise RuntimeError("completed memory intent has no durable projection job")

    def _effect_from_document_result(self, result: DocumentCommitResult) -> MemoryDocumentEffect:
        if result.status is not DocumentIntentStatus.COMPLETED or result.event is None:
            raise RuntimeError("document committer did not complete its source and projection enqueue")
        if result.event.document_id == "":
            raise RuntimeError("document change event has no document identity")
        return MemoryDocumentEffect(
            document_id=result.event.document_id,
            change_event_id=result.event.event_id,
            change_digest=canonical_digest(result.event.to_dict()),
        )

    def _validate_document_plan(self, plan: DocumentEditPlan, archive: SessionArchive) -> None:
        if plan.tenant_id != self._tenant_id(archive) or plan.owner_user_id != archive.user_id:
            raise PermissionError("document plan crosses the archived Session boundary")
        if not self._is_sha256(plan.evidence_digest):
            raise ValueError("document plan evidence digest is invalid")
        if not plan.idempotency_key:
            raise ValueError("document plan idempotency key is empty")

    def _stabilize_operations(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: Sequence[ContextOperation],
    ) -> list[ContextOperation]:
        stable: list[ContextOperation] = []
        for position, operation in enumerate(operations):
            if not isinstance(operation, ContextOperation):
                raise TypeError(f"{consumer} planner returned a non-ContextOperation")
            if operation.user_id != archive.user_id:
                raise PermissionError(f"{consumer} operation crosses the Session owner boundary")
            operation_key = stable_hash(
                [
                    group_id,
                    consumer,
                    position,
                    operation.context_type.value,
                    operation.action.value,
                    operation.target_uri,
                ],
                40,
            )
            operation.operation_id = f"op_{operation_key}"
            operation.created_at = archive.created_at
            operation.payload["commit_group_id"] = group_id
            operation.payload["commit_consumer"] = consumer
            stable.append(operation)
        return stable

    def _enqueue_session_commit(self, archive: SessionArchive, *, tenant_id: str) -> QueueJob:
        return self.queue_store.enqueue(
            QueueJob(
                job_id=archive.task_id,
                queue_name="commit",
                action="async_session_commit",
                target_uri=archive.archive_uri,
                payload={
                    "user_id": archive.user_id,
                    "session_id": archive.session_id,
                    "tenant_id": tenant_id,
                    "archive_digest": archive.archive_digest,
                    "manifest_digest": archive.manifest_digest,
                },
            )
        )

    def _project_session_archive(
        self,
        archive: SessionArchive,
        *,
        respect_applied_tombstones: bool = False,
    ) -> tuple[Any | None, str]:
        if self.session_projector is None:
            return None, "not_configured"
        async_outputs: dict[str, Any] | None = None
        if self.archive_store.async_outputs_done_for_task(archive):
            async_outputs = self.archive_store.read_async_outputs(archive)
        elif str(getattr(self.archive_store, "last_async_output_error", "") or ""):
            raise RuntimeError("Session async outputs failed integrity validation")
        kwargs: dict[str, Any] = {}
        if async_outputs is not None:
            kwargs["async_outputs"] = async_outputs
        if respect_applied_tombstones:
            kwargs["respect_applied_tombstones"] = True
        return self.session_projector.project(archive, **kwargs), "projected"

    def _record_projection(
        self,
        archive: SessionArchive,
        *,
        tenant_id: str,
        status: str,
        error: str = "",
    ) -> bool:
        if self.session_projector is None or not self.projection_journal.enabled:
            return False
        return self.projection_journal.record(
            archive,
            tenant_id=tenant_id,
            status=status,
            error=error,
        )

    def _require_runtime_ready(self) -> None:
        committer = getattr(self.committer, "delegate", self.committer)
        source_store = getattr(committer, "source_store", None)
        if source_store is None:
            source_store = getattr(self.memory_planner, "source_store", None)
        readiness = getattr(source_store, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if not callable(require_ready) or self._startup_recovery_group.get():
            return
        require_ready()

    @contextmanager
    def _startup_recovery_scope(self, group_id: str) -> Iterator[None]:
        token = self._startup_recovery_group.set(group_id)
        committer = getattr(self.committer, "delegate", self.committer)
        scope = getattr(committer, "_durable_startup_recovery_scope", None)
        try:
            if callable(scope):
                with cast(AbstractContextManager[None], scope(group_id)):
                    yield
            else:
                yield
        finally:
            self._startup_recovery_group.reset(token)

    def _tenant_id(self, archive: SessionArchive) -> str:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        bound = str(self.archive_store.tenant_id)
        claimed = tuple(
            str(value) for value in (metadata.get("tenant_id"), scope.get("tenant_id")) if value not in (None, "")
        )
        if any(value != bound for value in claimed):
            raise PermissionError("SessionArchive tenant differs from its bound store")
        return bound

    def _bind_archive_tenant(self, archive: SessionArchive) -> str:
        tenant_id = self._tenant_id(archive)
        metadata = dict(archive.metadata or {})
        metadata["tenant_id"] = tenant_id
        if "scope" in metadata:
            scope = dict(metadata.get("scope", {}) or {})
            scope["tenant_id"] = tenant_id
            metadata["scope"] = scope
        archive.metadata = metadata
        return tenant_id

    @staticmethod
    def _assert_archive_identity(
        requested: SessionArchive,
        persisted: SessionArchive,
        *,
        allow_materialized_digests: bool = False,
    ) -> None:
        if (
            requested.task_id != persisted.task_id
            or requested.user_id != persisted.user_id
            or requested.session_id != persisted.session_id
            or requested.archive_uri != persisted.archive_uri
        ):
            raise RuntimeError("SessionArchive request is detached from durable evidence")
        for field in ("archive_digest", "manifest_digest"):
            expected = str(getattr(requested, field, "") or "")
            actual = str(getattr(persisted, field, "") or "")
            if expected and expected != actual:
                raise RuntimeError(f"SessionArchive {field} changed across durable read")
            if not expected and not allow_materialized_digests:
                raise RuntimeError(f"SessionArchive {field} is not materialized")

    @staticmethod
    def _actor_binding(archive: SessionArchive) -> str:
        return f"session:{archive.task_id}:{archive.user_id}"

    @staticmethod
    def _evidence_reference(archive: SessionArchive, position: int) -> str:
        manifest = archive.manifest_uri or f"{archive.archive_uri}#manifest={archive.manifest_digest}"
        return f"{manifest}:memory-edit:{position}"

    @staticmethod
    def _skipped_action(_attempt_id: str) -> dict[str, Any]:
        return {
            "status": "skipped",
            "operation_count": 0,
            "operation_ids": [],
            "diff_id": "",
            "skipped": True,
        }

    @staticmethod
    def _is_coding_agent(archive: SessionArchive) -> bool:
        connect = dict(archive.metadata.get("connect", {}) or {})
        return connect.get("connect_type") == "agent" and connect.get("run_mode") == "context_reduction"

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        explicit = getattr(exc, "retryable", None)
        if isinstance(explicit, bool):
            return explicit
        return isinstance(exc, (OSError, TimeoutError, RevisionConflictError, DocumentConflictError))

    @staticmethod
    def _is_sha256(value: str) -> bool:
        return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
