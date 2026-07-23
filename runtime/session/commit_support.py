"""Session 提交的持久结果校验、排队和租户绑定。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, cast

from foundation.ids import stable_hash
from infrastructure.store.contracts.queue import QueueJob
from pre.session import SessionArchive
from runtime.session.commit_consumers import _SessionCommitConsumers
from transaction.model.context_operation import ContextOperation


class _SessionCommitSupport(_SessionCommitConsumers):
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
            raise RuntimeError("SessionArchive request is detached from the durable archive")
        for field in ("archive_digest", "manifest_digest"):
            expected = str(getattr(requested, field, "") or "")
            actual = str(getattr(persisted, field, "") or "")
            if expected and expected != actual:
                raise RuntimeError(f"SessionArchive {field} changed across durable read")
            if not expected and not allow_materialized_digests:
                raise RuntimeError(f"SessionArchive {field} is not materialized")

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
        return isinstance(exc, OSError | TimeoutError)
