"""Session 提交入口：保证先归档原始会话，再执行或排队派生提交。"""

from __future__ import annotations

from typing import Protocol

from infrastructure.store.contracts.session_archive import SessionArchiveStore
from pre.session import SessionArchive
from runtime.session.commit_model import SessionCommitResult


class SessionCommitBackend(Protocol):
    archive_store: SessionArchiveStore

    def sync_archive(
        self,
        archive: SessionArchive,
        *,
        enqueue_commit_job: bool = True,
    ) -> SessionCommitResult: ...

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult: ...

    def enqueue_failed_inline_commit(self, archive: SessionArchive) -> object: ...


def commit_session(
    backend: SessionCommitBackend,
    archive: SessionArchive,
    *,
    async_commit: bool = True,
) -> SessionCommitResult:
    """保持稳定的同步执行或耐久排队契约。"""

    if not async_commit:
        return backend.sync_archive(archive, enqueue_commit_job=True)
    try:
        backend.sync_archive(archive, enqueue_commit_job=False)
        return backend.async_commit(archive)
    except Exception:
        archive_store = backend.archive_store
        if archive_store.archive_exists(
            archive.archive_uri,
            tenant_id=archive_store.tenant_id,
        ):
            try:
                backend.enqueue_failed_inline_commit(archive)
            except Exception as enqueue_error:
                raise RuntimeError(
                    "inline Session commit failed and its durable retry job could not be enqueued"
                ) from enqueue_error
        raise


__all__ = ["SessionCommitBackend", "commit_session"]
