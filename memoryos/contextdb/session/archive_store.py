"""ContextDB-owned protocol for durable SessionArchive evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from memoryos.contextdb.session.session_model import SessionArchive


class SessionArchiveStore(Protocol):
    root: Path
    tenant_id: str

    def write_sync_archive(self, archive: SessionArchive) -> Path: ...

    def write_async_outputs(
        self,
        archive_uri: str,
        abstract: str,
        overview: str,
        memory_diff: dict,
        behavior_diff: dict,
        action_policy_diff: dict,
        context_diff: dict,
        tenant_id: str | None = None,
        commit_group_status: dict[str, Any] | None = None,
        complete: bool = True,
        task_id: str | None = None,
        created_at: str | None = None,
    ) -> Path: ...

    def async_outputs_done_for_task(self, archive: SessionArchive) -> bool: ...

    def read_async_outputs(self, archive: SessionArchive) -> dict[str, Any]: ...

    def read_archive(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        manifest_digest: str | None = None,
    ) -> SessionArchive: ...

    def read_archive_from_commit_head(
        self,
        head_path: Path,
        *,
        tenant_id: str,
        user_id: str,
    ) -> SessionArchive: ...

    def list_archives(
        self,
        *,
        tenant_id: str | None = None,
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> tuple[SessionArchive, ...]: ...

    def read_archive_at_manifest(
        self,
        archive_uri: str,
        manifest_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> SessionArchive: ...

    def archive_exists(self, archive_uri: str, *, tenant_id: str | None = None) -> bool: ...

    def archive_tenant(self, archive: SessionArchive) -> str: ...

    def read_event(
        self,
        archive_uri: str,
        event_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]: ...

    def current_manifest(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]: ...


__all__ = ["SessionArchiveStore"]
