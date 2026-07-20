"""SessionArchive 到 Context Catalog 投影的崩溃恢复前沿。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infrastructure.context.session_projector import workspace_id_from_session_metadata
from pre.session import SessionArchive


@dataclass(frozen=True)
class SessionProjectionJournalEntry:
    tenant_id: str
    archive_uri: str
    owner_user_id: str
    workspace_id: str
    session_id: str
    manifest_digest: str
    status: str
    error: str = ""


class SessionProjectionJournal:
    """对 Session 投影耐久前沿表提供强类型访问。"""

    def __init__(self, store: Any | None) -> None:
        self.store = store

    @property
    def enabled(self) -> bool:
        return callable(getattr(self.store, "set_session_projection_frontier", None))

    def record(self, archive: SessionArchive, *, tenant_id: str, status: str, error: str = "") -> bool:
        if status not in {"PENDING", "PROJECTED", "FAILED", "ABANDONED"}:
            raise ValueError("Session projection journal status is invalid")
        recorder = getattr(self.store, "set_session_projection_frontier", None)
        if not callable(recorder):
            return False
        recorder(
            tenant_id=tenant_id,
            archive_uri=archive.archive_uri,
            owner_user_id=archive.user_id,
            workspace_id=workspace_id_from_session_metadata(archive.metadata),
            session_id=archive.session_id,
            manifest_digest=str(archive.manifest_digest or ""),
            status=status,
            error=str(error)[:500],
        )
        return True

    def pending(
        self,
        *,
        tenant_id: str,
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> tuple[SessionProjectionJournalEntry, ...]:
        lister = getattr(self.store, "list_session_projection_frontier", None)
        if not callable(lister):
            return ()
        rows = lister(
            tenant_id=tenant_id,
            statuses=("PENDING", "FAILED"),
            after_archive_uri=after_archive_uri,
            limit=max(1, min(int(limit), 1_000)),
        )
        if not isinstance(rows, list):
            raise RuntimeError("Session projection journal returned an invalid row collection")
        return tuple(self._entry(row, tenant_id=tenant_id) for row in rows)

    def mark(
        self,
        entry: SessionProjectionJournalEntry,
        *,
        status: str,
        error: str = "",
    ) -> None:
        if status not in {"PROJECTED", "FAILED", "ABANDONED"}:
            raise ValueError("Session projection journal terminal status is invalid")
        recorder = getattr(self.store, "set_session_projection_frontier", None)
        if not callable(recorder):
            raise RuntimeError("Session projection journal is not configured")
        recorder(
            tenant_id=entry.tenant_id,
            archive_uri=entry.archive_uri,
            owner_user_id=entry.owner_user_id,
            workspace_id=entry.workspace_id,
            session_id=entry.session_id,
            manifest_digest=entry.manifest_digest,
            status=status,
            error=str(error)[:500],
        )

    @staticmethod
    def _entry(value: object, *, tenant_id: str) -> SessionProjectionJournalEntry:
        if not isinstance(value, dict):
            raise RuntimeError("Session projection journal row is invalid")
        row_tenant = str(value.get("tenant_id") or tenant_id)
        if row_tenant != tenant_id:
            raise RuntimeError("Session projection journal row crosses its tenant boundary")
        return SessionProjectionJournalEntry(
            tenant_id=row_tenant,
            archive_uri=str(value.get("source_uri") or ""),
            owner_user_id=str(value.get("owner_user_id") or ""),
            workspace_id=str(value.get("workspace_id") or ""),
            session_id=str(value.get("source_id") or ""),
            manifest_digest=str(value.get("source_digest") or ""),
            status=str(value.get("status") or ""),
            error=str(value.get("last_error") or ""),
        )


__all__ = ["SessionProjectionJournal", "SessionProjectionJournalEntry"]
