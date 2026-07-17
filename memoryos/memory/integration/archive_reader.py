"""Memory-owned protocol for immutable session evidence reads.

The memory commit path needs to verify evidence that was materialized by the
session archive adapter.  It owns this narrow read contract; filesystem
construction remains an adapter/runtime concern.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from memoryos.contextdb.session.session_model import SessionArchive


class SessionEvidenceArchiveReader(Protocol):
    """Read only the immutable evidence needed by memory commit validation."""

    def read_archive(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        manifest_digest: str | None = None,
    ) -> SessionArchive: ...

    def current_manifest(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]: ...

    def read_event(
        self,
        archive_uri: str,
        event_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]: ...


SessionEvidenceArchiveReaderFactory = Callable[
    [str | Path, str],
    SessionEvidenceArchiveReader,
]

_reader_factory: SessionEvidenceArchiveReaderFactory | None = None


def register_session_evidence_archive_reader_factory(
    factory: SessionEvidenceArchiveReaderFactory,
) -> None:
    """Register an adapter factory at an explicit composition boundary."""

    global _reader_factory
    _reader_factory = factory


def session_evidence_archive_reader(
    root: str | Path,
    tenant_id: str,
) -> SessionEvidenceArchiveReader:
    """Create the registered reader, failing clearly when none is configured."""

    factory = _reader_factory
    if factory is None:
        raise RuntimeError("Session evidence archive reader is not registered")
    return factory(root, tenant_id)


__all__ = [
    "SessionEvidenceArchiveReader",
    "SessionEvidenceArchiveReaderFactory",
    "register_session_evidence_archive_reader_factory",
    "session_evidence_archive_reader",
]
