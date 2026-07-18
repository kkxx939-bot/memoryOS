"""Narrow runtime surface consumed by worker orchestration."""

from __future__ import annotations

from typing import Any, Protocol


class WorkerRuntime(Protocol):
    root: str
    tenant_id: str
    recovery_worker: Any
    session_commit_service: Any
    memory_projection_worker: Any
    memory_document_edit_worker: Any
    memory_document_scan_worker: Any
    source_store: Any
    queue_store: Any
    vector_store: Any
    embedding_provider: Any
    context_db: Any
    readiness: Any


__all__ = ["WorkerRuntime"]
