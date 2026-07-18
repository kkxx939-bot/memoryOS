"""Storage protocol and typed failures for Markdown memory source bytes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from memoryos.memory.documents.model import MemoryDocument, RawPathState, ScanGeneration


class MemoryDocumentStoreError(RuntimeError):
    pass


class DocumentConflictError(MemoryDocumentStoreError):
    pass


class DocumentNotFoundError(MemoryDocumentStoreError):
    pass


class DocumentUnsafeError(MemoryDocumentStoreError):
    pass


class MemoryDocumentStore(Protocol):
    def read_state(self, tenant_id: str, owner_user_id: str, relative_path: str) -> RawPathState: ...

    def read_raw(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        document_id: str = "",
        relative_path: str = "",
    ) -> bytes: ...

    def full_scan(self, tenant_id: str, owner_user_id: str) -> ScanGeneration: ...

    def seed_registration(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        relative_path: str,
    ) -> None: ...

    def cleanup_operation_temps(
        self,
        tenant_id: str,
        owner_user_id: str,
        expected_raw_sha256_by_path: Mapping[str, str],
        operation_id: str,
    ) -> int: ...

    def create(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        after_bytes: bytes,
        *,
        expected: RawPathState,
        operation_id: str = "",
        fault_hook: Callable[[str], None] | None = None,
    ) -> MemoryDocument: ...

    def replace(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        after_bytes: bytes,
        *,
        expected_state: RawPathState,
        operation_id: str = "",
        fault_hook: Callable[[str], None] | None = None,
    ) -> MemoryDocument: ...

    def delete(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        *,
        expected_state: RawPathState,
        operation_id: str = "",
        fault_hook: Callable[[str], None] | None = None,
    ) -> RawPathState: ...

    def rename(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        new_relative_path: str,
        *,
        expected_old: RawPathState,
        expected_new: RawPathState,
        after_bytes: bytes | None = None,
        operation_id: str = "",
        fault_hook: Callable[[str], None] | None = None,
    ) -> MemoryDocument: ...

    def adopt(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        *,
        expected_raw_sha256: str,
        assigned_document_id: str | None = None,
        operation_id: str = "",
        fault_hook: Callable[[str], None] | None = None,
    ) -> MemoryDocument: ...


__all__ = [
    "DocumentConflictError",
    "DocumentNotFoundError",
    "DocumentUnsafeError",
    "MemoryDocumentStore",
    "MemoryDocumentStoreError",
]
