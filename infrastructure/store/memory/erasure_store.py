"""硬删除纪元、恢复游标和复活屏障的文件存储实现。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from foundation.integrity import canonical_json
from infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_json,
    atomic_write_json,
)
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.memory.layout import tenant_control_root
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.erase import (
    _MAX_ERASE_RECORD_BYTES,
    _MAX_ERASE_RECORDS_PER_OWNER,
    _MAX_INDEPENDENT_EVIDENCE_REFERENCES,
    DocumentEraseConflict,
    DocumentErasedError,
    DocumentEraseIntegrityError,
    DocumentEraseRecord,
    DocumentEraseStatus,
    EraseBackendProgress,
    _bounded_reference,
    _mapping,
    _path_digest,
    _validate_backend_name,
)


class MemoryDocumentEraseStore:
    """保存不含正文的删除纪元，用于拒绝陈旧任务复活文档。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def load(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentEraseRecord | None:
        path = self._record_path(tenant_id, owner_user_id, document_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        record = DocumentEraseRecord.from_dict(payload)
        if (record.tenant_id, record.owner_user_id, record.document_id) != (
            tenant_id,
            owner_user_id,
            document_id,
        ):
            raise DocumentEraseIntegrityError("erasure path identity does not match its payload")
        return record

    def records(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = _MAX_ERASE_RECORDS_PER_OWNER,
    ) -> tuple[DocumentEraseRecord, ...]:
        """在数量上限内枚举一个所有者的耐久删除纪元。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        maximum = int(limit)
        if maximum <= 0 or maximum > _MAX_ERASE_RECORDS_PER_OWNER:
            raise ValueError("erasure recovery record limit is invalid")
        directory = self._artifact_root(tenant) / "system" / "memory-documents" / owner / "erasures"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > maximum:
            raise DocumentEraseIntegrityError("owner erasure record count exceeds its recovery bound")
        records: list[DocumentEraseRecord] = []
        for name in names:
            if not name.endswith(".json") or "/" in name:
                raise DocumentEraseIntegrityError("erasure directory contains an unexpected artifact")
            try:
                document_id = validate_document_id(name.removesuffix(".json"))
            except ValueError as exc:
                raise DocumentEraseIntegrityError("erasure record filename is invalid") from exc
            record = self.load(tenant, owner, document_id)
            if record is None:  # pragma: no cover - 协作式扫描不会丢失耐久记录。
                raise DocumentEraseIntegrityError("erasure record disappeared during recovery scan")
            records.append(record)
        return tuple(records)

    def begin(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        relative_path: str,
        source_digest: str,
        document_revision_floor: int,
        projection_generation_floor: int,
        backend_names: Sequence[str],
        independent_evidence_retained: Sequence[str],
        started_at: str,
    ) -> DocumentEraseRecord:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        evidence = tuple(sorted({_bounded_reference(item) for item in independent_evidence_retained}))
        if len(evidence) > _MAX_INDEPENDENT_EVIDENCE_REFERENCES:
            raise ValueError("independent evidence reference count exceeds its bound")
        names = tuple(dict.fromkeys(backend_names))
        for name in names:
            _validate_backend_name(name)
        epoch_digest = hashlib.sha256(
            canonical_json(
                ["memory_document_erasure_epoch_v2", tenant, owner, identifier, source_digest, started_at]
            ).encode()
        ).hexdigest()
        record = DocumentEraseRecord(
            tenant_id=tenant,
            owner_user_id=owner,
            document_id=identifier,
            relative_path=relative,
            relative_path_digest=_path_digest(relative),
            document_kind=MemoryDocumentPathPolicy.kind_for(relative).value,
            erasure_epoch=f"erase_{epoch_digest}",
            source_digest=source_digest,
            document_revision_floor=document_revision_floor,
            projection_generation_floor=projection_generation_floor,
            status=DocumentEraseStatus.ERASING,
            backends=tuple(EraseBackendProgress(name) for name in names),
            independent_evidence_retained=evidence,
            started_at=started_at,
            updated_at=started_at,
        )
        path = self._record_path(tenant, owner, identifier)
        try:
            atomic_create_json(path, record.to_dict(), artifact_root=self._artifact_root(tenant))
        except ImmutableArtifactConflictError:
            pass
        durable = self.load(tenant, owner, identifier)
        if durable is None:
            raise DocumentEraseIntegrityError("erasure intent disappeared after durable publication")
        if durable.source_digest != source_digest:
            raise DocumentEraseConflict("document already has an erasure epoch for another source digest")
        if durable.independent_evidence_retained != evidence:
            raise DocumentEraseConflict("document erasure retry changed independent evidence disclosure")
        return durable

    def merge_backends(
        self,
        record: DocumentEraseRecord,
        backend_names: Sequence[str],
        *,
        updated_at: str,
    ) -> DocumentEraseRecord:
        known = {backend.backend_name for backend in record.backends}
        additions = []
        for name in backend_names:
            _validate_backend_name(name)
            if name not in known:
                additions.append(EraseBackendProgress(name))
                known.add(name)
        if not additions:
            return record
        if record.status == DocumentEraseStatus.ERASED:
            record = replace(record, status=DocumentEraseStatus.ERASE_PENDING, completed_at="")
        return self.write(replace(record, backends=record.backends + tuple(additions), updated_at=updated_at))

    def record_attempt(
        self,
        record: DocumentEraseRecord,
        backend_name: str,
        *,
        acknowledged: bool,
        attempted_at: str,
        failure_code: str = "",
    ) -> DocumentEraseRecord:
        updated_backends: list[EraseBackendProgress] = []
        found = False
        for backend in record.backends:
            if backend.backend_name != backend_name:
                updated_backends.append(backend)
                continue
            found = True
            updated_backends.append(
                replace(
                    backend,
                    acknowledged=backend.acknowledged or acknowledged,
                    attempt_count=backend.attempt_count + (0 if backend.acknowledged else 1),
                    last_attempt_at=backend.last_attempt_at if backend.acknowledged else attempted_at,
                    failure_code="" if backend.acknowledged or acknowledged else failure_code,
                )
            )
        if not found:
            raise DocumentEraseIntegrityError("erasure attempt named an unsealed backend")
        status = record.status
        if not acknowledged:
            status = DocumentEraseStatus.ERASE_PENDING
        return self.write(
            replace(
                record,
                backends=tuple(updated_backends),
                status=status,
                updated_at=attempted_at,
            )
        )

    def finish(self, record: DocumentEraseRecord, *, completed_at: str) -> DocumentEraseRecord:
        if record.pending_backends:
            return self.write(
                replace(
                    record,
                    status=DocumentEraseStatus.ERASE_PENDING,
                    updated_at=completed_at,
                    completed_at="",
                )
            )
        return self.write(
            replace(
                record,
                status=DocumentEraseStatus.ERASED,
                relative_path="",
                updated_at=completed_at,
                completed_at=completed_at,
            )
        )

    def write(self, record: DocumentEraseRecord) -> DocumentEraseRecord:
        current = self.load(record.tenant_id, record.owner_user_id, record.document_id)
        if current is None or current.erasure_epoch != record.erasure_epoch:
            raise DocumentEraseIntegrityError("erasure update is detached from its durable epoch")
        atomic_write_json(
            self._record_path(record.tenant_id, record.owner_user_id, record.document_id),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )
        return record

    def raise_floors(
        self,
        record: DocumentEraseRecord,
        *,
        document_revision_floor: int,
        projection_generation_floor: int,
        updated_at: str,
    ) -> DocumentEraseRecord:
        """在不改变删除纪元的前提下，耐久提高修订和投影水位。"""

        revision_floor = max(record.document_revision_floor, int(document_revision_floor))
        projection_floor = max(record.projection_generation_floor, int(projection_generation_floor))
        if revision_floor == record.document_revision_floor and projection_floor == record.projection_generation_floor:
            return record
        return self.write(
            replace(
                record,
                document_revision_floor=revision_floor,
                projection_generation_floor=projection_floor,
                updated_at=updated_at,
            )
        )

    def assert_mutation_allowed(self, tenant_id: str, owner_user_id: str, document_id: str) -> None:
        record = self.load(tenant_id, owner_user_id, document_id)
        if record is not None:
            raise DocumentErasedError(f"document identity is blocked by durable erasure epoch {record.erasure_epoch}")

    def assert_projection_allowed(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        *,
        projection_generation: int,
    ) -> None:
        record = self.load(tenant_id, owner_user_id, document_id)
        if record is not None:
            raise DocumentErasedError(
                "projection is rejected by a durable erasure epoch "
                f"at generation {record.projection_generation_floor}; received {projection_generation}"
            )

    def document_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> _LockedDocument:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        artifact_root = self._artifact_root(tenant)
        lock_path = artifact_root / "system" / "memory-documents" / owner / "locks" / f"{identifier}.lock"
        return _LockedDocument(lock_path, artifact_root)

    def owner_relation_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> _LockedDocument:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        artifact_root = self._artifact_root(tenant)
        lock_path = artifact_root / "system" / "memory-documents" / owner / "locks" / "relation-projection.lock"
        return _LockedDocument(lock_path, artifact_root)

    def _artifact_root(self, tenant_id: str) -> Path:
        return tenant_control_root(self.root, tenant_id)

    def _record_path(self, tenant_id: str, owner_user_id: str, document_id: str) -> Path:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        return self._artifact_root(tenant) / "system" / "memory-documents" / owner / "erasures" / f"{identifier}.json"

    def _read_json(self, path: Path, tenant_id: str) -> dict[str, Any] | None:
        parent_descriptor = _open_control_parent(path, self._artifact_root(tenant_id))
        try:
            try:
                descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
            except FileNotFoundError:
                return None
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise DocumentEraseIntegrityError("erasure record is not one regular file")
                if metadata.st_size > _MAX_ERASE_RECORD_BYTES:
                    raise DocumentEraseIntegrityError("erasure record exceeds its size bound")
                raw = _read_bounded(descriptor, _MAX_ERASE_RECORD_BYTES)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        if len(raw) > _MAX_ERASE_RECORD_BYTES:
            raise DocumentEraseIntegrityError("erasure record exceeds its size bound")
        try:
            payload = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentEraseIntegrityError("erasure record is invalid JSON") from exc
        return _mapping(payload)


class _LockedDocument:
    def __init__(self, lock_path: Path, artifact_root: Path) -> None:
        self.lock_path = lock_path
        self.artifact_root = artifact_root
        self.descriptor: int | None = None

    def __enter__(self) -> None:
        descriptor = open_private_lock(self.lock_path, root=self.artifact_root)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        self.descriptor = descriptor

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.descriptor is not None
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None


def _read_bounded(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = ["MemoryDocumentEraseStore"]
