"""多文档合并 Saga 进度的文件存储实现。"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from infrastructure.store.filesystem.durable_io import atomic_create_json, atomic_write_json
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.memory.layout import tenant_control_root
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.consolidation import (
    _MAX_SAGA_BYTES,
    _MAX_SAGAS_PER_OWNER,
    ConsolidationIntegrityError,
    ConsolidationSagaRecord,
    ConsolidationStatus,
    _bounded_list_limit,
    _mapping,
    _status_rank,
    _validate_prefixed_digest,
)


class MemoryDocumentConsolidationStore:
    """保存不含正文且可在崩溃后恢复的合并进度日志。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def create(self, record: ConsolidationSagaRecord) -> ConsolidationSagaRecord:
        atomic_create_json(
            self._record_path(record.tenant_id, record.owner_user_id, record.saga_id),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )
        durable = self.load(record.tenant_id, record.owner_user_id, record.saga_id)
        if durable is None:  # pragma: no cover - 仅创建发布的日志不会协作式消失。
            raise ConsolidationIntegrityError("consolidation journal disappeared after creation")
        return durable

    def load(
        self,
        tenant_id: str,
        owner_user_id: str,
        saga_id: str,
    ) -> ConsolidationSagaRecord | None:
        _validate_prefixed_digest(saga_id, "memsaga_", "saga_id")
        payload = self._read_json(self._record_path(tenant_id, owner_user_id, saga_id), tenant_id)
        if payload is None:
            return None
        record = ConsolidationSagaRecord.from_dict(payload)
        if (record.tenant_id, record.owner_user_id, record.saga_id) != (
            tenant_id,
            owner_user_id,
            saga_id,
        ):
            raise ConsolidationIntegrityError("consolidation path identity differs from its journal")
        return record

    def save(self, record: ConsolidationSagaRecord) -> ConsolidationSagaRecord:
        current = self.load(record.tenant_id, record.owner_user_id, record.saga_id)
        if current is None or current.identity_digest != record.identity_digest:
            raise ConsolidationIntegrityError("consolidation update is detached from its immutable identity")
        if record.next_source_index < current.next_source_index:
            raise ConsolidationIntegrityError("consolidation source cursor cannot move backward")
        if (
            current.target_projection_generation
            and record.target_projection_generation != current.target_projection_generation
        ):
            raise ConsolidationIntegrityError("consolidation target generation cannot change after commit")
        if _status_rank(record.status) < _status_rank(current.status):
            raise ConsolidationIntegrityError("consolidation status cannot move backward")
        atomic_write_json(
            self._record_path(record.tenant_id, record.owner_user_id, record.saga_id),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )
        return record

    def list_records(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = 1_000,
    ) -> tuple[ConsolidationSagaRecord, ...]:
        """返回经过身份校验且数量受限的所有者 Saga 快照。"""

        names = self._record_names(tenant_id, owner_user_id)
        if len(names) > _bounded_list_limit(limit):
            raise ConsolidationIntegrityError("consolidation journal count exceeds the requested bound")
        records: list[ConsolidationSagaRecord] = []
        for name in names:
            record = self.load(tenant_id, owner_user_id, name.removesuffix(".json"))
            if record is None:  # pragma: no cover - 启动阶段不存在并发日志清理器。
                raise ConsolidationIntegrityError("consolidation journal disappeared during bounded listing")
            records.append(record)
        return tuple(records)

    def list_pending(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = 1_000,
    ) -> tuple[ConsolidationSagaRecord, ...]:
        """返回未完成 Saga，超过恢复上限时显式失败。"""

        maximum = _bounded_list_limit(limit)
        pending = tuple(
            record
            for record in self.list_records(
                tenant_id,
                owner_user_id,
                limit=_MAX_SAGAS_PER_OWNER,
            )
            if record.status != ConsolidationStatus.COMPLETED
        )
        if len(pending) > maximum:
            raise ConsolidationIntegrityError("pending consolidation count exceeds the recovery bound")
        return pending

    @contextmanager
    def lock(self, tenant_id: str, owner_user_id: str, saga_id: str) -> Iterator[None]:
        artifact_root = self._artifact_root(tenant_id)
        lock_path = self._owner_root(tenant_id, owner_user_id) / "locks" / f"{saga_id}.lock"
        descriptor = open_private_lock(lock_path, root=artifact_root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _artifact_root(self, tenant_id: str) -> Path:
        return tenant_control_root(self.root, tenant_id)

    def _owner_root(self, tenant_id: str, owner_user_id: str) -> Path:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        return self._artifact_root(tenant) / "system" / "memory-documents" / owner / "consolidations"

    def _record_path(self, tenant_id: str, owner_user_id: str, saga_id: str) -> Path:
        _validate_prefixed_digest(saga_id, "memsaga_", "saga_id")
        return self._owner_root(tenant_id, owner_user_id) / f"{saga_id}.json"

    def _record_names(self, tenant_id: str, owner_user_id: str) -> tuple[str, ...]:
        directory = self._owner_root(tenant_id, owner_user_id)
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant_id))
        try:
            names = tuple(sorted(name for name in os.listdir(descriptor) if name.endswith(".json")))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_SAGAS_PER_OWNER:
            raise ConsolidationIntegrityError("consolidation journal count exceeds its hard bound")
        for name in names:
            try:
                _validate_prefixed_digest(name.removesuffix(".json"), "memsaga_", "saga_id")
            except ValueError as exc:
                raise ConsolidationIntegrityError(
                    "consolidation directory contains an unexpected JSON artifact"
                ) from exc
        return names

    def _read_json(self, path: Path, tenant_id: str) -> dict[str, object] | None:
        parent_descriptor = _open_control_parent(path, self._artifact_root(tenant_id))
        try:
            try:
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                return None
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ConsolidationIntegrityError("consolidation journal is not one regular file")
                if metadata.st_size > _MAX_SAGA_BYTES:
                    raise ConsolidationIntegrityError("consolidation journal exceeds its size bound")
                raw = b""
                while len(raw) <= _MAX_SAGA_BYTES:
                    chunk = os.read(descriptor, min(65536, _MAX_SAGA_BYTES + 1 - len(raw)))
                    if not chunk:
                        break
                    raw += chunk
                if len(raw) > _MAX_SAGA_BYTES:
                    raise ConsolidationIntegrityError("consolidation journal exceeds its size bound")
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        try:
            decoded = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConsolidationIntegrityError("consolidation journal is invalid JSON") from exc
        return _mapping(decoded)


__all__ = ["MemoryDocumentConsolidationStore"]
