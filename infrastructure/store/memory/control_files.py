"""控制记录仓储共享的安全路径与文件操作。"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.memory.control_common import (
    _MAX_CONTROL_BYTES,
    DocumentControlIntegrityError,
)
from infrastructure.store.memory.control_common import (
    is_hex as _is_hex,
)
from infrastructure.store.memory.control_common import (
    mapping as _mapping,
)
from infrastructure.store.memory.control_common import (
    validate_prefixed_digest as _validate_prefixed_digest,
)
from infrastructure.store.memory.control_intent import (
    DocumentCommitIntent,
)
from infrastructure.store.memory.layout import tenant_control_root
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


class ControlFileMixin:
    # 由 MemoryDocumentControlStore 在初始化时绑定。
    root: Path

    def _artifact_root(self, tenant_id: str) -> Path:
        return tenant_control_root(self.root, tenant_id)

    def _owner_root(self, tenant_id: str, owner_user_id: str) -> Path:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        return self._artifact_root(tenant) / "system" / "memory-documents" / owner

    def _intent_path(self, tenant_id: str, owner_user_id: str, intent_id: str) -> Path:
        _validate_prefixed_digest(intent_id, "mdintent_", "intent_id")
        return self._owner_root(tenant_id, owner_user_id) / "intents" / f"{intent_id}.json"

    def _conflict_path(self, tenant_id: str, owner_user_id: str, intent_id: str) -> Path:
        _validate_prefixed_digest(intent_id, "mdintent_", "intent_id")
        return self._owner_root(tenant_id, owner_user_id) / "conflicts" / f"{intent_id}.json"

    def _adoption_receipt_path(self, tenant_id: str, owner_user_id: str, receipt_id: str) -> Path:
        _validate_prefixed_digest(receipt_id, "mdadopt_", "receipt_id")
        return self._owner_root(tenant_id, owner_user_id) / "adoptions" / f"{receipt_id}.json"

    def _root_identity_path(self, tenant_id: str, owner_user_id: str) -> Path:
        return self._owner_root(tenant_id, owner_user_id) / "scan-root.json"

    def _bootstrap_path(self, tenant_id: str, owner_user_id: str) -> Path:
        return self._owner_root(tenant_id, owner_user_id) / "bootstrap.json"

    def _adoption_identity_path(self, tenant_id: str, owner_user_id: str, document_id: str) -> Path:
        identifier = validate_document_id(document_id)
        return self._owner_root(tenant_id, owner_user_id) / "adoption-identities" / f"{identifier}.json"

    def _event_path(self, intent: DocumentCommitIntent, event_id: str) -> Path:
        MemoryDocumentPathPolicy.trusted_segment(event_id, "event_id")
        return (
            self._owner_root(intent.tenant_id, intent.owner_user_id)
            / "events"
            / intent.document_id
            / f"{intent.logical_revision:020d}-{event_id}.json"
        )

    def _control_path(self, tenant_id: str, owner_user_id: str, document_id: str) -> Path:
        identifier = validate_document_id(document_id)
        return self._owner_root(tenant_id, owner_user_id) / "documents" / f"{identifier}.json"

    def _publication_barrier_path(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> Path:
        identifier = validate_document_id(document_id)
        return self._owner_root(tenant_id, owner_user_id) / "publication-barriers" / f"{identifier}.json"

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
                    raise DocumentControlIntegrityError("document control artifact is not one regular file")
                if metadata.st_size > _MAX_CONTROL_BYTES:
                    raise DocumentControlIntegrityError("document control artifact exceeds its size bound")
                chunks: list[bytes] = []
                remaining = _MAX_CONTROL_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                if len(raw) > _MAX_CONTROL_BYTES:
                    raise DocumentControlIntegrityError("document control artifact exceeds its size bound")
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        try:
            payload = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentControlIntegrityError("document control artifact is invalid JSON") from exc
        return _mapping(payload, "document control artifact")

    def _json_names(self, directory: Path, tenant_id: str) -> tuple[str, ...]:
        sentinel = directory / ".scan"
        descriptor = _open_control_parent(sentinel, self._artifact_root(tenant_id))
        try:
            names = tuple(
                name
                for name in os.listdir(descriptor)
                if name.endswith(".json") and name.startswith("mdintent_") and "/" not in name
            )
        finally:
            os.close(descriptor)
        return tuple(sorted(names))

    def _unlink_regular_if_present(self, path: Path, tenant_id: str) -> int:
        parent_descriptor = _open_control_parent(path, self._artifact_root(tenant_id))
        try:
            try:
                metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return 0
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DocumentControlIntegrityError("document purge encountered a non-regular artifact")
            os.unlink(path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            return 1
        finally:
            os.close(parent_descriptor)

    def _purge_event_directory(self, tenant_id: str, owner_user_id: str, document_id: str) -> int:
        directory = self._owner_root(tenant_id, owner_user_id) / "events" / document_id
        if not directory.exists():
            return 0
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant_id))
        removed = 0
        try:
            for name in os.listdir(descriptor):
                prefix, separator, suffix = name.partition("-")
                event_id = suffix.removesuffix(".json")
                if (
                    not separator
                    or len(prefix) != 20
                    or not prefix.isdigit()
                    or not name.endswith(".json")
                    or not event_id.startswith("memchg_")
                    or not _is_hex(event_id.removeprefix("memchg_"), 64)
                ):
                    raise DocumentControlIntegrityError("document purge encountered an unexpected event artifact")
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise DocumentControlIntegrityError("document purge encountered a non-regular event artifact")
                os.unlink(name, dir_fd=descriptor)
                removed += 1
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = _open_control_parent(directory, self._artifact_root(tenant_id))
        try:
            try:
                os.rmdir(directory.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return removed

