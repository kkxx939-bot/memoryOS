"""Markdown 记忆文档不可变的精确字节历史。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrastructure.store.filesystem.durable_io import atomic_create_bytes, atomic_create_json
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.memory.control_store import DocumentCommitIntent
from infrastructure.store.memory.layout import tenant_control_root
from memory.core.model import AbsentPath, DocumentChangeEvent, DocumentEditKind, PresentPath
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy

_REVISION_SCHEMA = "memory_document_revision_v1"
_MAX_REVISION_METADATA_BYTES = 256 * 1024


class DocumentRevisionIntegrityError(RuntimeError):
    """修订记录或不可变内容 Blob 校验失败。"""


@dataclass(frozen=True)
class DocumentRevisionRecord:
    tenant_id: str
    owner_user_id: str
    document_id: str
    logical_revision: int
    projection_generation: int
    event_id: str
    edit_kind: DocumentEditKind
    relative_path: str
    state: str
    raw_sha256: str
    size: int
    content_blob_digest: str
    content_blob_role: str
    created_at: str

    def __post_init__(self) -> None:
        MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        validate_document_id(self.document_id)
        if self.logical_revision <= 0 or self.projection_generation <= 0:
            raise ValueError("revision generations must be positive")
        if self.relative_path:
            MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        if self.state not in {"PRESENT", "ABSENT"}:
            raise ValueError("revision state must be PRESENT or ABSENT")
        if self.raw_sha256 and not _is_sha256(self.raw_sha256):
            raise ValueError("revision raw digest is invalid")
        if self.content_blob_digest and not _is_sha256(self.content_blob_digest):
            raise ValueError("revision content blob digest is invalid")
        if self.content_blob_role not in {"after", "before_delete", ""}:
            raise ValueError("revision content blob role is invalid")
        if bool(self.content_blob_digest) != bool(self.content_blob_role):
            raise ValueError("revision content blob digest and role must be present together")
        if self.size < 0 or not self.event_id or not self.created_at:
            raise ValueError("revision metadata is invalid")
        if not self.event_id.startswith("memchg_") or not _is_sha256(self.event_id.removeprefix("memchg_")):
            raise ValueError("revision event ID is invalid")
        if self.state == "PRESENT" and (not self.relative_path or not self.raw_sha256):
            raise ValueError("present revision requires a path and digest")
        if self.state == "PRESENT" and self.content_blob_digest != self.raw_sha256:
            raise ValueError("present revision content must exactly match its raw state")
        if self.content_blob_role == "after" and self.state != "PRESENT":
            raise ValueError("an after revision blob requires a PRESENT state")
        if self.content_blob_role == "before_delete" and (
            self.state != "ABSENT" or self.edit_kind != DocumentEditKind.DELETE
        ):
            raise ValueError("a before-delete revision blob requires an ABSENT DELETE revision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _REVISION_SCHEMA,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "document_id": self.document_id,
            "logical_revision": self.logical_revision,
            "projection_generation": self.projection_generation,
            "event_id": self.event_id,
            "edit_kind": self.edit_kind.value,
            "relative_path": self.relative_path,
            "state": self.state,
            "raw_sha256": self.raw_sha256,
            "size": self.size,
            "content_blob_digest": self.content_blob_digest,
            "content_blob_role": self.content_blob_role,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentRevisionRecord:
        if payload.get("schema") != _REVISION_SCHEMA:
            raise DocumentRevisionIntegrityError("document revision schema is unsupported")
        try:
            return cls(
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                logical_revision=int(payload["logical_revision"]),
                projection_generation=int(payload["projection_generation"]),
                event_id=str(payload["event_id"]),
                edit_kind=DocumentEditKind(str(payload["edit_kind"])),
                relative_path=str(payload.get("relative_path") or ""),
                state=str(payload["state"]),
                raw_sha256=str(payload.get("raw_sha256") or ""),
                size=int(payload["size"]),
                content_blob_digest=str(payload.get("content_blob_digest") or ""),
                content_blob_role=str(payload.get("content_blob_role") or ""),
                created_at=str(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentRevisionIntegrityError("document revision metadata is malformed") from exc


class MemoryDocumentRevisionStore:
    """受保护的不可变 Blob，以及不含正文的修订清单。"""

    def __init__(self, root: str | Path, *, max_blob_bytes: int = 2 * 1024 * 1024) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        if max_blob_bytes <= 0:
            raise ValueError("revision blob size limit must be positive")
        self.max_blob_bytes = max_blob_bytes

    def stage_blob(self, tenant_id: str, owner_user_id: str, document_id: str, raw_bytes: bytes) -> str:
        raw = bytes(raw_bytes)
        if len(raw) > self.max_blob_bytes:
            raise ValueError("revision content exceeds its configured byte limit")
        digest = hashlib.sha256(raw).hexdigest()
        path = self._blob_path(tenant_id, owner_user_id, document_id, digest)
        atomic_create_bytes(path, raw, artifact_root=self._artifact_root(tenant_id))
        return digest

    def read_blob(self, tenant_id: str, owner_user_id: str, document_id: str, digest: str) -> bytes:
        path = self._blob_path(tenant_id, owner_user_id, document_id, digest)
        parent_descriptor = _open_control_parent(path, self._artifact_root(tenant_id))
        try:
            try:
                descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
            except FileNotFoundError as exc:
                raise DocumentRevisionIntegrityError("revision content blob is missing") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise DocumentRevisionIntegrityError("revision content blob is not one regular file")
                if metadata.st_size > self.max_blob_bytes:
                    raise DocumentRevisionIntegrityError("revision content blob exceeds its size bound")
                chunks: list[bytes] = []
                remaining = self.max_blob_bytes + 1
                while remaining:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        if len(raw) > self.max_blob_bytes or hashlib.sha256(raw).hexdigest() != digest:
            raise DocumentRevisionIntegrityError("revision content blob digest does not match its key")
        return raw

    def record_revision(
        self,
        intent: DocumentCommitIntent,
        event: DocumentChangeEvent,
    ) -> DocumentRevisionRecord:
        if (
            event.event_id != intent.event_id
            or event.document_id != intent.document_id
            or event.logical_revision != intent.logical_revision
        ):
            raise ValueError("revision event is detached from its prepared intent")
        final_effect = intent.effects[-1]
        if intent.edit_kind == DocumentEditKind.RENAME:
            final_effect = next(effect for effect in intent.effects if effect.relative_path == intent.new_relative_path)
        if isinstance(final_effect.after, PresentPath):
            state = "PRESENT"
            relative_path = final_effect.relative_path
            raw_sha256 = final_effect.after.raw_sha256
            size = final_effect.after.size
        elif isinstance(final_effect.after, AbsentPath):
            state = "ABSENT"
            relative_path = intent.old_relative_path
            raw_sha256 = ""
            size = 0
        else:  # pragma: no cover - DocumentPathEffect already rejects UNSAFE.
            raise ValueError("revision cannot persist an unsafe after state")
        record = DocumentRevisionRecord(
            tenant_id=intent.tenant_id,
            owner_user_id=intent.owner_user_id,
            document_id=intent.document_id,
            logical_revision=intent.logical_revision,
            projection_generation=intent.projection_generation,
            event_id=intent.event_id,
            edit_kind=intent.edit_kind,
            relative_path=relative_path,
            state=state,
            raw_sha256=raw_sha256,
            size=size,
            content_blob_digest=intent.revision_blob_digest,
            content_blob_role=intent.revision_blob_role,
            created_at=event.occurred_at,
        )
        if record.content_blob_digest:
            raw = self.read_blob(
                record.tenant_id,
                record.owner_user_id,
                record.document_id,
                record.content_blob_digest,
            )
            expected_blob_size = size
            if record.content_blob_role == "before_delete":
                before = intent.effects[0].before
                if not isinstance(before, PresentPath):  # pragma: no cover - intent validation owns this.
                    raise ValueError("DELETE revision has no exact PRESENT before state")
                expected_blob_size = before.size
            if len(raw) != expected_blob_size:
                raise DocumentRevisionIntegrityError("revision content blob size does not match its manifest")
        atomic_create_json(
            self._revision_path(
                record.tenant_id,
                record.owner_user_id,
                record.document_id,
                record.logical_revision,
            ),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )
        return record

    def load_revision(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        logical_revision: int,
    ) -> DocumentRevisionRecord | None:
        path = self._revision_path(tenant_id, owner_user_id, document_id, logical_revision)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        record = DocumentRevisionRecord.from_dict(payload)
        if (record.tenant_id, record.owner_user_id, record.document_id, record.logical_revision) != (
            tenant_id,
            owner_user_id,
            document_id,
            logical_revision,
        ):
            raise DocumentRevisionIntegrityError("revision path identity does not match its metadata")
        return record

    def latest_revision(self, tenant_id: str, owner_user_id: str, document_id: str) -> int:
        identifier = validate_document_id(document_id)
        directory = self._owner_root(tenant_id, owner_user_id) / "revisions" / identifier
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant_id))
        try:
            revisions = [
                int(name.removesuffix(".json"))
                for name in os.listdir(descriptor)
                if len(name) == 25 and name.endswith(".json") and name.removesuffix(".json").isdigit()
            ]
        finally:
            os.close(descriptor)
        return max(revisions, default=0)

    def list_revisions(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> tuple[DocumentRevisionRecord, ...]:
        latest = self.latest_revision(tenant_id, owner_user_id, document_id)
        records: list[DocumentRevisionRecord] = []
        for revision in range(1, latest + 1):
            record = self.load_revision(tenant_id, owner_user_id, document_id, revision)
            if record is not None:
                records.append(record)
        return tuple(records)

    def read_revision_blob(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        logical_revision: int,
    ) -> bytes:
        record = self.load_revision(tenant_id, owner_user_id, document_id, logical_revision)
        if record is None:
            raise DocumentRevisionIntegrityError("document revision does not exist")
        if not record.content_blob_digest:
            raise DocumentRevisionIntegrityError("document revision has no restorable content blob")
        return self.read_blob(tenant_id, owner_user_id, document_id, record.content_blob_digest)

    def prune_unreferenced_blobs(
        self,
        tenant_id: str,
        owner_user_id: str,
        intents: Iterable[DocumentCommitIntent],
    ) -> int:
        """移除既无提交意图、也无修订权限支撑的暂存明文。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        referenced: dict[str, set[str]] = {}
        for intent in intents:
            if intent.tenant_id != tenant or intent.owner_user_id != owner:
                raise DocumentRevisionIntegrityError("blob GC intent is outside the bound owner")
            digests = referenced.setdefault(intent.document_id, set())
            if intent.after_blob_digest:
                digests.add(intent.after_blob_digest)
            if intent.revision_blob_digest:
                digests.add(intent.revision_blob_digest)

        blobs_root = self._owner_root(tenant, owner) / "blobs"
        if blobs_root.is_symlink():
            raise DocumentRevisionIntegrityError("revision blob root is a symbolic link")
        if not blobs_root.exists():
            return 0
        descriptor = _open_control_parent(blobs_root / ".scan", self._artifact_root(tenant))
        removed = 0
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            for document_id in sorted(os.listdir(descriptor)):
                try:
                    identifier = validate_document_id(document_id)
                except ValueError as exc:
                    raise DocumentRevisionIntegrityError("blob GC found an unexpected document directory") from exc
                metadata = os.stat(identifier, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise DocumentRevisionIntegrityError("blob GC found a non-directory document entry")
                document_descriptor = os.open(identifier, directory_flags, dir_fd=descriptor)
                try:
                    allowed = referenced.setdefault(identifier, set())
                    for record in self.list_revisions(tenant, owner, identifier):
                        if record.content_blob_digest:
                            allowed.add(record.content_blob_digest)
                    for name in sorted(os.listdir(document_descriptor)):
                        digest = name.removesuffix(".blob")
                        if not name.endswith(".blob") or not _is_sha256(digest):
                            raise DocumentRevisionIntegrityError("blob GC found an unexpected artifact")
                        blob_metadata = os.stat(name, dir_fd=document_descriptor, follow_symlinks=False)
                        if not stat.S_ISREG(blob_metadata.st_mode) or blob_metadata.st_nlink != 1:
                            raise DocumentRevisionIntegrityError("blob GC found a non-regular artifact")
                        if digest not in allowed:
                            os.unlink(name, dir_fd=document_descriptor)
                            removed += 1
                    os.fsync(document_descriptor)
                finally:
                    os.close(document_descriptor)
            if removed:
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return removed

    def purge_document(self, tenant_id: str, owner_user_id: str, document_id: str) -> int:
        """耐久移除一个文档所有可枚举的正文 Blob 和修订记录。"""

        identifier = validate_document_id(document_id)
        artifact_root = self._artifact_root(tenant_id)
        owner_root = self._owner_root(tenant_id, owner_user_id)
        removed = _purge_flat_directory(
            owner_root / "blobs" / identifier,
            artifact_root=artifact_root,
            allowed_name=lambda name: name.endswith(".blob") and _is_sha256(name.removesuffix(".blob")),
        )
        removed += _purge_flat_directory(
            owner_root / "revisions" / identifier,
            artifact_root=artifact_root,
            allowed_name=lambda name: (
                len(name) == 25 and name.endswith(".json") and name.removesuffix(".json").isdigit()
            ),
        )
        return removed

    def _artifact_root(self, tenant_id: str) -> Path:
        return tenant_control_root(self.root, tenant_id)

    def _owner_root(self, tenant_id: str, owner_user_id: str) -> Path:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        return self._artifact_root(tenant) / "system" / "memory-documents" / owner

    def _blob_path(self, tenant_id: str, owner_user_id: str, document_id: str, digest: str) -> Path:
        identifier = validate_document_id(document_id)
        if not _is_sha256(digest):
            raise ValueError("revision blob key must be a SHA-256 digest")
        return self._owner_root(tenant_id, owner_user_id) / "blobs" / identifier / f"{digest}.blob"

    def _revision_path(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        logical_revision: int,
    ) -> Path:
        identifier = validate_document_id(document_id)
        if logical_revision <= 0:
            raise ValueError("logical revision must be positive")
        return self._owner_root(tenant_id, owner_user_id) / "revisions" / identifier / f"{logical_revision:020d}.json"

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
                    raise DocumentRevisionIntegrityError("revision metadata is not one regular file")
                if metadata.st_size > _MAX_REVISION_METADATA_BYTES:
                    raise DocumentRevisionIntegrityError("revision metadata exceeds its size bound")
                chunks: list[bytes] = []
                remaining = _MAX_REVISION_METADATA_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        if len(raw) > _MAX_REVISION_METADATA_BYTES:
            raise DocumentRevisionIntegrityError("revision metadata exceeds its size bound")
        try:
            payload = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentRevisionIntegrityError("revision metadata is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise DocumentRevisionIntegrityError("revision metadata must be a JSON object")
        return payload


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _purge_flat_directory(
    directory: Path,
    *,
    artifact_root: Path,
    allowed_name: Any,
) -> int:
    """不跟随目录项，解除一个已知扁平产物目录中的文件链接。"""

    if not directory.exists():
        return 0
    descriptor = _open_control_parent(directory / ".scan", artifact_root)
    removed = 0
    try:
        for name in os.listdir(descriptor):
            if not isinstance(name, str) or not allowed_name(name):
                raise DocumentRevisionIntegrityError("revision purge encountered an unexpected artifact")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DocumentRevisionIntegrityError("revision purge encountered a non-regular artifact")
            os.unlink(name, dir_fd=descriptor)
            removed += 1
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = _open_control_parent(directory, artifact_root)
    try:
        try:
            os.rmdir(directory.name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return removed


__all__ = [
    "DocumentRevisionIntegrityError",
    "DocumentRevisionRecord",
    "MemoryDocumentRevisionStore",
]
