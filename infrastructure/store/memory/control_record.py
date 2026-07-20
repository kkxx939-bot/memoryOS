"""文档当前控制快照与删除发布屏障模型。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from infrastructure.store.memory.control_common import (
    _CONTROL_SCHEMA,
    _PUBLICATION_BARRIER_SCHEMA,
    DocumentControlIntegrityError,
    DocumentDeletionStatus,
)
from infrastructure.store.memory.control_common import (
    is_hex as _is_hex,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


@dataclass(frozen=True)
class DocumentControlRecord:
    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    raw_sha256: str
    size: int
    logical_revision: int
    projection_generation: int
    status: str
    last_event_id: str
    updated_at: str
    restored_from_deletion_generation: int = 0

    def __post_init__(self) -> None:
        MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        validate_document_id(self.document_id)
        if self.relative_path:
            MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        if self.raw_sha256 and not _is_hex(self.raw_sha256, 64):
            raise ValueError("control raw digest must be empty or SHA-256")
        if self.size < 0 or self.logical_revision <= 0 or self.projection_generation <= 0:
            raise ValueError("control generation or size is invalid")
        if self.restored_from_deletion_generation < 0:
            raise ValueError("control restored deletion generation cannot be negative")
        if self.restored_from_deletion_generation >= self.projection_generation:
            raise ValueError("control restored deletion generation must precede publication")
        if self.status not in {"present", "deleted"}:
            raise ValueError("document control status is invalid")
        if self.status == "present" and (not self.relative_path or not self.raw_sha256):
            raise ValueError("present control row requires a path and raw digest")
        if self.status == "deleted" and (self.raw_sha256 or self.size):
            raise ValueError("deleted control row cannot claim live content")
        if self.status == "deleted" and self.restored_from_deletion_generation:
            raise ValueError("deleted control row cannot claim a restored publication lineage")
        if not self.last_event_id.startswith("memchg_") or not _is_hex(self.last_event_id.removeprefix("memchg_"), 64):
            raise ValueError("document control event ID is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _CONTROL_SCHEMA, **self.__dict__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentControlRecord:
        if payload.get("schema") != _CONTROL_SCHEMA:
            raise DocumentControlIntegrityError("document control schema is unsupported")
        try:
            return cls(
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                relative_path=str(payload.get("relative_path") or ""),
                raw_sha256=str(payload.get("raw_sha256") or ""),
                size=int(payload["size"]),
                logical_revision=int(payload["logical_revision"]),
                projection_generation=int(payload["projection_generation"]),
                status=str(payload["status"]),
                last_event_id=str(payload["last_event_id"]),
                updated_at=str(payload["updated_at"]),
                restored_from_deletion_generation=int(
                    payload.get("restored_from_deletion_generation", 0)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document control row is malformed") from exc


@dataclass(frozen=True)
class DocumentPublicationBarrier:
    """阻止已删除字节重新出现的无正文耐久权限记录。"""

    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    deletion_generation: int
    deletion_event_digest: str
    status: DocumentDeletionStatus
    updated_at: str
    relative_path_digest: str = ""

    def __post_init__(self) -> None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        identifier = validate_document_id(self.document_id)
        status = DocumentDeletionStatus(self.status)
        relative = str(self.relative_path or "")
        digest = str(self.relative_path_digest or "")
        if relative:
            relative = MemoryDocumentPathPolicy.normalize_relative_path(relative)
            expected_digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
            if digest and digest != expected_digest:
                raise ValueError("publication barrier relative path digest is detached")
            digest = expected_digest
        elif status is not DocumentDeletionStatus.HARD_ERASED or not _is_hex(digest, 64):
            raise ValueError("only a hard-erased barrier may retain a path digest without a path")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)
        object.__setattr__(self, "document_id", identifier)
        object.__setattr__(self, "relative_path", relative)
        object.__setattr__(self, "relative_path_digest", digest)
        object.__setattr__(self, "status", status)
        if self.deletion_generation <= 0:
            raise ValueError("publication barrier deletion generation must be positive")
        if not _is_hex(self.deletion_event_digest, 64):
            raise ValueError("publication barrier event digest must be SHA-256")
        if not self.updated_at:
            raise ValueError("publication barrier timestamp must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _PUBLICATION_BARRIER_SCHEMA,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "relative_path_digest": self.relative_path_digest,
            "deletion_generation": self.deletion_generation,
            "deletion_event_digest": self.deletion_event_digest,
            "status": self.status.value,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentPublicationBarrier:
        if payload.get("schema") != _PUBLICATION_BARRIER_SCHEMA:
            raise DocumentControlIntegrityError("document publication barrier schema is unsupported")
        try:
            return cls(
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                relative_path=str(payload["relative_path"]),
                relative_path_digest=str(payload["relative_path_digest"]),
                deletion_generation=int(payload["deletion_generation"]),
                deletion_event_digest=str(payload["deletion_event_digest"]),
                status=DocumentDeletionStatus(str(payload["status"])),
                updated_at=str(payload["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document publication barrier is malformed") from exc
