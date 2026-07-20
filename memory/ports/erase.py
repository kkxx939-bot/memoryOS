"""记忆文档硬删除的共享模型和持久化协议。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from memory.core.model import MemoryDocumentKind
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import DocumentConflictError

_ERASE_SCHEMA = "memory_document_erasure_v2"
_MAX_ERASE_RECORD_BYTES = 512 * 1024
_MAX_ERASE_RECORDS_PER_OWNER = 10_000
_MAX_INDEPENDENT_EVIDENCE_REFERENCES = 256
_BACKEND_NAME = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_LOCAL_LIVE = "local.live_source"
_LOCAL_REVISIONS = "local.revision_artifacts"
_LOCAL_REVIEWS = "local.review_artifacts"
_LOCAL_CONTROLS = "local.control_artifacts"


class DocumentEraseStatus(str, Enum):
    ERASING = "ERASING"
    ERASE_PENDING = "ERASE_PENDING"
    ERASED = "ERASED"


class DocumentEraseConflict(DocumentConflictError):
    """删除请求与精确的当前正文状态不一致。"""


class DocumentErasedError(DocumentConflictError):
    """耐久删除纪元禁止重新创建或投影该文档身份。"""


class DocumentEraseIntegrityError(RuntimeError):
    """不含正文的耐久删除记录未通过完整性校验。"""


@dataclass(frozen=True)
class EraseBackendProgress:
    backend_name: str
    acknowledged: bool = False
    attempt_count: int = 0
    last_attempt_at: str = ""
    failure_code: str = ""

    def __post_init__(self) -> None:
        _validate_backend_name(self.backend_name)
        if self.attempt_count < 0:
            raise ValueError("erasure backend attempt count cannot be negative")
        if self.acknowledged and self.failure_code:
            raise ValueError("acknowledged erasure backend cannot retain a failure code")
        if self.failure_code and (
            len(self.failure_code) > 160
            or not all(character.isalnum() or character in "_.:-" for character in self.failure_code)
        ):
            raise ValueError("erasure backend failure code is invalid")
        if self.attempt_count and not self.last_attempt_at:
            raise ValueError("attempted erasure backend requires a timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "acknowledged": self.acknowledged,
            "attempt_count": self.attempt_count,
            "last_attempt_at": self.last_attempt_at,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EraseBackendProgress:
        return cls(
            backend_name=str(payload["backend_name"]),
            acknowledged=bool(payload.get("acknowledged", False)),
            attempt_count=int(payload.get("attempt_count", 0)),
            last_attempt_at=str(payload.get("last_attempt_at") or ""),
            failure_code=str(payload.get("failure_code") or ""),
        )


@dataclass(frozen=True)
class DocumentEraseRecord:
    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    relative_path_digest: str
    document_kind: str
    erasure_epoch: str
    source_digest: str
    document_revision_floor: int
    projection_generation_floor: int
    status: DocumentEraseStatus
    backends: tuple[EraseBackendProgress, ...]
    independent_evidence_retained: tuple[str, ...]
    started_at: str
    updated_at: str
    completed_at: str = ""

    def __post_init__(self) -> None:
        MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        validate_document_id(self.document_id)
        if self.relative_path:
            relative = MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
            if MemoryDocumentPathPolicy.kind_for(relative).value != self.document_kind:
                raise ValueError("erasure document kind is detached from its safe relative path")
            if _path_digest(relative) != self.relative_path_digest:
                raise ValueError("erasure relative path digest is detached from its path")
        else:
            MemoryDocumentKind(self.document_kind)
            if not _is_sha256(self.relative_path_digest):
                raise ValueError("erasure scrubbed relative path requires its digest")
        if not self.erasure_epoch.startswith("erase_") or not _is_sha256(self.erasure_epoch.removeprefix("erase_")):
            raise ValueError("erasure epoch is invalid")
        if not _is_sha256(self.source_digest):
            raise ValueError("erasure source digest must be SHA-256")
        if self.document_revision_floor < 0 or self.projection_generation_floor < 0:
            raise ValueError("erasure revision generation floors cannot be negative")
        names = [backend.backend_name for backend in self.backends]
        if len(names) != len(set(names)) or _LOCAL_LIVE not in names:
            raise ValueError("erasure backend set is invalid")
        if not self.started_at or not self.updated_at:
            raise ValueError("erasure timestamps must be non-empty")
        evidence = tuple(sorted({_bounded_reference(item) for item in self.independent_evidence_retained}))
        if len(evidence) > _MAX_INDEPENDENT_EVIDENCE_REFERENCES:
            raise ValueError("independent evidence reference count exceeds its bound")
        object.__setattr__(self, "independent_evidence_retained", evidence)
        all_acknowledged = all(backend.acknowledged for backend in self.backends)
        if self.status == DocumentEraseStatus.ERASED and (not all_acknowledged or not self.completed_at):
            raise ValueError("ERASED tombstone requires all backend acknowledgements")
        if self.status != DocumentEraseStatus.ERASED and self.completed_at:
            raise ValueError("only an ERASED tombstone may have a completion time")

    @property
    def document_uri(self) -> str:
        return MemoryDocumentPathPolicy.document_uri(self.owner_user_id, self.document_id)

    @property
    def pending_backends(self) -> tuple[str, ...]:
        return tuple(backend.backend_name for backend in self.backends if not backend.acknowledged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _ERASE_SCHEMA,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "relative_path_digest": self.relative_path_digest,
            "document_kind": self.document_kind,
            "erasure_epoch": self.erasure_epoch,
            "source_digest": self.source_digest,
            "document_revision_floor": self.document_revision_floor,
            "projection_generation_floor": self.projection_generation_floor,
            "status": self.status.value,
            "backends": [backend.to_dict() for backend in self.backends],
            "independent_evidence_retained": list(self.independent_evidence_retained),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentEraseRecord:
        if payload.get("schema") != _ERASE_SCHEMA:
            raise DocumentEraseIntegrityError("document erasure schema is unsupported")
        try:
            backend_payload = payload["backends"]
            if not isinstance(backend_payload, list):
                raise TypeError("erasure backends must be a list")
            return cls(
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                relative_path=str(payload["relative_path"]),
                relative_path_digest=str(payload["relative_path_digest"]),
                document_kind=str(payload["document_kind"]),
                erasure_epoch=str(payload["erasure_epoch"]),
                source_digest=str(payload["source_digest"]),
                document_revision_floor=int(payload.get("document_revision_floor", 0)),
                projection_generation_floor=int(payload.get("projection_generation_floor", 0)),
                status=DocumentEraseStatus(str(payload["status"])),
                backends=tuple(EraseBackendProgress.from_dict(_mapping(item)) for item in backend_payload),
                independent_evidence_retained=tuple(
                    str(item) for item in _sequence(payload["independent_evidence_retained"])
                ),
                started_at=str(payload["started_at"]),
                updated_at=str(payload["updated_at"]),
                completed_at=str(payload.get("completed_at") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentEraseIntegrityError("document erasure record is malformed") from exc


@dataclass(frozen=True)
class DerivedEraseRequest:
    tenant_id: str
    owner_user_id: str
    document_id: str
    document_uri: str
    relative_path: str
    document_kind: str
    erasure_epoch: str
    source_digest: str
    document_revision_floor: int
    projection_generation_floor: int
    relative_path_digest: str = ""


class DocumentEraseCleanupBackend(Protocol):
    name: str

    def erase_document(self, request: DerivedEraseRequest) -> bool:
        """只有后端确认清理已经耐久完成时才返回真。"""
        ...


class DocumentEraseFloorProvider(Protocol):
    def projection_generation_floor(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int: ...


class DocumentReviewPurger(Protocol):
    def purge_document(self, tenant_id: str, owner_user_id: str, document_id: str) -> int: ...


@dataclass(frozen=True)
class DocumentEraseResult:
    record: DocumentEraseRecord
    independent_evidence_retained: tuple[str, ...] = ()
    media_disclaimer: str = (
        "Logical hard erase completed for configured MemoryOS stores; ordinary file deletion "
        "does not guarantee physical-media secure erasure."
    )

    @property
    def completed(self) -> bool:
        return self.record.status == DocumentEraseStatus.ERASED


@dataclass(frozen=True)
class DocumentEraseRecoveryReport:
    completed_document_ids: tuple[str, ...] = ()
    pending_document_ids: tuple[str, ...] = ()


class DocumentEraseStore(Protocol):
    """硬删除事务所需的耐久纪元存储边界。"""

    def load(self, tenant_id: str, owner_user_id: str, document_id: str) -> DocumentEraseRecord | None: ...

    def records(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = _MAX_ERASE_RECORDS_PER_OWNER,
    ) -> tuple[DocumentEraseRecord, ...]: ...

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
    ) -> DocumentEraseRecord: ...

    def merge_backends(
        self,
        record: DocumentEraseRecord,
        backend_names: Sequence[str],
        *,
        updated_at: str,
    ) -> DocumentEraseRecord: ...

    def record_attempt(
        self,
        record: DocumentEraseRecord,
        backend_name: str,
        *,
        acknowledged: bool,
        attempted_at: str,
        failure_code: str = "",
    ) -> DocumentEraseRecord: ...

    def finish(self, record: DocumentEraseRecord, *, completed_at: str) -> DocumentEraseRecord: ...

    def write(self, record: DocumentEraseRecord) -> DocumentEraseRecord: ...

    def raise_floors(
        self,
        record: DocumentEraseRecord,
        *,
        document_revision_floor: int,
        projection_generation_floor: int,
        updated_at: str,
    ) -> DocumentEraseRecord: ...

    def assert_mutation_allowed(self, tenant_id: str, owner_user_id: str, document_id: str) -> None: ...

    def assert_projection_allowed(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        *,
        projection_generation: int,
    ) -> None: ...

    def document_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> AbstractContextManager[None]: ...

    def owner_relation_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> AbstractContextManager[None]: ...


def _progress(record: DocumentEraseRecord, backend_name: str) -> EraseBackendProgress:
    try:
        return next(item for item in record.backends if item.backend_name == backend_name)
    except StopIteration as exc:
        raise DocumentEraseIntegrityError("erasure backend is missing from its durable intent") from exc


def _bounded_reference(value: object) -> str:
    reference = str(value or "").strip()
    if not reference or len(reference) > 2048 or any(ord(character) < 32 for character in reference):
        raise ValueError("independent evidence reference is empty, too large or contains controls")
    return reference


def _path_digest(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("independent evidence references must be an array")
    return value


def _validate_backend_name(value: str) -> None:
    if not _BACKEND_NAME.fullmatch(str(value or "")):
        raise ValueError("hard-erasure cleanup backend name is invalid")


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DocumentEraseIntegrityError("erasure metadata field must be an object")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "DerivedEraseRequest",
    "DocumentEraseCleanupBackend",
    "DocumentEraseConflict",
    "DocumentEraseFloorProvider",
    "DocumentEraseIntegrityError",
    "DocumentEraseRecord",
    "DocumentEraseRecoveryReport",
    "DocumentEraseResult",
    "DocumentEraseStatus",
    "DocumentEraseStore",
    "DocumentErasedError",
    "DocumentReviewPurger",
    "EraseBackendProgress",
]
