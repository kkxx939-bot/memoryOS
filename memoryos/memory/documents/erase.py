"""Durable whole-document hard-erasure saga and resurrection guard."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from memoryos.core.clock import utc_now
from memoryos.core.durable_io import ImmutableArtifactConflictError, atomic_create_json, atomic_write_json
from memoryos.core.durable_io.atomic_file import _open_control_parent
from memoryos.core.file_lock import open_private_lock
from memoryos.core.integrity import canonical_json
from memoryos.memory.documents.control_store import (
    DocumentDeletionStatus,
    DocumentIntentStatus,
    DocumentPublicationBarrier,
    MemoryDocumentControlStore,
)
from memoryos.memory.documents.frontmatter import validate_document_id
from memoryos.memory.documents.layout import tenant_control_root
from memoryos.memory.documents.model import (
    ABSENT,
    DocumentEditKind,
    ManagedDocument,
    MemoryDocumentKind,
    PresentPath,
    UnsafePath,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.revision_store import MemoryDocumentRevisionStore
from memoryos.memory.documents.store import DocumentConflictError, MemoryDocumentStore

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
    """The requested erasure is detached from the exact live source state."""


class DocumentErasedError(DocumentConflictError):
    """A durable erasure epoch forbids recreating or projecting this identity."""


class DocumentEraseIntegrityError(RuntimeError):
    """A content-free durable erasure record failed validation."""


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
        """Return true only after the backend's durable cleanup is acknowledged."""
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


class MemoryDocumentEraseStore:
    """Content-free erasure epochs retained to reject stale resurrection work."""

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
        """Enumerate one bounded owner's content-free durable erasure epochs."""

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
            if record is None:  # pragma: no cover - a cooperative scan cannot lose a durable record.
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
        """Durably raise content-free high-water marks without changing an epoch."""

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
        lock_path = (
            artifact_root
            / "system"
            / "memory-documents"
            / owner
            / "locks"
            / "relation-projection.lock"
        )
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


class MemoryDocumentEraser:
    """Replayable hard-erasure coordinator over source, history and derived stores."""

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        control_store: MemoryDocumentControlStore,
        revision_store: MemoryDocumentRevisionStore,
        *,
        review_store: DocumentReviewPurger | None = None,
        cleanup_backends: Sequence[DocumentEraseCleanupBackend] = (),
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.document_store = document_store
        self.control_store = control_store
        self.revision_store = revision_store
        self.review_store = review_store
        self.cleanup_backends = tuple(cleanup_backends)
        self.clock = clock
        self.erase_store = MemoryDocumentEraseStore(control_store.root)
        names = [backend.name for backend in self.cleanup_backends]
        if len(names) != len(set(names)):
            raise ValueError("hard-erasure cleanup backend names must be unique")
        for name in names:
            _validate_backend_name(name)
            if name.startswith("local."):
                raise ValueError("configured cleanup backend cannot use the reserved local namespace")

    def hard_erase(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        expected_source_digest: str,
        relative_path: str = "",
        independent_evidence_retained: Sequence[str] = (),
    ) -> DocumentEraseResult:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        if not _is_sha256(expected_source_digest):
            raise ValueError("hard erase requires an exact lowercase source digest")
        if relative_path:
            relative_path = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        evidence = tuple(sorted({_bounded_reference(item) for item in independent_evidence_retained}))
        backend_names = self._backend_names()
        with self._document_lock(tenant, owner, identifier):
            record = self.erase_store.load(tenant, owner, identifier)
            if record is None:
                path = relative_path or self._registered_path(tenant, owner, identifier)
                if not path:
                    raise DocumentEraseConflict("hard erase requires one exact live registered document")
                live = self.document_store.read_state(tenant, owner, path)
                control = self.control_store.load_control(tenant, owner, identifier)
                if isinstance(live, PresentPath):
                    if live.raw_sha256 != expected_source_digest:
                        raise DocumentEraseConflict(
                            "hard erase expected digest does not match the live document"
                        )
                elif live == ABSENT:
                    revisions = self.revision_store.list_revisions(tenant, owner, identifier)
                    latest = revisions[-1] if revisions else None
                    if (
                        control is None
                        or control.status != "deleted"
                        or control.relative_path != path
                        or latest is None
                        or latest.state != "ABSENT"
                        or latest.edit_kind is not DocumentEditKind.DELETE
                        or latest.relative_path != path
                        or latest.content_blob_role != "before_delete"
                        or latest.content_blob_digest != expected_source_digest
                    ):
                        raise DocumentEraseConflict(
                            "hard erase ABSENT target is not one exact soft-forgotten document"
                        )
                else:
                    raise DocumentEraseConflict(
                        "hard erase expected digest does not match the live document"
                    )
                active = [
                    intent
                    for intent in self.control_store.incomplete_intents(tenant, owner)
                    if intent.document_id == identifier
                    and intent.status not in {DocumentIntentStatus.COMPLETED, DocumentIntentStatus.CONFLICTED}
                ]
                if active:
                    raise DocumentEraseConflict("hard erase requires existing document intents to finish recovery")
                revision_floor = control.logical_revision if control is not None else 0
                projection_floor = max(
                    control.projection_generation if control is not None else 0,
                    self._serving_projection_generation_floor(tenant, owner, identifier),
                    self._protected_projection_generation_floor(tenant, owner, identifier),
                )
                record = self.erase_store.begin(
                    tenant_id=tenant,
                    owner_user_id=owner,
                    document_id=identifier,
                    relative_path=path,
                    source_digest=expected_source_digest,
                    document_revision_floor=revision_floor,
                    projection_generation_floor=projection_floor,
                    backend_names=backend_names,
                    independent_evidence_retained=evidence,
                    started_at=self.clock(),
                )
            elif record.source_digest != expected_source_digest:
                raise DocumentEraseConflict("hard erase retry changed its exact source digest")
            elif evidence and evidence != record.independent_evidence_retained:
                raise DocumentEraseConflict("hard erase retry changed independent evidence disclosure")
            record = self.erase_store.merge_backends(record, backend_names, updated_at=self.clock())
            current_control = self.control_store.load_control(tenant, owner, identifier)
            record = self.erase_store.raise_floors(
                record,
                document_revision_floor=(
                    current_control.logical_revision if current_control is not None else 0
                ),
                projection_generation_floor=max(
                    current_control.projection_generation if current_control is not None else 0,
                    self._serving_projection_generation_floor(tenant, owner, identifier),
                    self._protected_projection_generation_floor(tenant, owner, identifier),
                ),
                updated_at=self.clock(),
            )
            self._seal_hard_publication_barrier(record)
            if record.status != DocumentEraseStatus.ERASED:
                record = self._run(record, relative_path=relative_path)
            return DocumentEraseResult(record, record.independent_evidence_retained)

    def recover_owner(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = _MAX_ERASE_RECORDS_PER_OWNER,
    ) -> DocumentEraseRecoveryReport:
        """Replay every configured cleanup backend for bounded durable epochs."""

        completed: list[str] = []
        pending: list[str] = []
        for record in self.erase_store.records(tenant_id, owner_user_id, limit=limit):
            result = self.hard_erase(
                tenant_id=record.tenant_id,
                owner_user_id=record.owner_user_id,
                document_id=record.document_id,
                expected_source_digest=record.source_digest,
                relative_path=record.relative_path,
            )
            (completed if result.completed else pending).append(record.document_id)
        return DocumentEraseRecoveryReport(tuple(completed), tuple(pending))

    def _seal_hard_publication_barrier(
        self,
        record: DocumentEraseRecord,
    ) -> DocumentPublicationBarrier:
        current = self.control_store.load_publication_barrier(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        digest = record.erasure_epoch.removeprefix("erase_")
        generation = record.projection_generation_floor + 1
        if current is not None:
            if current.status is DocumentDeletionStatus.HARD_ERASED:
                if (
                    current.deletion_event_digest != digest
                    or current.relative_path_digest != record.relative_path_digest
                ):
                    raise DocumentEraseIntegrityError(
                        "hard-erasure epoch conflicts with the protected publication barrier"
                    )
                generation = max(generation, current.deletion_generation)
            else:
                generation = max(generation, current.deletion_generation + 1)
        return self.control_store.write_publication_barrier(
            DocumentPublicationBarrier(
                tenant_id=record.tenant_id,
                owner_user_id=record.owner_user_id,
                document_id=record.document_id,
                relative_path=record.relative_path,
                relative_path_digest=record.relative_path_digest,
                deletion_generation=generation,
                deletion_event_digest=digest,
                status=DocumentDeletionStatus.HARD_ERASED,
                updated_at=self.clock(),
            )
        )

    def _serving_projection_generation_floor(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        floor = 0
        for backend in self.cleanup_backends:
            provider = getattr(backend, "projection_generation_floor", None)
            if not callable(provider):
                continue
            value = cast(DocumentEraseFloorProvider, backend).projection_generation_floor(
                tenant_id,
                owner_user_id,
                document_id,
            )
            if int(value) < 0:
                raise DocumentEraseIntegrityError("hard-erasure serving generation floor is negative")
            floor = max(floor, int(value))
        return floor

    def _protected_projection_generation_floor(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        barrier = self.control_store.load_publication_barrier(
            tenant_id,
            owner_user_id,
            document_id,
        )
        if barrier is None:
            return 0
        if barrier.status is DocumentDeletionStatus.HARD_ERASED:
            return max(0, barrier.deletion_generation - 1)
        return barrier.deletion_generation

    def _run(self, record: DocumentEraseRecord, *, relative_path: str) -> DocumentEraseRecord:
        local_actions: tuple[tuple[str, Callable[[], bool]], ...] = (
            (_LOCAL_LIVE, lambda: self._erase_live(record, relative_path)),
            (_LOCAL_REVISIONS, lambda: self._erase_revisions(record)),
            *(((_LOCAL_REVIEWS, lambda: self._erase_reviews(record)),) if self.review_store is not None else ()),
            (_LOCAL_CONTROLS, lambda: self._erase_controls(record)),
        )
        for name, action in local_actions:
            if _progress(record, name).acknowledged:
                continue
            record = self._attempt(record, name, action)
            if not _progress(record, name).acknowledged:
                return self.erase_store.finish(record, completed_at=self.clock())

        request = DerivedEraseRequest(
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            document_id=record.document_id,
            document_uri=record.document_uri,
            relative_path=record.relative_path,
            document_kind=record.document_kind,
            erasure_epoch=record.erasure_epoch,
            source_digest=record.source_digest,
            document_revision_floor=record.document_revision_floor,
            projection_generation_floor=record.projection_generation_floor,
            relative_path_digest=record.relative_path_digest,
        )
        for backend in self.cleanup_backends:
            if _progress(record, backend.name).acknowledged:
                continue
            record = self._attempt(
                record,
                backend.name,
                partial(backend.erase_document, request),
            )
        if not record.pending_backends:
            self.control_store.scrub_hard_erasure_path(
                record.tenant_id,
                record.owner_user_id,
                record.document_id,
                expected_relative_path_digest=record.relative_path_digest,
                expected_deletion_event_digest=record.erasure_epoch.removeprefix("erase_"),
                updated_at=self.clock(),
            )
        return self.erase_store.finish(record, completed_at=self.clock())

    def _attempt(
        self,
        record: DocumentEraseRecord,
        backend_name: str,
        action: Callable[[], bool],
    ) -> DocumentEraseRecord:
        attempted_at = self.clock()
        try:
            acknowledged = bool(action())
            failure_code = "" if acknowledged else "NOT_ACKNOWLEDGED"
        except Exception as exc:  # noqa: BLE001 - durable saga records a content-free typed failure.
            acknowledged = False
            failure_code = type(exc).__name__
        return self.erase_store.record_attempt(
            record,
            backend_name,
            acknowledged=acknowledged,
            attempted_at=attempted_at,
            failure_code=failure_code,
        )

    def _erase_live(self, record: DocumentEraseRecord, relative_path: str) -> bool:
        path = relative_path or self._registered_path(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        if not path:
            return True
        live = self.document_store.read_state(record.tenant_id, record.owner_user_id, path)
        if live == ABSENT:
            return True
        if isinstance(live, UnsafePath):
            raise DocumentEraseConflict("live document path became unsafe during erasure")
        if not isinstance(live, PresentPath) or live.raw_sha256 != record.source_digest:
            raise DocumentEraseConflict("live document changed after the durable erasure intent")
        self.document_store.full_scan(record.tenant_id, record.owner_user_id)
        self.document_store.delete(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
            expected_state=live,
        )
        return self.document_store.read_state(record.tenant_id, record.owner_user_id, path) == ABSENT

    def _erase_revisions(self, record: DocumentEraseRecord) -> bool:
        self.revision_store.purge_document(record.tenant_id, record.owner_user_id, record.document_id)
        return True

    def _erase_reviews(self, record: DocumentEraseRecord) -> bool:
        assert self.review_store is not None
        self.review_store.purge_document(record.tenant_id, record.owner_user_id, record.document_id)
        return True

    def _erase_controls(self, record: DocumentEraseRecord) -> bool:
        self.control_store.purge_document(record.tenant_id, record.owner_user_id, record.document_id)
        return True

    def _registered_path(self, tenant_id: str, owner_user_id: str, document_id: str) -> str:
        scan = self.document_store.full_scan(tenant_id, owner_user_id)
        if not scan.complete or scan.errors:
            raise DocumentEraseConflict("hard erase requires a complete live document registration scan")
        matches = [
            item.relative_path
            for item in scan.registrations
            if isinstance(item, ManagedDocument) and item.document_id == document_id
        ]
        if len(matches) > 1:
            raise DocumentEraseConflict("hard erase document identity is duplicated")
        return matches[0] if matches else ""

    def _backend_names(self) -> tuple[str, ...]:
        local = [_LOCAL_LIVE, _LOCAL_REVISIONS]
        if self.review_store is not None:
            local.append(_LOCAL_REVIEWS)
        local.append(_LOCAL_CONTROLS)
        return tuple(local + [backend.name for backend in self.cleanup_backends])

    def _document_lock(self, tenant_id: str, owner_user_id: str, document_id: str) -> _LockedDocument:
        return self.erase_store.document_lock(tenant_id, owner_user_id, document_id)


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
    "DocumentErasedError",
    "DocumentReviewPurger",
    "EraseBackendProgress",
    "MemoryDocumentEraseStore",
    "MemoryDocumentEraser",
]
