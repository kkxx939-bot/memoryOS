"""Content-free roll-forward saga for multi-document consolidation.

The target document is committed and exactly projected before any redundant
source is soft-forgotten.  Every source deletion remains an independent
single-document CAS, so a crash can leave duplicate content but cannot make
the saga roll back or delete a source before the target is durable.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from memoryos.core.clock import utc_now
from memoryos.core.durable_io import atomic_create_json, atomic_write_json
from memoryos.core.durable_io.atomic_file import _open_control_parent
from memoryos.core.file_lock import open_private_lock
from memoryos.core.integrity import canonical_json
from memoryos.memory.documents.commit import DocumentCommitResult, MemoryDocumentCommitter
from memoryos.memory.documents.control_store import document_intent_id
from memoryos.memory.documents.frontmatter import validate_document_id
from memoryos.memory.documents.layout import tenant_control_root
from memoryos.memory.documents.model import (
    DocumentEditKind,
    DocumentEditPlan,
    PresentPath,
    raw_state_to_dict,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.store import DocumentConflictError

_SAGA_SCHEMA = "memory_document_consolidation_v1"
_MAX_SAGA_BYTES = 1024 * 1024
_MAX_SOURCES = 1_000
_MAX_SAGAS_PER_OWNER = 10_000


class ConsolidationIntegrityError(RuntimeError):
    """A consolidation journal is malformed or identity-detached."""


class ConsolidationInputRequired(RuntimeError):
    """Recovery reached a target that was never durably prepared."""


class ConsolidationStatus(str, Enum):
    PREPARED = "PREPARED"
    TARGET_COMMITTED = "TARGET_COMMITTED"
    AWAITING_TARGET_PROJECTION = "AWAITING_TARGET_PROJECTION"
    SOFT_FORGETTING = "SOFT_FORGETTING"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class ConsolidationSource:
    """Content-free exact state of one redundant source document."""

    document_id: str
    relative_path: str
    raw_sha256: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", validate_document_id(self.document_id))
        normalized = MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        object.__setattr__(self, "relative_path", normalized)
        _require_digest(self.raw_sha256, "source raw digest")
        if self.size < 0:
            raise ValueError("consolidation source size cannot be negative")

    @property
    def expected_state(self) -> PresentPath:
        return PresentPath(self.relative_path, self.raw_sha256, self.size)

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "raw_sha256": self.raw_sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConsolidationSource:
        try:
            return cls(
                document_id=str(payload["document_id"]),
                relative_path=str(payload["relative_path"]),
                raw_sha256=str(payload["raw_sha256"]),
                size=_coerce_int(payload["size"], "consolidation source size"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConsolidationIntegrityError("consolidation source is malformed") from exc


@dataclass(frozen=True)
class ConsolidationSagaRecord:
    """Durable content-free progress for one ordered consolidation."""

    saga_id: str
    identity_digest: str | None
    idempotency_digest: str
    tenant_id: str
    owner_user_id: str
    actor_binding: str
    target_document_id: str
    target_relative_path: str
    target_source_digest: str
    target_plan_digest: str
    target_intent_id: str
    sources: tuple[ConsolidationSource, ...]
    status: ConsolidationStatus
    target_projection_generation: int
    target_projection_confirmed_at: str
    next_source_index: int
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _validate_prefixed_digest(self.saga_id, "memsaga_", "saga_id")
        _require_digest(self.idempotency_digest, "consolidation idempotency digest")
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        target = validate_document_id(self.target_document_id)
        target_path = MemoryDocumentPathPolicy.normalize_relative_path(self.target_relative_path)
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)
        object.__setattr__(self, "target_document_id", target)
        object.__setattr__(self, "target_relative_path", target_path)
        _require_digest(self.target_source_digest, "target source digest")
        _require_digest(self.target_plan_digest, "target plan digest")
        _validate_prefixed_digest(self.target_intent_id, "mdintent_", "target_intent_id")
        if not self.actor_binding or len(self.actor_binding) > 512:
            raise ValueError("consolidation actor binding must be non-empty and bounded")
        if len(self.sources) > _MAX_SOURCES:
            raise ValueError("consolidation source count exceeds its bound")
        source_ids = tuple(source.document_id for source in self.sources)
        if len(set(source_ids)) != len(source_ids) or target in source_ids:
            raise ValueError("consolidation sources must be unique and cannot include the target")
        if not 0 <= self.next_source_index <= len(self.sources):
            raise ValueError("consolidation source cursor is invalid")
        if self.target_projection_generation < 0:
            raise ValueError("target projection generation cannot be negative")
        if self.status == ConsolidationStatus.PREPARED and self.target_projection_generation:
            raise ValueError("a prepared consolidation cannot claim a target generation")
        if self.status != ConsolidationStatus.PREPARED and self.target_projection_generation <= 0:
            raise ValueError("an advanced consolidation requires a target generation")
        if self.target_projection_confirmed_at and self.target_projection_generation <= 0:
            raise ValueError("target projection confirmation requires a generation")
        if self.next_source_index and not self.target_projection_confirmed_at:
            raise ValueError("source deletion cannot precede target projection confirmation")
        if self.status == ConsolidationStatus.COMPLETED and self.next_source_index != len(self.sources):
            raise ValueError("a completed consolidation must finish every source")
        if not self.created_at or not self.updated_at:
            raise ValueError("consolidation timestamps must be non-empty")
        expected_saga_id = consolidation_saga_id(tenant, owner, self.idempotency_digest)
        if self.saga_id != expected_saga_id:
            raise ValueError("consolidation saga ID is detached from its trusted scope")
        expected_identity = consolidation_identity_digest(self)
        if self.identity_digest is None:
            object.__setattr__(self, "identity_digest", expected_identity)
        elif self.identity_digest != expected_identity:
            raise ValueError("consolidation immutable identity digest does not match")

    def immutable_payload(self) -> dict[str, object]:
        return {
            "schema": _SAGA_SCHEMA,
            "saga_id": self.saga_id,
            "idempotency_digest": self.idempotency_digest,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "actor_binding": self.actor_binding,
            "target_document_id": self.target_document_id,
            "target_relative_path": self.target_relative_path,
            "target_source_digest": self.target_source_digest,
            "target_plan_digest": self.target_plan_digest,
            "target_intent_id": self.target_intent_id,
            "sources": [source.to_dict() for source in self.sources],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.immutable_payload(),
            "identity_digest": self.identity_digest or "",
            "status": self.status.value,
            "target_projection_generation": self.target_projection_generation,
            "target_projection_confirmed_at": self.target_projection_confirmed_at,
            "next_source_index": self.next_source_index,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConsolidationSagaRecord:
        if payload.get("schema") != _SAGA_SCHEMA:
            raise ConsolidationIntegrityError("consolidation schema is unsupported")
        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list):
            raise ConsolidationIntegrityError("consolidation sources must be an array")
        try:
            return cls(
                saga_id=str(payload["saga_id"]),
                identity_digest=str(payload["identity_digest"]),
                idempotency_digest=str(payload["idempotency_digest"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                actor_binding=str(payload["actor_binding"]),
                target_document_id=str(payload["target_document_id"]),
                target_relative_path=str(payload["target_relative_path"]),
                target_source_digest=str(payload["target_source_digest"]),
                target_plan_digest=str(payload["target_plan_digest"]),
                target_intent_id=str(payload["target_intent_id"]),
                sources=tuple(ConsolidationSource.from_dict(_mapping(item)) for item in raw_sources),
                status=ConsolidationStatus(str(payload["status"])),
                target_projection_generation=_coerce_int(
                    payload["target_projection_generation"],
                    "target projection generation",
                ),
                target_projection_confirmed_at=str(payload.get("target_projection_confirmed_at") or ""),
                next_source_index=_coerce_int(payload["next_source_index"], "consolidation source cursor"),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConsolidationIntegrityError("consolidation journal is malformed") from exc


def consolidation_saga_id(tenant_id: str, owner_user_id: str, idempotency_digest: str) -> str:
    encoded = canonical_json(
        ["memory_document_consolidation_v1", tenant_id, owner_user_id, idempotency_digest]
    ).encode()
    return f"memsaga_{hashlib.sha256(encoded).hexdigest()}"


def consolidation_identity_digest(record: ConsolidationSagaRecord) -> str:
    return hashlib.sha256(canonical_json(record.immutable_payload()).encode()).hexdigest()


class MemoryDocumentConsolidationStore:
    """Crash-safe content-free consolidation progress journal."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def create(self, record: ConsolidationSagaRecord) -> ConsolidationSagaRecord:
        atomic_create_json(
            self._record_path(record.tenant_id, record.owner_user_id, record.saga_id),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )
        durable = self.load(record.tenant_id, record.owner_user_id, record.saga_id)
        if durable is None:  # pragma: no cover - create-only publication cannot cooperatively disappear.
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
        """Return a bounded, identity-checked snapshot of owner sagas."""

        names = self._record_names(tenant_id, owner_user_id)
        if len(names) > _bounded_list_limit(limit):
            raise ConsolidationIntegrityError("consolidation journal count exceeds the requested bound")
        records: list[ConsolidationSagaRecord] = []
        for name in names:
            record = self.load(tenant_id, owner_user_id, name.removesuffix(".json"))
            if record is None:  # pragma: no cover - startup has no concurrent journal purger.
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
        """Return non-completed sagas without silently truncating recovery."""

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


@dataclass(frozen=True)
class ConsolidationResult:
    saga_id: str
    status: ConsolidationStatus
    target_document_id: str
    target_projection_generation: int
    target_projection_confirmed: bool
    soft_forgotten_document_ids: tuple[str, ...]
    pending_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConsolidationRecoveryReport:
    """Bounded startup recovery outcome; identifiers contain no document body."""

    examined: int
    completed_saga_ids: tuple[str, ...] = ()
    awaiting_projection_saga_ids: tuple[str, ...] = ()
    awaiting_input_saga_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "examined": self.examined,
            "completed": len(self.completed_saga_ids),
            "awaiting_projection": len(self.awaiting_projection_saga_ids),
            "awaiting_input": len(self.awaiting_input_saga_ids),
        }


ConsolidationFaultHook = Callable[[str, ConsolidationSagaRecord], None]


class ConsolidationProjectionReader(Protocol):
    """Minimal derived-state proof required before source deletion."""

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> Mapping[str, object] | None: ...


class MemoryDocumentConsolidator:
    """Roll a bounded multi-document consolidation forward, never backward."""

    def __init__(
        self,
        committer: MemoryDocumentCommitter,
        projection_store: ConsolidationProjectionReader,
        *,
        saga_store: MemoryDocumentConsolidationStore | None = None,
        clock: Callable[[], str] = utc_now,
        test_hook: ConsolidationFaultHook | None = None,
    ) -> None:
        self.committer = committer
        self.projection_store = projection_store
        self.saga_store = saga_store or MemoryDocumentConsolidationStore(committer.control_store.root)
        self.clock = clock
        self.test_hook = test_hook

    def consolidate(
        self,
        target_plan: DocumentEditPlan,
        sources: Sequence[ConsolidationSource],
        *,
        idempotency_key: str,
        actor_binding: str,
    ) -> ConsolidationResult:
        """Commit/project the target, then soft-forget each source in order."""

        prepared = self._prepare_record(
            target_plan,
            sources,
            idempotency_key=idempotency_key,
            actor_binding=actor_binding,
        )
        with self.saga_store.lock(prepared.tenant_id, prepared.owner_user_id, prepared.saga_id):
            record = self.saga_store.load(prepared.tenant_id, prepared.owner_user_id, prepared.saga_id)
            if record is None:
                record = self.saga_store.create(prepared)
                self._notify("after_saga_checkpoint", record)
            elif record.identity_digest != prepared.identity_digest:
                raise ConsolidationIntegrityError("consolidation retry changed its immutable inputs")
            return self._advance(record, target_plan=target_plan)

    def resume(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        saga_id: str,
        actor_binding: str,
    ) -> ConsolidationResult:
        """Resume a saga after its target intent was durably prepared."""

        with self.saga_store.lock(tenant_id, owner_user_id, saga_id):
            record = self.saga_store.load(tenant_id, owner_user_id, saga_id)
            if record is None:
                raise ConsolidationInputRequired("consolidation journal does not exist")
            if record.actor_binding != actor_binding:
                raise PermissionError("consolidation recovery actor binding differs from its journal")
            return self._advance(record, target_plan=None)

    def resume_all(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        limit: int = 1_000,
    ) -> ConsolidationRecoveryReport:
        """Resume every bounded pending saga using its sealed actor binding.

        A PREPARED saga without a durable target intent cannot be recreated
        from a content-free journal.  It is therefore reported as awaiting
        input and left untouched; in particular, no source deletion runs.
        """

        pending = self.saga_store.list_pending(tenant_id, owner_user_id, limit=limit)
        completed: list[str] = []
        awaiting_projection: list[str] = []
        awaiting_input: list[str] = []
        for snapshot in pending:
            with self.saga_store.lock(tenant_id, owner_user_id, snapshot.saga_id):
                record = self.saga_store.load(tenant_id, owner_user_id, snapshot.saga_id)
                if record is None or record.status == ConsolidationStatus.COMPLETED:
                    continue
                try:
                    result = self._advance(record, target_plan=None)
                except ConsolidationInputRequired:
                    awaiting_input.append(record.saga_id)
                    continue
                if result.status == ConsolidationStatus.COMPLETED:
                    completed.append(record.saga_id)
                elif not result.target_projection_confirmed:
                    awaiting_projection.append(record.saga_id)
        return ConsolidationRecoveryReport(
            examined=len(pending),
            completed_saga_ids=tuple(completed),
            awaiting_projection_saga_ids=tuple(awaiting_projection),
            awaiting_input_saga_ids=tuple(awaiting_input),
        )

    def _advance(
        self,
        record: ConsolidationSagaRecord,
        *,
        target_plan: DocumentEditPlan | None,
    ) -> ConsolidationResult:
        if record.status == ConsolidationStatus.COMPLETED:
            self._require_target_live(record)
            return self._result(record, projection_confirmed=self._target_projection_matches(record))

        if record.target_projection_generation == 0:
            target_result = self._commit_target(record, target_plan)
            self._notify("after_target_commit", record)
            control = target_result.control or self.committer.control_store.load_control(
                record.tenant_id,
                record.owner_user_id,
                record.target_document_id,
            )
            if control is None or control.status != "present" or control.raw_sha256 != record.target_source_digest:
                raise DocumentConflictError("consolidation target commit did not install its exact Markdown")
            record = self.saga_store.save(
                replace(
                    record,
                    status=ConsolidationStatus.TARGET_COMMITTED,
                    target_projection_generation=control.projection_generation,
                    updated_at=self.clock(),
                )
            )
            self._notify("after_target_checkpoint", record)

        self._require_target_live(record)
        if not self._target_projection_matches(record):
            if record.status in {
                ConsolidationStatus.TARGET_COMMITTED,
                ConsolidationStatus.AWAITING_TARGET_PROJECTION,
            }:
                record = self.saga_store.save(
                    replace(
                        record,
                        status=ConsolidationStatus.AWAITING_TARGET_PROJECTION,
                        updated_at=self.clock(),
                    )
                )
            return self._result(record, projection_confirmed=False)

        if not record.target_projection_confirmed_at:
            final_status = ConsolidationStatus.COMPLETED if not record.sources else ConsolidationStatus.SOFT_FORGETTING
            record = self.saga_store.save(
                replace(
                    record,
                    status=final_status,
                    target_projection_confirmed_at=self.clock(),
                    updated_at=self.clock(),
                )
            )
            self._notify("after_projection_checkpoint", record)

        while record.next_source_index < len(record.sources):
            self._require_target_live(record)
            if not self._target_projection_matches(record):
                return self._result(record, projection_confirmed=False)
            source_index = record.next_source_index
            source = record.sources[source_index]
            source_plan = self._source_delete_plan(record, source, source_index)
            result = self.committer.commit(
                source_plan,
                actor_binding=record.actor_binding,
                evidence_reference=f"consolidation:{record.saga_id}:source:{source_index}",
            )
            if result.control is None or result.control.status != "deleted":
                raise DocumentConflictError("consolidation source soft-forget did not install ABSENT")
            self._notify("after_source_commit", record)
            next_index = source_index + 1
            status = (
                ConsolidationStatus.COMPLETED
                if next_index == len(record.sources)
                else ConsolidationStatus.SOFT_FORGETTING
            )
            record = self.saga_store.save(
                replace(
                    record,
                    status=status,
                    next_source_index=next_index,
                    updated_at=self.clock(),
                )
            )
            self._notify("after_source_checkpoint", record)
        return self._result(record, projection_confirmed=True)

    def _commit_target(
        self,
        record: ConsolidationSagaRecord,
        target_plan: DocumentEditPlan | None,
    ) -> DocumentCommitResult:
        existing = self.committer.control_store.load_intent(
            record.tenant_id,
            record.owner_user_id,
            record.target_intent_id,
        )
        if existing is not None:
            return self.committer.recover_intent(
                record.tenant_id,
                record.owner_user_id,
                record.target_intent_id,
            )
        if target_plan is None:
            raise ConsolidationInputRequired(
                "target plan bytes were never durably prepared; resubmit the exact consolidation request"
            )
        if _target_plan_digest(target_plan) != record.target_plan_digest:
            raise ConsolidationIntegrityError("resubmitted target plan differs from its saga journal")
        return self.committer.commit(
            target_plan,
            actor_binding=record.actor_binding,
            evidence_reference=f"consolidation:{record.saga_id}:target",
        )

    def _require_target_live(self, record: ConsolidationSagaRecord) -> None:
        control = self.committer.control_store.load_control(
            record.tenant_id,
            record.owner_user_id,
            record.target_document_id,
        )
        live = self.committer.document_store.read_state(
            record.tenant_id,
            record.owner_user_id,
            record.target_relative_path,
        )
        if (
            control is None
            or control.status != "present"
            or control.raw_sha256 != record.target_source_digest
            or control.projection_generation != record.target_projection_generation
            or not isinstance(live, PresentPath)
            or live.raw_sha256 != record.target_source_digest
        ):
            raise DocumentConflictError("consolidation target changed after commit; redundant sources were preserved")

    def _target_projection_matches(self, record: ConsolidationSagaRecord) -> bool:
        state = self.projection_store.get_memory_document_projection_state(
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            document_id=record.target_document_id,
        )
        if state is None:
            return False
        return (
            str(state.get("tenant_id") or "") == record.tenant_id
            and str(state.get("owner_user_id") or "") == record.owner_user_id
            and str(state.get("document_id") or "") == record.target_document_id
            and str(state.get("source_digest") or "") == record.target_source_digest
            and _coerce_int(
                state.get("projection_generation") or 0,
                "serving projection generation",
            )
            == record.target_projection_generation
            and str(state.get("projection_status") or "") == "PROJECTED"
            and not str(state.get("deletion_status") or "")
        )

    def _prepare_record(
        self,
        target_plan: DocumentEditPlan,
        sources: Sequence[ConsolidationSource],
        *,
        idempotency_key: str,
        actor_binding: str,
    ) -> ConsolidationSagaRecord:
        if target_plan.edit_kind not in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE}:
            raise ValueError("consolidation target must be a CREATE or UPDATE plan")
        if target_plan.after_bytes is None:
            raise ValueError("consolidation target plan requires exact after bytes")
        tenant = MemoryDocumentPathPolicy.trusted_segment(target_plan.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(target_plan.owner_user_id, "owner_user_id")
        target = validate_document_id(target_plan.document_id)
        path = MemoryDocumentPathPolicy.normalize_relative_path(target_plan.relative_path)
        key = str(idempotency_key or "").strip()
        if not key or len(key) > 512:
            raise ValueError("consolidation idempotency key must be non-empty and bounded")
        actor = str(actor_binding or "").strip()
        if not actor or len(actor) > 512:
            raise ValueError("consolidation actor binding must be non-empty and bounded")
        bounded_sources = tuple(sources)
        if len(bounded_sources) > _MAX_SOURCES:
            raise ValueError("consolidation source count exceeds its bound")
        idempotency_digest = hashlib.sha256(key.encode()).hexdigest()
        saga_id = consolidation_saga_id(tenant, owner, idempotency_digest)
        target_key_digest = hashlib.sha256(target_plan.idempotency_key.encode()).hexdigest()
        now = self.clock()
        return ConsolidationSagaRecord(
            saga_id=saga_id,
            identity_digest=None,
            idempotency_digest=idempotency_digest,
            tenant_id=tenant,
            owner_user_id=owner,
            actor_binding=actor,
            target_document_id=target,
            target_relative_path=path,
            target_source_digest=hashlib.sha256(target_plan.after_bytes).hexdigest(),
            target_plan_digest=_target_plan_digest(target_plan),
            target_intent_id=document_intent_id(tenant, owner, target, target_key_digest),
            sources=bounded_sources,
            status=ConsolidationStatus.PREPARED,
            target_projection_generation=0,
            target_projection_confirmed_at="",
            next_source_index=0,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _source_delete_plan(
        record: ConsolidationSagaRecord,
        source: ConsolidationSource,
        source_index: int,
    ) -> DocumentEditPlan:
        evidence_digest = hashlib.sha256(
            canonical_json(
                [
                    "memory_document_consolidation_source_v1",
                    record.saga_id,
                    source_index,
                    source.document_id,
                    source.relative_path,
                    source.raw_sha256,
                ]
            ).encode()
        ).hexdigest()
        return DocumentEditPlan(
            idempotency_key=f"consolidation:{record.saga_id}:soft-forget:{source_index}",
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            edit_kind=DocumentEditKind.DELETE,
            expected_state=source.expected_state,
            evidence_digest=evidence_digest,
            edit_summary="consolidation soft forget redundant source",
            document_id=source.document_id,
            relative_path=source.relative_path,
            expected_registration_document_id=source.document_id,
        )

    @staticmethod
    def _result(
        record: ConsolidationSagaRecord,
        *,
        projection_confirmed: bool,
    ) -> ConsolidationResult:
        return ConsolidationResult(
            saga_id=record.saga_id,
            status=record.status,
            target_document_id=record.target_document_id,
            target_projection_generation=record.target_projection_generation,
            target_projection_confirmed=projection_confirmed,
            soft_forgotten_document_ids=tuple(
                source.document_id for source in record.sources[: record.next_source_index]
            ),
            pending_document_ids=tuple(source.document_id for source in record.sources[record.next_source_index :]),
        )

    def _notify(self, stage: str, record: ConsolidationSagaRecord) -> None:
        if self.test_hook is not None:
            self.test_hook(stage, record)


def _target_plan_digest(plan: DocumentEditPlan) -> str:
    after_digest = hashlib.sha256(plan.after_bytes).hexdigest() if plan.after_bytes is not None else ""
    payload = {
        "idempotency_key": plan.idempotency_key,
        "tenant_id": plan.tenant_id,
        "owner_user_id": plan.owner_user_id,
        "edit_kind": plan.edit_kind.value,
        "expected_state": raw_state_to_dict(plan.expected_state),
        "evidence_digest": plan.evidence_digest,
        "edit_summary": plan.edit_summary,
        "document_id": plan.document_id,
        "relative_path": plan.relative_path,
        "after_digest": after_digest,
        "new_relative_path": plan.new_relative_path,
        "expected_new_state": raw_state_to_dict(plan.expected_new_state),
        "expected_registration_document_id": plan.expected_registration_document_id,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConsolidationIntegrityError("consolidation journal object is malformed")
    return {str(key): item for key, item in value.items()}


def _coerce_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str, bytes, bytearray)):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _validate_prefixed_digest(value: str, prefix: str, label: str) -> None:
    if not value.startswith(prefix):
        raise ValueError(f"{label} has an invalid prefix")
    _require_digest(value.removeprefix(prefix), label)


def _status_rank(status: ConsolidationStatus) -> int:
    return {
        ConsolidationStatus.PREPARED: 0,
        ConsolidationStatus.TARGET_COMMITTED: 1,
        ConsolidationStatus.AWAITING_TARGET_PROJECTION: 2,
        ConsolidationStatus.SOFT_FORGETTING: 3,
        ConsolidationStatus.COMPLETED: 4,
    }[status]


def _bounded_list_limit(limit: int) -> int:
    if not 1 <= limit <= _MAX_SAGAS_PER_OWNER:
        raise ValueError(f"consolidation list limit must be between 1 and {_MAX_SAGAS_PER_OWNER}")
    return limit


__all__ = [
    "ConsolidationInputRequired",
    "ConsolidationIntegrityError",
    "ConsolidationProjectionReader",
    "ConsolidationRecoveryReport",
    "ConsolidationResult",
    "ConsolidationSagaRecord",
    "ConsolidationSource",
    "ConsolidationStatus",
    "MemoryDocumentConsolidationStore",
    "MemoryDocumentConsolidator",
    "consolidation_identity_digest",
    "consolidation_saga_id",
]
