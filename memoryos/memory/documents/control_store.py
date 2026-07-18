"""Content-free durable control records for Markdown document commits."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from memoryos.core.durable_io import ImmutableArtifactConflictError, atomic_create_json, atomic_write_json
from memoryos.core.durable_io.atomic_file import _open_control_parent
from memoryos.core.file_lock import open_private_lock
from memoryos.core.integrity import canonical_json
from memoryos.memory.documents.frontmatter import validate_document_id
from memoryos.memory.documents.layout import tenant_control_root
from memoryos.memory.documents.model import (
    AbsentPath,
    DocumentChangeEvent,
    DocumentEditKind,
    PresentPath,
    RawPathState,
    UnsafePath,
    raw_state_from_dict,
    raw_state_to_dict,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy

_INTENT_SCHEMA = "memory_document_intent_v1"
_EVENT_SCHEMA = "memory_document_change_event_v1"
_CONTROL_SCHEMA = "memory_document_control_v1"
_ADOPTION_RECEIPT_SCHEMA = "memory_document_adoption_receipt_v1"
_ADOPTION_IDENTITY_SCHEMA = "memory_document_adoption_identity_v1"
_ROOT_IDENTITY_SCHEMA = "memory_document_root_identity_v1"
_PUBLICATION_BARRIER_SCHEMA = "memory_document_publication_barrier_v1"
_MAX_CONTROL_BYTES = 1024 * 1024
_MAX_LINEAGE_EVENTS = 10_000
_MAX_PUBLICATION_BARRIERS = 10_000
_MAX_DOCUMENT_CONTROLS = 10_000
_MAX_ADOPTION_RECEIPTS = 10_000


class DocumentControlIntegrityError(RuntimeError):
    """A durable document control artifact is malformed or identity-detached."""


class DocumentIntentStatus(str, Enum):
    PREPARED = "PREPARED"
    INSTALLED = "INSTALLED"
    EVENT_APPENDED = "EVENT_APPENDED"
    PROJECTION_ENQUEUED = "PROJECTION_ENQUEUED"
    COMPLETED = "COMPLETED"
    CONFLICTED = "CONFLICTED"


class DocumentDeletionStatus(str, Enum):
    """Durable publication states that outlive every rebuildable serving row."""

    SOFT_FORGOTTEN = "SOFT_FORGOTTEN"
    HARD_ERASED = "HARD_ERASED"


@dataclass(frozen=True)
class DocumentRootIdentity:
    """Content-free durable binding to one owner's controlled source root."""

    tenant_id: str
    owner_user_id: str
    root_identity: str

    def __post_init__(self) -> None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        if not _is_hex(self.root_identity, 32):
            raise ValueError("document root identity must be a 128-bit lowercase hex digest")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _ROOT_IDENTITY_SCHEMA, **self.__dict__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentRootIdentity:
        if (
            payload.get("schema") != _ROOT_IDENTITY_SCHEMA
            or set(payload) != {"schema", "tenant_id", "owner_user_id", "root_identity"}
        ):
            raise DocumentControlIntegrityError("document root identity schema is unsupported")
        try:
            return cls(
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                root_identity=str(payload["root_identity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document root identity is malformed") from exc


@dataclass(frozen=True)
class DocumentRootIdentityGuard:
    """Capability valid only while one owner's root-identity lock is held."""

    _store: MemoryDocumentControlStore
    tenant_id: str
    owner_user_id: str

    def ensure(
        self,
        root_identity: str,
        *,
        allow_prepared_bootstrap: bool = False,
    ) -> DocumentRootIdentity:
        requested = DocumentRootIdentity(
            tenant_id=self.tenant_id,
            owner_user_id=self.owner_user_id,
            root_identity=root_identity,
        )
        return self._store._ensure_root_identity_locked(
            requested,
            allow_prepared_bootstrap=allow_prepared_bootstrap,
        )


@dataclass(frozen=True)
class DocumentAdoptionReceipt:
    """Content-free authority for retrying one unmanaged-file adoption.

    The assigned document identity is derived from the complete request
    identity.  Consequently concurrent creators publish identical immutable
    bytes, while a retry after the live front-matter rewrite can recover the
    same identity without retaining the source body.
    """

    receipt_id: str
    request_digest: str
    tenant_id: str
    owner_user_id: str
    relative_path: str
    expected_raw_sha256: str
    document_id: str
    actor_binding: str
    evidence_reference: str
    evidence_digest: str
    idempotency_key: str
    edit_summary: str

    def __post_init__(self) -> None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        expected = str(self.expected_raw_sha256)
        if not _is_hex(expected, 64):
            raise ValueError("adoption expected raw digest must be SHA-256")
        request_digest = adoption_request_digest(tenant, owner, relative, expected)
        receipt_id = f"mdadopt_{request_digest}"
        document_id = adoption_document_id(request_digest)
        if (
            self.request_digest != request_digest
            or self.receipt_id != receipt_id
            or self.document_id != document_id
        ):
            raise ValueError("document adoption receipt is detached from its request identity")
        if (
            not self.actor_binding
            or len(self.actor_binding) > 512
            or any(ord(character) < 32 and character not in "\t" for character in self.actor_binding)
        ):
            raise ValueError("document adoption actor binding is invalid")
        if self.evidence_reference != f"adoption-receipt:{receipt_id}":
            raise ValueError("document adoption evidence reference is detached from its receipt")
        if self.evidence_digest != expected:
            raise ValueError("document adoption evidence digest is detached from its source digest")
        if self.idempotency_key != f"adoption:{receipt_id}":
            raise ValueError("document adoption idempotency key is detached from its receipt")
        if self.edit_summary != "adopt unmanaged Markdown document":
            raise ValueError("document adoption edit summary is invalid")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)
        object.__setattr__(self, "relative_path", relative)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _ADOPTION_RECEIPT_SCHEMA, **self.__dict__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentAdoptionReceipt:
        if payload.get("schema") != _ADOPTION_RECEIPT_SCHEMA:
            raise DocumentControlIntegrityError("document adoption receipt schema is unsupported")
        try:
            return cls(
                receipt_id=str(payload["receipt_id"]),
                request_digest=str(payload["request_digest"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                relative_path=str(payload["relative_path"]),
                expected_raw_sha256=str(payload["expected_raw_sha256"]),
                document_id=str(payload["document_id"]),
                actor_binding=str(payload["actor_binding"]),
                evidence_reference=str(payload["evidence_reference"]),
                evidence_digest=str(payload["evidence_digest"]),
                idempotency_key=str(payload["idempotency_key"]),
                edit_summary=str(payload["edit_summary"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document adoption receipt is malformed") from exc


def adoption_request_digest(
    tenant_id: str,
    owner_user_id: str,
    relative_path: str,
    expected_raw_sha256: str,
) -> str:
    tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
    owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
    relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
    if not _is_hex(expected_raw_sha256, 64):
        raise ValueError("adoption expected raw digest must be SHA-256")
    encoded = canonical_json(
        ["memory_document_adoption_v1", tenant, owner, relative, expected_raw_sha256]
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def adoption_document_id(request_digest: str) -> str:
    if not _is_hex(request_digest, 64):
        raise ValueError("adoption request digest must be SHA-256")
    suffix = hashlib.sha256(
        canonical_json(["memory_document_adoption_document_v1", request_digest]).encode()
    ).hexdigest()
    return validate_document_id(f"memdoc_{suffix}")


@dataclass(frozen=True)
class DocumentPathEffect:
    """One exact path transition in a single-document effect vector."""

    relative_path: str
    before: RawPathState
    after: RawPathState

    def __post_init__(self) -> None:
        normalized = MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        object.__setattr__(self, "relative_path", normalized)
        for label, state in (("before", self.before), ("after", self.after)):
            if isinstance(state, UnsafePath):
                raise ValueError(f"{label} UNSAFE state cannot be persisted as an authorized effect")
            if isinstance(state, PresentPath) and state.relative_path != normalized:
                raise ValueError(f"{label} PRESENT state is detached from its effect path")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "before": raw_state_to_dict(self.before),
            "after": raw_state_to_dict(self.after),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentPathEffect:
        return cls(
            relative_path=str(payload["relative_path"]),
            before=raw_state_from_dict(_mapping(payload.get("before"), "intent before state")),
            after=raw_state_from_dict(_mapping(payload.get("after"), "intent after state")),
        )


@dataclass(frozen=True)
class DocumentCommitIntent:
    """Content-free roll-forward journal for one document identity."""

    intent_id: str
    idempotency_digest: str
    identity_digest: str | None
    tenant_id: str
    owner_user_id: str
    document_id: str
    edit_kind: DocumentEditKind
    effects: tuple[DocumentPathEffect, ...]
    after_blob_digest: str
    revision_blob_digest: str
    revision_blob_role: str
    logical_revision: int
    projection_generation: int
    event_id: str
    projection_job_id: str
    old_relative_path: str
    new_relative_path: str
    actor_binding: str
    evidence_reference: str
    evidence_digest: str
    edit_summary: str
    status: DocumentIntentStatus
    created_at: str
    updated_at: str
    restored_from_deletion_generation: int = 0
    conflict_reason: str = ""

    def __post_init__(self) -> None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        document_id = validate_document_id(self.document_id)
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)
        object.__setattr__(self, "document_id", document_id)
        if not self.intent_id.startswith("mdintent_") or not _is_hex(self.intent_id.removeprefix("mdintent_"), 64):
            raise ValueError("document intent ID is invalid")
        for label, digest in (
            ("idempotency digest", self.idempotency_digest),
            ("evidence digest", self.evidence_digest),
        ):
            if not _is_hex(digest, 64):
                raise ValueError(f"{label} must be a SHA-256 digest")
        for label, digest in (
            ("after blob digest", self.after_blob_digest),
            ("revision blob digest", self.revision_blob_digest),
        ):
            if digest and not _is_hex(digest, 64):
                raise ValueError(f"{label} must be empty or a SHA-256 digest")
        if self.revision_blob_role not in {"after", "before_delete", ""}:
            raise ValueError("revision blob role is invalid")
        if self.after_blob_digest and self.revision_blob_role != "after":
            raise ValueError("an after blob must also be the revision after blob")
        if self.revision_blob_role and not self.revision_blob_digest:
            raise ValueError("revision blob role requires a revision blob digest")
        if self.logical_revision <= 0 or self.projection_generation <= 0:
            raise ValueError("document revision generations must be positive")
        if self.restored_from_deletion_generation < 0:
            raise ValueError("restored deletion generation cannot be negative")
        if self.restored_from_deletion_generation >= self.projection_generation:
            raise ValueError("restored deletion generation must precede publication")
        if self.edit_kind == DocumentEditKind.DELETE and self.restored_from_deletion_generation:
            raise ValueError("a deletion intent cannot claim a restored publication lineage")
        if not self.event_id.startswith("memchg_") or not _is_hex(self.event_id.removeprefix("memchg_"), 64):
            raise ValueError("document event ID is invalid")
        if not self.projection_job_id.startswith("memory_projection_") or not _is_hex(
            self.projection_job_id.removeprefix("memory_projection_"), 64
        ):
            raise ValueError("document projection job ID is invalid")
        if not self.effects or len(self.effects) > 2:
            raise ValueError("document intent requires a bounded effect vector")
        if not self.actor_binding or not self.evidence_reference or not self.edit_summary:
            raise ValueError("document intent lineage and summary must be non-empty")
        if len(self.actor_binding) > 512 or len(self.evidence_reference) > 2048 or len(self.edit_summary) > 500:
            raise ValueError("document intent lineage or summary exceeds its bound")
        if not self.created_at or not self.updated_at:
            raise ValueError("document intent timestamps must be non-empty")
        if self.status == DocumentIntentStatus.CONFLICTED and not self.conflict_reason:
            raise ValueError("conflicted document intent requires a bounded reason")
        if self.status != DocumentIntentStatus.CONFLICTED and self.conflict_reason:
            raise ValueError("only a conflicted document intent may carry a conflict reason")
        if len(self.conflict_reason) > 500:
            raise ValueError("document intent conflict reason exceeds its bound")
        self._validate_effect_shape()
        expected_intent_id = document_intent_id(tenant, owner, document_id, self.idempotency_digest)
        if self.intent_id != expected_intent_id:
            raise ValueError("document intent ID is detached from tenant, owner, document or idempotency")
        expected_identity_digest = document_intent_identity_digest(self)
        if self.identity_digest is None:
            object.__setattr__(self, "identity_digest", expected_identity_digest)
        elif not _is_hex(self.identity_digest, 64) or self.identity_digest != expected_identity_digest:
            raise ValueError("document intent immutable identity digest does not match")

    def _validate_effect_shape(self) -> None:
        if self.edit_kind == DocumentEditKind.CREATE:
            if len(self.effects) != 1:
                raise ValueError("CREATE intent requires one path effect")
            effect = self.effects[0]
            if not isinstance(effect.before, AbsentPath) or not isinstance(effect.after, PresentPath):
                raise ValueError("CREATE intent must transition ABSENT to PRESENT")
            if self.old_relative_path or self.new_relative_path != effect.relative_path:
                raise ValueError("CREATE intent paths are detached from its effect")
            if (
                self.after_blob_digest != effect.after.raw_sha256
                or self.revision_blob_digest != effect.after.raw_sha256
                or self.revision_blob_role != "after"
            ):
                raise ValueError("CREATE intent blobs are detached from its exact after state")
            return
        if self.edit_kind == DocumentEditKind.UPDATE:
            if len(self.effects) != 1:
                raise ValueError("UPDATE intent requires one path effect")
            effect = self.effects[0]
            if not isinstance(effect.before, PresentPath) or not isinstance(effect.after, PresentPath):
                raise ValueError("UPDATE intent must transition PRESENT to PRESENT")
            if self.old_relative_path != effect.relative_path or self.new_relative_path != effect.relative_path:
                raise ValueError("UPDATE intent paths are detached from its effect")
            if (
                self.after_blob_digest != effect.after.raw_sha256
                or self.revision_blob_digest != effect.after.raw_sha256
                or self.revision_blob_role != "after"
            ):
                raise ValueError("UPDATE intent blobs are detached from its exact after state")
            return
        if self.edit_kind == DocumentEditKind.DELETE:
            if len(self.effects) != 1:
                raise ValueError("DELETE intent requires one path effect")
            effect = self.effects[0]
            if not isinstance(effect.before, PresentPath) or not isinstance(effect.after, AbsentPath):
                raise ValueError("DELETE intent must transition PRESENT to ABSENT")
            if self.old_relative_path != effect.relative_path or self.new_relative_path:
                raise ValueError("DELETE intent paths are detached from its effect")
            if (
                self.after_blob_digest
                or self.revision_blob_digest != effect.before.raw_sha256
                or self.revision_blob_role != "before_delete"
            ):
                raise ValueError("DELETE intent must not invent an after blob")
            return
        if len(self.effects) != 2:
            raise ValueError("RENAME intent requires a two-path effect vector")
        old_effect, new_effect = self.effects
        if (
            not isinstance(old_effect.before, PresentPath)
            or not isinstance(old_effect.after, AbsentPath)
            or not isinstance(new_effect.before, AbsentPath)
            or not isinstance(new_effect.after, PresentPath)
        ):
            raise ValueError("RENAME intent effect vector has an invalid transition")
        if (
            self.old_relative_path != old_effect.relative_path
            or self.new_relative_path != new_effect.relative_path
            or old_effect.relative_path == new_effect.relative_path
        ):
            raise ValueError("RENAME intent paths are detached from its effect vector")
        if (
            self.after_blob_digest != new_effect.after.raw_sha256
            or self.revision_blob_digest != new_effect.after.raw_sha256
            or self.revision_blob_role != "after"
        ):
            raise ValueError("RENAME intent blobs are detached from its exact two-path after state")

    def immutable_payload(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "idempotency_digest": self.idempotency_digest,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "document_id": self.document_id,
            "edit_kind": self.edit_kind.value,
            "effects": [effect.to_dict() for effect in self.effects],
            "after_blob_digest": self.after_blob_digest,
            "revision_blob_digest": self.revision_blob_digest,
            "revision_blob_role": self.revision_blob_role,
            "logical_revision": self.logical_revision,
            "projection_generation": self.projection_generation,
            "event_id": self.event_id,
            "projection_job_id": self.projection_job_id,
            "old_relative_path": self.old_relative_path,
            "new_relative_path": self.new_relative_path,
            "actor_binding": self.actor_binding,
            "evidence_reference": self.evidence_reference,
            "evidence_digest": self.evidence_digest,
            "edit_summary": self.edit_summary,
            "restored_from_deletion_generation": self.restored_from_deletion_generation,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _INTENT_SCHEMA,
            **self.immutable_payload(),
            "identity_digest": self.identity_digest,
            "status": self.status.value,
            "updated_at": self.updated_at,
            "conflict_reason": self.conflict_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentCommitIntent:
        if payload.get("schema") != _INTENT_SCHEMA:
            raise DocumentControlIntegrityError("document intent schema is unsupported")
        try:
            effects_payload = payload["effects"]
            if not isinstance(effects_payload, list):
                raise TypeError("effects must be a list")
            return cls(
                intent_id=str(payload["intent_id"]),
                idempotency_digest=str(payload["idempotency_digest"]),
                identity_digest=str(payload["identity_digest"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                edit_kind=DocumentEditKind(str(payload["edit_kind"])),
                effects=tuple(
                    DocumentPathEffect.from_dict(_mapping(item, "intent effect")) for item in effects_payload
                ),
                after_blob_digest=str(payload.get("after_blob_digest") or ""),
                revision_blob_digest=str(payload.get("revision_blob_digest") or ""),
                revision_blob_role=str(payload.get("revision_blob_role") or ""),
                logical_revision=int(payload["logical_revision"]),
                projection_generation=int(payload["projection_generation"]),
                event_id=str(payload["event_id"]),
                projection_job_id=str(payload["projection_job_id"]),
                old_relative_path=str(payload.get("old_relative_path") or ""),
                new_relative_path=str(payload.get("new_relative_path") or ""),
                actor_binding=str(payload["actor_binding"]),
                evidence_reference=str(payload["evidence_reference"]),
                evidence_digest=str(payload["evidence_digest"]),
                edit_summary=str(payload["edit_summary"]),
                status=DocumentIntentStatus(str(payload["status"])),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                restored_from_deletion_generation=int(
                    payload.get("restored_from_deletion_generation", 0)
                ),
                conflict_reason=str(payload.get("conflict_reason") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document intent is malformed") from exc


def document_intent_id(tenant_id: str, owner_user_id: str, document_id: str, idempotency_digest: str) -> str:
    encoded = canonical_json(
        ["memory_document_intent_v1", tenant_id, owner_user_id, document_id, idempotency_digest]
    ).encode()
    return f"mdintent_{hashlib.sha256(encoded).hexdigest()}"


def document_intent_identity_digest(intent: DocumentCommitIntent) -> str:
    return hashlib.sha256(canonical_json(intent.immutable_payload()).encode()).hexdigest()


def deletion_event_digest(
    *,
    event_id: str,
    document_id: str,
    before_raw_digest: str,
    projection_generation: int,
) -> str:
    """Bind a content-free deletion fence to one exact document event."""

    _validate_prefixed_digest(event_id, "memchg_", "event_id")
    validate_document_id(document_id)
    if not _is_hex(before_raw_digest, 64) or projection_generation <= 0:
        raise ValueError("deletion event requires an exact before digest and positive generation")
    material = "\x1f".join(
        (event_id, document_id, before_raw_digest, str(projection_generation))
    )
    return hashlib.sha256(material.encode()).hexdigest()


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
    """Content-free durable authority preventing deleted-byte resurrection."""

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


class MemoryDocumentControlStore:
    """Filesystem control journal; never an authority over live Markdown."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    @contextmanager
    def root_identity_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> Iterator[DocumentRootIdentityGuard]:
        """Serialize one owner's first root-authority publication."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        artifact_root = self._artifact_root(tenant)
        descriptor = open_private_lock(
            self._owner_root(tenant, owner) / "locks" / "root-identity.lock",
            root=artifact_root,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield DocumentRootIdentityGuard(self, tenant, owner)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def prepare_adoption_receipt(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        actor_binding: str,
    ) -> DocumentAdoptionReceipt:
        """Create/replay a receipt only after owner root authority exists."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        with self.root_identity_lock(tenant, owner):
            if self.load_root_identity(tenant, owner) is None:
                raise DocumentControlIntegrityError(
                    "document adoption receipt requires an existing source root identity"
                )
            return self._prepare_adoption_receipt_locked(
                tenant,
                owner,
                relative_path,
                expected_raw_sha256,
                actor_binding=actor_binding,
            )

    def _prepare_adoption_receipt_locked(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        actor_binding: str,
    ) -> DocumentAdoptionReceipt:
        """Publish receipt and identity index while holding owner authority lock."""

        request_digest = adoption_request_digest(
            tenant_id,
            owner_user_id,
            relative_path,
            expected_raw_sha256,
        )
        receipt = DocumentAdoptionReceipt(
            receipt_id=f"mdadopt_{request_digest}",
            request_digest=request_digest,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            relative_path=relative_path,
            expected_raw_sha256=expected_raw_sha256,
            document_id=adoption_document_id(request_digest),
            actor_binding=actor_binding,
            evidence_reference=f"adoption-receipt:mdadopt_{request_digest}",
            evidence_digest=expected_raw_sha256,
            idempotency_key=f"adoption:mdadopt_{request_digest}",
            edit_summary="adopt unmanaged Markdown document",
        )
        try:
            atomic_create_json(
                self._adoption_receipt_path(receipt.tenant_id, receipt.owner_user_id, receipt.receipt_id),
                receipt.to_dict(),
                artifact_root=self._artifact_root(receipt.tenant_id),
            )
        except ImmutableArtifactConflictError:
            pass
        durable = self.load_adoption_receipt(
            receipt.tenant_id,
            receipt.owner_user_id,
            receipt.receipt_id,
        )
        if durable is None:
            raise DocumentControlIntegrityError("document adoption receipt conflicts with its request identity")
        identity_payload = {
            "schema": _ADOPTION_IDENTITY_SCHEMA,
            "tenant_id": durable.tenant_id,
            "owner_user_id": durable.owner_user_id,
            "document_id": durable.document_id,
            "receipt_id": durable.receipt_id,
            "request_digest": durable.request_digest,
        }
        try:
            atomic_create_json(
                self._adoption_identity_path(durable.tenant_id, durable.owner_user_id, durable.document_id),
                identity_payload,
                artifact_root=self._artifact_root(durable.tenant_id),
            )
        except ImmutableArtifactConflictError:
            pass
        indexed = self.load_adoption_receipt_for_document(
            durable.tenant_id,
            durable.owner_user_id,
            durable.document_id,
        )
        if indexed != durable:
            raise DocumentControlIntegrityError("document adoption identity index conflicts with its receipt")
        return durable

    def ensure_root_identity(
        self,
        tenant_id: str,
        owner_user_id: str,
        root_identity: str,
        *,
        allow_prepared_bootstrap: bool = False,
    ) -> DocumentRootIdentity:
        """Create one immutable root binding; never bless a replacement inode."""

        requested = DocumentRootIdentity(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            root_identity=root_identity,
        )
        with self.root_identity_lock(
            requested.tenant_id,
            requested.owner_user_id,
        ) as guard:
            return guard.ensure(
                requested.root_identity,
                allow_prepared_bootstrap=allow_prepared_bootstrap,
            )

    def _ensure_root_identity_locked(
        self,
        requested: DocumentRootIdentity,
        *,
        allow_prepared_bootstrap: bool,
    ) -> DocumentRootIdentity:
        current = self.load_root_identity(requested.tenant_id, requested.owner_user_id)
        if current is not None:
            if current != requested:
                raise DocumentControlIntegrityError(
                    "document source root identity changed and requires explicit reset"
                )
            return current
        bootstrap = self._read_json(
            self._bootstrap_path(requested.tenant_id, requested.owner_user_id),
            requested.tenant_id,
        )
        if bootstrap is not None:
            valid_prepared = bool(
                bootstrap.get("schema") == "memory_document_bootstrap_v1"
                and bootstrap.get("status") == "PREPARED"
                and bootstrap.get("tenant_id") == requested.tenant_id
                and bootstrap.get("owner_user_id") == requested.owner_user_id
            )
            if not allow_prepared_bootstrap or not valid_prepared:
                raise DocumentControlIntegrityError(
                    "existing bootstrap authority is missing its source root identity"
                )
        elif allow_prepared_bootstrap:
            raise DocumentControlIntegrityError(
                "root identity bootstrap authority requires an exact PREPARED marker"
            )
        if self.controls(requested.tenant_id, requested.owner_user_id):
            raise DocumentControlIntegrityError(
                "existing document controls are missing their durable source root identity"
            )
        if self.incomplete_intents(requested.tenant_id, requested.owner_user_id):
            raise DocumentControlIntegrityError(
                "existing document intents are missing their durable source root identity"
            )
        if self.adoption_receipts(requested.tenant_id, requested.owner_user_id):
            raise DocumentControlIntegrityError(
                "existing adoption receipts are missing their durable source root identity"
            )
        try:
            atomic_create_json(
                self._root_identity_path(requested.tenant_id, requested.owner_user_id),
                requested.to_dict(),
                artifact_root=self._artifact_root(requested.tenant_id),
            )
        except ImmutableArtifactConflictError:
            pass
        durable = self.load_root_identity(requested.tenant_id, requested.owner_user_id)
        if durable != requested:
            raise DocumentControlIntegrityError("document source root identity publication conflicted")
        return requested

    def load_root_identity(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> DocumentRootIdentity | None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        payload = self._read_json(self._root_identity_path(tenant, owner), tenant)
        if payload is None:
            return None
        identity = DocumentRootIdentity.from_dict(payload)
        if identity.tenant_id != tenant or identity.owner_user_id != owner:
            raise DocumentControlIntegrityError(
                "document root identity path binding does not match its payload"
            )
        return identity

    def root_identity_blockers(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[str, ...]:
        """Return durable authorities that forbid initial root publication."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        with self.root_identity_lock(tenant, owner):
            if self.load_root_identity(tenant, owner) is not None:
                return ()
            blockers: list[str] = []
            if self.controls(tenant, owner):
                blockers.append("controls")
            if self.incomplete_intents(tenant, owner):
                blockers.append("intents")
            if self.adoption_receipts(tenant, owner):
                blockers.append("adoption_receipts")
            if self._read_json(self._bootstrap_path(tenant, owner), tenant) is not None:
                blockers.append("bootstrap")
            return tuple(blockers)

    def load_adoption_receipt(
        self,
        tenant_id: str,
        owner_user_id: str,
        receipt_id: str,
    ) -> DocumentAdoptionReceipt | None:
        _validate_prefixed_digest(receipt_id, "mdadopt_", "receipt_id")
        payload = self._read_json(
            self._adoption_receipt_path(tenant_id, owner_user_id, receipt_id),
            tenant_id,
        )
        if payload is None:
            return None
        receipt = DocumentAdoptionReceipt.from_dict(payload)
        if (
            receipt.receipt_id != receipt_id
            or receipt.tenant_id != tenant_id
            or receipt.owner_user_id != owner_user_id
        ):
            raise DocumentControlIntegrityError("document adoption receipt path identity does not match")
        return receipt

    def adoption_receipts(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[DocumentAdoptionReceipt, ...]:
        """Enumerate an owner's bounded exact adoption authorities."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "adoptions"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = tuple(sorted(os.listdir(descriptor)))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_ADOPTION_RECEIPTS:
            raise DocumentControlIntegrityError("document adoption receipt count exceeds its bound")
        receipts: list[DocumentAdoptionReceipt] = []
        for name in names:
            receipt_id = name.removesuffix(".json")
            if (
                not name.endswith(".json")
                or not receipt_id.startswith("mdadopt_")
                or not _is_hex(receipt_id.removeprefix("mdadopt_"), 64)
            ):
                raise DocumentControlIntegrityError(
                    "document adoption directory contains an unexpected artifact"
                )
            receipt = self.load_adoption_receipt(tenant, owner, receipt_id)
            if receipt is None:  # pragma: no cover - cooperative snapshots retain files.
                raise DocumentControlIntegrityError(
                    "document adoption receipt disappeared during enumeration"
                )
            receipts.append(receipt)
        return tuple(receipts)

    def load_adoption_receipt_for_document(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentAdoptionReceipt | None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        payload = self._read_json(
            self._adoption_identity_path(tenant, owner, identifier),
            tenant,
        )
        if payload is None:
            return None
        try:
            if payload.get("schema") != _ADOPTION_IDENTITY_SCHEMA:
                raise ValueError("unsupported schema")
            receipt_id = str(payload["receipt_id"])
            request_digest = str(payload["request_digest"])
            if (
                payload.get("tenant_id") != tenant
                or payload.get("owner_user_id") != owner
                or payload.get("document_id") != identifier
                or receipt_id != f"mdadopt_{request_digest}"
            ):
                raise ValueError("identity mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document adoption identity index is malformed") from exc
        receipt = self.load_adoption_receipt(tenant, owner, receipt_id)
        if (
            receipt is None
            or receipt.request_digest != request_digest
            or receipt.document_id != identifier
        ):
            raise DocumentControlIntegrityError("document adoption identity index is detached from its receipt")
        return receipt

    def prepare_intent(self, intent: DocumentCommitIntent) -> DocumentCommitIntent:
        path = self._intent_path(intent.tenant_id, intent.owner_user_id, intent.intent_id)
        try:
            atomic_create_json(path, intent.to_dict(), artifact_root=self._artifact_root(intent.tenant_id))
        except ImmutableArtifactConflictError:
            # A concurrent idempotent preparer may have won create-only publication.
            pass
        durable = self.load_intent(intent.tenant_id, intent.owner_user_id, intent.intent_id)
        if durable is None:  # pragma: no cover - create-only publication cannot disappear cooperatively.
            raise DocumentControlIntegrityError("prepared document intent disappeared after publication")
        return durable

    def load_intent(self, tenant_id: str, owner_user_id: str, intent_id: str) -> DocumentCommitIntent | None:
        path = self._intent_path(tenant_id, owner_user_id, intent_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        intent = DocumentCommitIntent.from_dict(payload)
        if intent.intent_id != intent_id or intent.tenant_id != tenant_id or intent.owner_user_id != owner_user_id:
            raise DocumentControlIntegrityError("document intent path identity does not match its payload")
        return intent

    def update_intent(
        self,
        intent: DocumentCommitIntent,
        status: DocumentIntentStatus,
        *,
        updated_at: str,
        conflict_reason: str = "",
    ) -> DocumentCommitIntent:
        current = self.load_intent(intent.tenant_id, intent.owner_user_id, intent.intent_id)
        if current is None or current.identity_digest != intent.identity_digest:
            raise DocumentControlIntegrityError("document intent update is detached from its immutable identity")
        if current.status in {DocumentIntentStatus.COMPLETED, DocumentIntentStatus.CONFLICTED}:
            if current.status != status:
                return current
            return current
        if status != DocumentIntentStatus.CONFLICTED and _status_rank(status) < _status_rank(current.status):
            return current
        updated = replace(current, status=status, updated_at=updated_at, conflict_reason=conflict_reason[:500])
        atomic_write_json(
            self._intent_path(updated.tenant_id, updated.owner_user_id, updated.intent_id),
            updated.to_dict(),
            artifact_root=self._artifact_root(updated.tenant_id),
        )
        if status == DocumentIntentStatus.CONFLICTED:
            atomic_write_json(
                self._conflict_path(updated.tenant_id, updated.owner_user_id, updated.intent_id),
                updated.to_dict(),
                artifact_root=self._artifact_root(updated.tenant_id),
            )
        return updated

    def intents(self, tenant_id: str, owner_user_id: str) -> tuple[DocumentCommitIntent, ...]:
        """Enumerate every durable intent, including terminal references."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "intents"
        names = self._json_names(directory, tenant)
        intents: list[DocumentCommitIntent] = []
        for name in names:
            intent = self.load_intent(tenant, owner, name.removesuffix(".json"))
            if intent is not None:
                intents.append(intent)
        return tuple(sorted(intents, key=lambda item: (item.created_at, item.intent_id)))

    def incomplete_intents(self, tenant_id: str, owner_user_id: str) -> tuple[DocumentCommitIntent, ...]:
        return tuple(
            intent
            for intent in self.intents(tenant_id, owner_user_id)
            if intent.status != DocumentIntentStatus.COMPLETED
        )

    def append_event(self, intent: DocumentCommitIntent, event: DocumentChangeEvent) -> None:
        if not _event_matches_intent(event, intent):
            raise ValueError("document event is detached from its prepared intent")
        payload = {
            "schema": _EVENT_SCHEMA,
            "intent_id": intent.intent_id,
            "intent_identity_digest": intent.identity_digest,
            **event.to_dict(),
        }
        atomic_create_json(
            self._event_path(intent, event.event_id),
            payload,
            artifact_root=self._artifact_root(intent.tenant_id),
        )

    def load_event(self, intent: DocumentCommitIntent) -> DocumentChangeEvent | None:
        payload = self._read_json(self._event_path(intent, intent.event_id), intent.tenant_id)
        if payload is None:
            return None
        if (
            payload.get("schema") != _EVENT_SCHEMA
            or payload.get("intent_id") != intent.intent_id
            or payload.get("intent_identity_digest") != intent.identity_digest
        ):
            raise DocumentControlIntegrityError("document event is detached from its intent")
        try:
            event = DocumentChangeEvent(
                event_id=str(payload["event_id"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                edit_kind=DocumentEditKind(str(payload["edit_kind"])),
                old_relative_path=str(payload.get("old_relative_path") or ""),
                new_relative_path=str(payload.get("new_relative_path") or ""),
                before_raw_digest=str(payload.get("before_raw_digest") or ""),
                after_raw_digest=str(payload.get("after_raw_digest") or ""),
                logical_revision=int(payload["logical_revision"]),
                projection_generation=int(payload["projection_generation"]),
                occurred_at=str(payload["occurred_at"]),
                actor_binding=str(payload["actor_binding"]),
                evidence_reference=str(payload["evidence_reference"]),
                evidence_digest=str(payload["evidence_digest"]),
                edit_summary=str(payload["edit_summary"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document event is malformed") from exc
        if not _event_matches_intent(event, intent):
            raise DocumentControlIntegrityError("document event identity does not match its path")
        return event

    def load_event_binding(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        event_id: str,
    ) -> tuple[DocumentCommitIntent, DocumentChangeEvent] | None:
        """Resolve one immutable change-event ID without persisting its intent ID elsewhere."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        _validate_prefixed_digest(event_id, "memchg_", "event_id")
        directory = self._owner_root(tenant, owner) / "events" / identifier
        if not directory.exists():
            return None
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_LINEAGE_EVENTS:
            raise DocumentControlIntegrityError("document event count exceeds its bounded limit")
        suffix = f"-{event_id}.json"
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) > 1:
            raise DocumentControlIntegrityError("document event ID is duplicated")
        if not matches:
            return None
        name = matches[0]
        prefix = name.removesuffix(suffix)
        if len(prefix) != 20 or not prefix.isdigit() or "/" in name:
            raise DocumentControlIntegrityError("document event path is malformed")
        payload = self._read_json(directory / name, tenant)
        if payload is None:
            return None
        intent_id = str(payload.get("intent_id") or "")
        intent = self.load_intent(tenant, owner, intent_id)
        if intent is None or intent.document_id != identifier or intent.event_id != event_id:
            raise DocumentControlIntegrityError("document event is detached from its intent")
        event = self.load_event(intent)
        if event is None or event.event_id != event_id:
            raise DocumentControlIntegrityError("document event binding disappeared")
        return intent, event

    def write_control(self, record: DocumentControlRecord) -> None:
        atomic_write_json(
            self._control_path(record.tenant_id, record.owner_user_id, record.document_id),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )

    def load_control(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentControlRecord | None:
        path = self._control_path(tenant_id, owner_user_id, document_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        record = DocumentControlRecord.from_dict(payload)
        if (record.tenant_id, record.owner_user_id, record.document_id) != (
            tenant_id,
            owner_user_id,
            document_id,
        ):
            raise DocumentControlIntegrityError("document control path identity does not match its payload")
        return record

    def controls(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[DocumentControlRecord, ...]:
        """Enumerate one owner's exact durable control snapshot safely."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "documents"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = tuple(sorted(os.listdir(descriptor)))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_DOCUMENT_CONTROLS:
            raise DocumentControlIntegrityError("document control count exceeds its bound")
        records: list[DocumentControlRecord] = []
        present_paths: set[str] = set()
        for name in names:
            if not name.endswith(".json") or "/" in name:
                raise DocumentControlIntegrityError(
                    "document control directory contains an unexpected artifact"
                )
            try:
                document_id = validate_document_id(name.removesuffix(".json"))
            except ValueError as exc:
                raise DocumentControlIntegrityError("document control filename is invalid") from exc
            record = self.load_control(tenant, owner, document_id)
            if record is None:  # pragma: no cover - stable directory snapshot cannot lose a cooperative file.
                raise DocumentControlIntegrityError("document control disappeared during enumeration")
            if record.status == "present":
                if record.relative_path in present_paths:
                    raise DocumentControlIntegrityError(
                        "multiple present document controls claim one relative path"
                    )
                present_paths.add(record.relative_path)
            records.append(record)
        return tuple(records)

    def write_publication_barrier(
        self,
        barrier: DocumentPublicationBarrier,
    ) -> DocumentPublicationBarrier:
        """Publish one monotonic deletion fence outside rebuildable serving state."""

        current = self.load_publication_barrier(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
        )
        if current is not None:
            current_identity = (
                current.relative_path_digest,
                current.deletion_generation,
                current.deletion_event_digest,
                current.status,
            )
            requested_identity = (
                barrier.relative_path_digest,
                barrier.deletion_generation,
                barrier.deletion_event_digest,
                barrier.status,
            )
            if current.status is DocumentDeletionStatus.HARD_ERASED:
                same_erasure_identity = (
                    barrier.relative_path_digest == current.relative_path_digest
                    and barrier.deletion_event_digest == current.deletion_event_digest
                    and barrier.status is current.status
                )
                if not same_erasure_identity or barrier.deletion_generation < current.deletion_generation:
                    raise DocumentControlIntegrityError("hard-erased document publication barrier is immutable")
                if barrier.deletion_generation == current.deletion_generation:
                    return current
            if barrier.deletion_generation < current.deletion_generation:
                raise DocumentControlIntegrityError("document publication barrier generation regressed")
            if barrier.deletion_generation == current.deletion_generation:
                if requested_identity != current_identity:
                    raise DocumentControlIntegrityError(
                        "document publication barrier conflicts at the current generation"
                    )
                return current
        atomic_write_json(
            self._publication_barrier_path(
                barrier.tenant_id,
                barrier.owner_user_id,
                barrier.document_id,
            ),
            barrier.to_dict(),
            artifact_root=self._artifact_root(barrier.tenant_id),
        )
        durable = self.load_publication_barrier(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
        )
        if durable is None:  # pragma: no cover - durable publication cannot disappear cooperatively.
            raise DocumentControlIntegrityError("document publication barrier disappeared after publication")
        return durable

    def scrub_hard_erasure_path(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        *,
        expected_relative_path_digest: str,
        expected_deletion_event_digest: str,
        updated_at: str,
    ) -> DocumentPublicationBarrier:
        """Remove a semantic path after every hard-erasure backend has acknowledged."""

        current = self.load_publication_barrier(tenant_id, owner_user_id, document_id)
        if current is None:
            raise DocumentControlIntegrityError("hard-erasure publication barrier is missing")
        if (
            current.status is not DocumentDeletionStatus.HARD_ERASED
            or current.relative_path_digest != expected_relative_path_digest
            or current.deletion_event_digest != expected_deletion_event_digest
        ):
            raise DocumentControlIntegrityError("hard-erasure publication barrier identity changed")
        if not current.relative_path:
            return current
        scrubbed = replace(current, relative_path="", updated_at=updated_at)
        atomic_write_json(
            self._publication_barrier_path(tenant_id, owner_user_id, document_id),
            scrubbed.to_dict(),
            artifact_root=self._artifact_root(tenant_id),
        )
        durable = self.load_publication_barrier(tenant_id, owner_user_id, document_id)
        if durable is None or durable != scrubbed:
            raise DocumentControlIntegrityError("hard-erasure path scrub was not durable")
        return durable

    def ensure_soft_forget_barrier(
        self,
        intent: DocumentCommitIntent,
    ) -> DocumentPublicationBarrier:
        """Fence a DELETE intent before its live bytes may be unlinked."""

        if intent.edit_kind is not DocumentEditKind.DELETE or len(intent.effects) != 1:
            raise ValueError("soft-forget publication barrier requires one DELETE intent")
        effect = intent.effects[0]
        if not isinstance(effect.before, PresentPath) or not isinstance(effect.after, AbsentPath):
            raise ValueError("soft-forget publication barrier requires PRESENT to ABSENT")
        digest = deletion_event_digest(
            event_id=intent.event_id,
            document_id=intent.document_id,
            before_raw_digest=effect.before.raw_sha256,
            projection_generation=intent.projection_generation,
        )
        return self.write_publication_barrier(
            DocumentPublicationBarrier(
                tenant_id=intent.tenant_id,
                owner_user_id=intent.owner_user_id,
                document_id=intent.document_id,
                relative_path=effect.relative_path,
                deletion_generation=intent.projection_generation,
                deletion_event_digest=digest,
                status=DocumentDeletionStatus.SOFT_FORGOTTEN,
                updated_at=intent.updated_at,
            )
        )

    def load_publication_barrier(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentPublicationBarrier | None:
        path = self._publication_barrier_path(tenant_id, owner_user_id, document_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        barrier = DocumentPublicationBarrier.from_dict(payload)
        if (barrier.tenant_id, barrier.owner_user_id, barrier.document_id) != (
            tenant_id,
            owner_user_id,
            document_id,
        ):
            raise DocumentControlIntegrityError(
                "document publication barrier path identity does not match its payload"
            )
        return barrier

    def publication_barriers(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[DocumentPublicationBarrier, ...]:
        """Read the bounded protected barrier set used by offline/full rebuilds."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "publication-barriers"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_PUBLICATION_BARRIERS:
            raise DocumentControlIntegrityError("document publication barrier count exceeds its bound")
        barriers: list[DocumentPublicationBarrier] = []
        for name in names:
            if not name.endswith(".json") or "/" in name:
                raise DocumentControlIntegrityError(
                    "document publication barrier directory contains an unexpected artifact"
                )
            try:
                document_id = validate_document_id(name.removesuffix(".json"))
            except ValueError as exc:
                raise DocumentControlIntegrityError(
                    "document publication barrier filename is invalid"
                ) from exc
            barrier = self.load_publication_barrier(tenant, owner, document_id)
            if barrier is None:  # pragma: no cover - stable directory snapshot cannot lose a cooperative file.
                raise DocumentControlIntegrityError("document publication barrier disappeared during scan")
            barriers.append(barrier)
        return tuple(barriers)

    def lineage_references(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> tuple[str, ...]:
        """Return bounded, content-free evidence references for one document.

        The document-specific event directory is the durable lineage index.  A
        hard erase reads it before removing control artifacts so the caller can
        disclose independent Session evidence without retaining document text.
        """

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        directory = self._owner_root(tenant, owner) / "events" / identifier
        if not directory.exists():
            return ()
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_LINEAGE_EVENTS:
            raise DocumentControlIntegrityError("document lineage exceeds its bounded event limit")
        references: set[str] = set()
        for name in names:
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
                raise DocumentControlIntegrityError("document lineage contains an unexpected event artifact")
            payload = self._read_json(directory / name, tenant)
            if (
                payload is None
                or payload.get("schema") != _EVENT_SCHEMA
                or payload.get("tenant_id") != tenant
                or payload.get("owner_user_id") != owner
                or payload.get("document_id") != identifier
            ):
                raise DocumentControlIntegrityError("document lineage event identity is invalid")
            reference = str(payload.get("evidence_reference") or "")
            if reference:
                references.add(reference)
        return tuple(sorted(references))

    def purge_document(self, tenant_id: str, owner_user_id: str, document_id: str) -> int:
        """Durably remove document commit metadata after live/body erasure.

        The erasure tombstone is stored outside these paths.  An unfinished
        commit is never discarded because it may still own a live CAS.  The
        content-free adoption receipt is intentionally retained beside the
        tombstone so the hard-erased assigned identity cannot be reused.
        """

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        matching: list[DocumentCommitIntent] = []
        for name in self._json_names(self._owner_root(tenant, owner) / "intents", tenant):
            intent = self.load_intent(tenant, owner, name.removesuffix(".json"))
            if intent is not None and intent.document_id == identifier:
                matching.append(intent)
        unfinished = [
            intent.intent_id
            for intent in matching
            if intent.status not in {DocumentIntentStatus.COMPLETED, DocumentIntentStatus.CONFLICTED}
        ]
        if unfinished:
            raise DocumentControlIntegrityError("cannot purge a document with an unfinished commit intent")

        removed = 0
        for intent in matching:
            removed += self._unlink_regular_if_present(
                self._intent_path(tenant, owner, intent.intent_id),
                tenant,
            )
            removed += self._unlink_regular_if_present(
                self._conflict_path(tenant, owner, intent.intent_id),
                tenant,
            )
        removed += self._purge_event_directory(tenant, owner, identifier)
        removed += self._unlink_regular_if_present(
            self._control_path(tenant, owner, identifier),
            tenant,
        )
        return removed

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


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DocumentControlIntegrityError(f"{label} must be a JSON object")
    return value


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _validate_prefixed_digest(value: str, prefix: str, label: str) -> None:
    if not value.startswith(prefix) or not _is_hex(value.removeprefix(prefix), 64):
        raise ValueError(f"{label} is invalid")


def _status_rank(status: DocumentIntentStatus) -> int:
    return {
        DocumentIntentStatus.PREPARED: 0,
        DocumentIntentStatus.INSTALLED: 1,
        DocumentIntentStatus.EVENT_APPENDED: 2,
        DocumentIntentStatus.PROJECTION_ENQUEUED: 3,
        DocumentIntentStatus.COMPLETED: 4,
        DocumentIntentStatus.CONFLICTED: 5,
    }[status]


def _event_matches_intent(event: DocumentChangeEvent, intent: DocumentCommitIntent) -> bool:
    before_digest = next(
        (state.raw_sha256 for state in (effect.before for effect in intent.effects) if isinstance(state, PresentPath)),
        "",
    )
    after_digest = next(
        (
            state.raw_sha256
            for state in (effect.after for effect in reversed(intent.effects))
            if isinstance(state, PresentPath)
        ),
        "",
    )
    return (
        event.event_id == intent.event_id
        and event.tenant_id == intent.tenant_id
        and event.owner_user_id == intent.owner_user_id
        and event.document_id == intent.document_id
        and event.edit_kind == intent.edit_kind
        and event.old_relative_path == intent.old_relative_path
        and event.new_relative_path == intent.new_relative_path
        and event.before_raw_digest == before_digest
        and event.after_raw_digest == after_digest
        and event.logical_revision == intent.logical_revision
        and event.projection_generation == intent.projection_generation
        and event.occurred_at == intent.created_at
        and event.actor_binding == intent.actor_binding
        and event.evidence_reference == intent.evidence_reference
        and event.evidence_digest == intent.evidence_digest
        and event.edit_summary == intent.edit_summary
    )


__all__ = [
    "DocumentAdoptionReceipt",
    "DocumentCommitIntent",
    "DocumentControlIntegrityError",
    "DocumentControlRecord",
    "DocumentDeletionStatus",
    "DocumentIntentStatus",
    "DocumentPathEffect",
    "DocumentPublicationBarrier",
    "DocumentRootIdentity",
    "MemoryDocumentControlStore",
    "adoption_document_id",
    "adoption_request_digest",
    "deletion_event_digest",
    "document_intent_id",
    "document_intent_identity_digest",
]
