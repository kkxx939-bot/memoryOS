"""单个 Markdown 记忆文档提交意图及其精确路径副作用。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from foundation.integrity import canonical_json
from infrastructure.store.memory.control_common import (
    _INTENT_SCHEMA,
    DocumentControlIntegrityError,
    DocumentIntentStatus,
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
from memory.core.model import (
    AbsentPath,
    DocumentEditKind,
    PresentPath,
    RawPathState,
    UnsafePath,
    raw_state_from_dict,
    raw_state_to_dict,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


@dataclass(frozen=True)
class DocumentPathEffect:
    """单文档副作用向量中的一次精确路径转换。"""

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
    """针对一个文档身份、且不含正文的前滚日志。"""

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
        required = {
            "schema",
            "intent_id",
            "idempotency_digest",
            "tenant_id",
            "owner_user_id",
            "document_id",
            "edit_kind",
            "effects",
            "after_blob_digest",
            "revision_blob_digest",
            "revision_blob_role",
            "logical_revision",
            "projection_generation",
            "event_id",
            "projection_job_id",
            "old_relative_path",
            "new_relative_path",
            "actor_binding",
            "evidence_reference",
            "evidence_digest",
            "edit_summary",
            "restored_from_deletion_generation",
            "created_at",
            "identity_digest",
            "status",
            "updated_at",
            "conflict_reason",
        }
        if set(payload) != required:
            raise DocumentControlIntegrityError("document intent is malformed")
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
                restored_from_deletion_generation=int(payload.get("restored_from_deletion_generation", 0)),
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
    """把无正文删除屏障绑定到一个精确文档事件。"""

    _validate_prefixed_digest(event_id, "memchg_", "event_id")
    validate_document_id(document_id)
    if not _is_hex(before_raw_digest, 64) or projection_generation <= 0:
        raise ValueError("deletion event requires an exact before digest and positive generation")
    material = "\x1f".join((event_id, document_id, before_raw_digest, str(projection_generation)))
    return hashlib.sha256(material.encode()).hexdigest()
