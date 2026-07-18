"""Domain models for user-editable Markdown memory documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class MemoryDocumentKind(str, Enum):
    ROOT_INDEX = "root_index"
    PROFILE = "profile"
    PREFERENCES = "preferences"
    KNOWLEDGE_INDEX = "knowledge_index"
    ENTITY = "entity"
    TOPIC = "topic"
    EPISODE = "episode"
    OPEN_LOOPS = "open_loops"
    EXPERIENCE = "experience"


class MemoryCandidateKind(str, Enum):
    PROFILE_FACT = "profile_fact"
    PREFERENCE = "preference"
    ENTITY_NOTE = "entity_note"
    TOPIC_NOTE = "topic_note"
    EPISODE = "episode"
    OPEN_LOOP = "open_loop"
    EXPERIENCE = "experience"


class DocumentEditKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


class RegistrationStatus(str, Enum):
    MANAGED = "managed"
    UNMANAGED = "unmanaged"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class AbsentPath:
    """The controlled relative path does not exist."""


@dataclass(frozen=True)
class PresentPath:
    relative_path: str
    raw_sha256: str
    size: int

    def __post_init__(self) -> None:
        if not self.relative_path or len(self.raw_sha256) != 64 or self.size < 0:
            raise ValueError("invalid PRESENT raw path state")


@dataclass(frozen=True)
class UnsafePath:
    relative_path: str
    reason: str

    def __post_init__(self) -> None:
        if not self.relative_path or not self.reason:
            raise ValueError("invalid UNSAFE raw path state")


RawPathState = AbsentPath | PresentPath | UnsafePath
ABSENT = AbsentPath()


@dataclass(frozen=True)
class ManagedDocument:
    relative_path: str
    document_id: str
    raw_sha256: str
    size: int
    status: RegistrationStatus = field(default=RegistrationStatus.MANAGED, init=False)


@dataclass(frozen=True)
class UnmanagedDocument:
    relative_path: str
    raw_sha256: str
    size: int
    reason: str
    status: RegistrationStatus = field(default=RegistrationStatus.UNMANAGED, init=False)


@dataclass(frozen=True)
class QuarantinedDocument:
    relative_path: str
    reason: str
    raw_sha256: str = ""
    size: int = 0
    status: RegistrationStatus = field(default=RegistrationStatus.QUARANTINED, init=False)


DocumentRegistrationState = ManagedDocument | UnmanagedDocument | QuarantinedDocument


@dataclass(frozen=True)
class MemoryDocument:
    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    document_kind: MemoryDocumentKind
    raw_sha256: str
    size: int
    raw_bytes: bytes = field(repr=False)
    body: str = field(repr=False)
    front_matter: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "front_matter", MappingProxyType(dict(self.front_matter)))

    @property
    def uri(self) -> str:
        return f"memoryos://user/{self.owner_user_id}/memory/documents/{self.document_id}"


@dataclass(frozen=True)
class ScanGeneration:
    generation_id: str
    tenant_id: str
    owner_user_id: str
    root_identity: str
    observed_at: str
    complete: bool
    registrations: tuple[DocumentRegistrationState, ...] = ()
    unsafe_paths: tuple[UnsafePath, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def managed(self) -> tuple[ManagedDocument, ...]:
        return tuple(item for item in self.registrations if isinstance(item, ManagedDocument))


@dataclass(frozen=True)
class MemoryEditProposal:
    candidate_kind: MemoryCandidateKind
    title: str
    body: str
    evidence_refs: tuple[str, ...]
    subject: str = ""
    entity_hints: tuple[str, ...] = ()
    topic_hints: tuple[str, ...] = ()
    occurred_at: str = ""
    temporal_status: str = ""
    relation_hints: tuple[str, ...] = ()
    field_evidence_refs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.body.strip() or not self.evidence_refs:
            raise ValueError("memory edit proposal requires title, body and evidence")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("memory edit proposal confidence must be between zero and one")
        object.__setattr__(self, "field_evidence_refs", MappingProxyType(dict(self.field_evidence_refs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_kind": self.candidate_kind.value,
            "title": self.title,
            "body": self.body,
            "evidence_refs": list(self.evidence_refs),
            "subject": self.subject,
            "entity_hints": list(self.entity_hints),
            "topic_hints": list(self.topic_hints),
            "occurred_at": self.occurred_at,
            "temporal_status": self.temporal_status,
            "relation_hints": list(self.relation_hints),
            "field_evidence_refs": {
                key: list(values) for key, values in sorted(self.field_evidence_refs.items())
            },
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryEditProposal:
        allowed = {
            "candidate_kind",
            "title",
            "body",
            "evidence_refs",
            "subject",
            "entity_hints",
            "topic_hints",
            "occurred_at",
            "temporal_status",
            "relation_hints",
            "field_evidence_refs",
            "confidence",
        }
        if set(payload) - allowed:
            raise ValueError("sealed memory proposal contains unsupported fields")
        field_refs = payload.get("field_evidence_refs", {})
        if not isinstance(field_refs, Mapping):
            raise ValueError("sealed field evidence must be an object")
        return cls(
            candidate_kind=MemoryCandidateKind(str(payload["candidate_kind"])),
            title=str(payload["title"]),
            body=str(payload["body"]),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
            subject=str(payload.get("subject") or ""),
            entity_hints=tuple(str(item) for item in payload.get("entity_hints", [])),
            topic_hints=tuple(str(item) for item in payload.get("topic_hints", [])),
            occurred_at=str(payload.get("occurred_at") or ""),
            temporal_status=str(payload.get("temporal_status") or ""),
            relation_hints=tuple(str(item) for item in payload.get("relation_hints", [])),
            field_evidence_refs={
                str(key): tuple(str(item) for item in values)
                for key, values in field_refs.items()
            },
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class DocumentEditPlan:
    idempotency_key: str
    tenant_id: str
    owner_user_id: str
    edit_kind: DocumentEditKind
    expected_state: RawPathState
    evidence_digest: str
    edit_summary: str
    document_id: str = ""
    relative_path: str = ""
    after_bytes: bytes | None = field(default=None, repr=False)
    new_relative_path: str = ""
    expected_new_state: RawPathState = ABSENT
    expected_registration_document_id: str = ""


@dataclass(frozen=True)
class DocumentChangeEvent:
    event_id: str
    tenant_id: str
    owner_user_id: str
    document_id: str
    edit_kind: DocumentEditKind
    old_relative_path: str
    new_relative_path: str
    before_raw_digest: str
    after_raw_digest: str
    logical_revision: int
    projection_generation: int
    occurred_at: str
    actor_binding: str
    evidence_reference: str
    evidence_digest: str
    edit_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "document_id": self.document_id,
            "edit_kind": self.edit_kind.value,
            "old_relative_path": self.old_relative_path,
            "new_relative_path": self.new_relative_path,
            "before_raw_digest": self.before_raw_digest,
            "after_raw_digest": self.after_raw_digest,
            "logical_revision": self.logical_revision,
            "projection_generation": self.projection_generation,
            "occurred_at": self.occurred_at,
            "actor_binding": self.actor_binding,
            "evidence_reference": self.evidence_reference,
            "evidence_digest": self.evidence_digest,
            "edit_summary": self.edit_summary,
        }


def raw_state_to_dict(state: RawPathState) -> dict[str, Any]:
    if isinstance(state, AbsentPath):
        return {"state": "ABSENT"}
    if isinstance(state, PresentPath):
        return {
            "state": "PRESENT",
            "relative_path": state.relative_path,
            "raw_sha256": state.raw_sha256,
            "size": state.size,
        }
    return {"state": "UNSAFE", "relative_path": state.relative_path, "reason": state.reason}


def raw_state_from_dict(payload: Mapping[str, Any]) -> RawPathState:
    state = str(payload.get("state") or "")
    if state == "ABSENT":
        return ABSENT
    if state == "PRESENT":
        return PresentPath(
            relative_path=str(payload["relative_path"]),
            raw_sha256=str(payload["raw_sha256"]),
            size=int(payload["size"]),
        )
    if state == "UNSAFE":
        return UnsafePath(relative_path=str(payload["relative_path"]), reason=str(payload["reason"]))
    raise ValueError("unknown raw path state")


__all__ = [
    "ABSENT",
    "AbsentPath",
    "DocumentChangeEvent",
    "DocumentEditKind",
    "DocumentEditPlan",
    "DocumentRegistrationState",
    "ManagedDocument",
    "MemoryCandidateKind",
    "MemoryDocument",
    "MemoryDocumentKind",
    "MemoryEditProposal",
    "PresentPath",
    "QuarantinedDocument",
    "RawPathState",
    "RegistrationStatus",
    "ScanGeneration",
    "UnmanagedDocument",
    "UnsafePath",
    "raw_state_from_dict",
    "raw_state_to_dict",
]
