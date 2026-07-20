"""多文档合并事务的共享模型和持久化协议。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from foundation.integrity import canonical_json
from memory.core.model import DocumentEditPlan, PresentPath, raw_state_to_dict
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy

_SAGA_SCHEMA = "memory_document_consolidation_v1"
_MAX_SAGA_BYTES = 1024 * 1024
_MAX_SOURCES = 1_000
_MAX_SAGAS_PER_OWNER = 10_000


class ConsolidationIntegrityError(RuntimeError):
    """合并日志格式损坏或与其稳定身份脱离。"""


class ConsolidationInputRequired(RuntimeError):
    """恢复流程遇到了从未耐久准备的目标。"""


class ConsolidationStatus(str, Enum):
    PREPARED = "PREPARED"
    TARGET_COMMITTED = "TARGET_COMMITTED"
    AWAITING_TARGET_PROJECTION = "AWAITING_TARGET_PROJECTION"
    SOFT_FORGETTING = "SOFT_FORGETTING"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class ConsolidationSource:
    """一份冗余来源文档的不含正文精确状态。"""

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
    """一次有序合并的不含正文耐久进度。"""

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
    """数量受限的启动恢复结果，其中标识符不包含文档正文。"""

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
    """删除来源文档前所需的最小派生状态证明。"""

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> Mapping[str, object] | None: ...


class ConsolidationSagaStore(Protocol):
    """合并事务推进所需的耐久进度存储边界。"""

    def create(self, record: ConsolidationSagaRecord) -> ConsolidationSagaRecord: ...

    def load(
        self,
        tenant_id: str,
        owner_user_id: str,
        saga_id: str,
    ) -> ConsolidationSagaRecord | None: ...

    def save(self, record: ConsolidationSagaRecord) -> ConsolidationSagaRecord: ...

    def list_records(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = 1_000,
    ) -> tuple[ConsolidationSagaRecord, ...]: ...

    def list_pending(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = 1_000,
    ) -> tuple[ConsolidationSagaRecord, ...]: ...

    def lock(
        self,
        tenant_id: str,
        owner_user_id: str,
        saga_id: str,
    ) -> AbstractContextManager[None]: ...


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
    "ConsolidationFaultHook",
    "ConsolidationInputRequired",
    "ConsolidationIntegrityError",
    "ConsolidationProjectionReader",
    "ConsolidationRecoveryReport",
    "ConsolidationResult",
    "ConsolidationSagaRecord",
    "ConsolidationSagaStore",
    "ConsolidationSource",
    "ConsolidationStatus",
    "consolidation_identity_digest",
    "consolidation_saga_id",
]
