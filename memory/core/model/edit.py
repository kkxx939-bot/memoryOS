"""记忆写入计划和提交后领域事件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.core.model.state import ABSENT, DocumentEditKind, RawPathState


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


__all__ = ["DocumentChangeEvent", "DocumentEditPlan"]
