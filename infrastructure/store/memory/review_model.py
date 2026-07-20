"""记忆编辑审核的状态、工作流和耐久记录模型。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from foundation.integrity import canonical_json
from memory.core.model import (
    DocumentEditKind,
    RawPathState,
    raw_state_from_dict,
    raw_state_to_dict,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy

_REVIEW_SCHEMA = "memory_document_review_v3"
_MAX_INDEPENDENT_EVIDENCE_REFERENCES = 1
_MAX_CONSOLIDATION_SOURCES = 100

class MemoryEditReviewStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"


class MemoryEditReviewWorkflow(str, Enum):
    DOCUMENT_EDIT = "DOCUMENT_EDIT"
    CONSOLIDATION = "CONSOLIDATION"


class MemoryEditReviewIntegrityError(RuntimeError):
    """封存审核记录或其精确正文 Blob 校验失败。"""


@dataclass(frozen=True)
class ReviewConsolidationSource:
    """封存在合并审核中的精确无正文来源绑定。"""

    document_id: str
    relative_path: str
    raw_sha256: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", validate_document_id(self.document_id))
        object.__setattr__(
            self,
            "relative_path",
            MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path),
        )
        if not _is_sha256(self.raw_sha256):
            raise ValueError("review consolidation source digest must be SHA-256")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("review consolidation source size is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "raw_sha256": self.raw_sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ReviewConsolidationSource:
        try:
            raw_size = payload["size"]
            if isinstance(raw_size, bool) or not isinstance(raw_size, int):
                raise ValueError("review consolidation source size is invalid")
            return cls(
                document_id=str(payload["document_id"]),
                relative_path=str(payload["relative_path"]),
                raw_sha256=str(payload["raw_sha256"]),
                size=raw_size,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryEditReviewIntegrityError(
                "review consolidation source is malformed"
            ) from exc


@dataclass(frozen=True)
class MemoryEditReviewRecord:
    proposal_id: str
    tenant_id: str
    owner_user_id: str
    document_id: str
    edit_kind: DocumentEditKind
    expected_state: RawPathState
    expected_new_state: RawPathState
    relative_path: str
    new_relative_path: str
    expected_registration_document_id: str
    request_id_digest: str
    evidence_digest: str
    edit_summary: str
    after_blob_digest: str
    proposed_diff_blob_digest: str
    independent_evidence_references: tuple[str, ...]
    workflow_kind: MemoryEditReviewWorkflow
    consolidation_sources: tuple[ReviewConsolidationSource, ...]
    sealed_digest: str | None
    status: MemoryEditReviewStatus
    created_at: str
    updated_at: str
    commit_intent_id: str = ""
    replacement_proposal_id: str = ""
    consolidation_saga_id: str = ""

    def __post_init__(self) -> None:
        MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        validate_document_id(self.document_id)
        _validate_proposal_id(self.proposal_id)
        MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        if self.new_relative_path:
            MemoryDocumentPathPolicy.normalize_relative_path(self.new_relative_path)
        if self.expected_registration_document_id:
            validate_document_id(self.expected_registration_document_id)
        for label, digest in (
            ("request ID digest", self.request_id_digest),
            ("evidence digest", self.evidence_digest),
            ("proposed diff digest", self.proposed_diff_blob_digest),
        ):
            if not _is_sha256(digest):
                raise ValueError(f"{label} must be a SHA-256 digest")
        if self.after_blob_digest and not _is_sha256(self.after_blob_digest):
            raise ValueError("after blob digest must be empty or SHA-256")
        if self.edit_kind in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE} and not self.after_blob_digest:
            raise ValueError("create/update review requires an exact after blob")
        if self.edit_kind is DocumentEditKind.DELETE and self.after_blob_digest:
            raise ValueError("delete review cannot invent an after blob")
        if not self.edit_summary or len(self.edit_summary) > 500:
            raise ValueError("review edit summary is empty or too large")
        if not self.created_at or not self.updated_at:
            raise ValueError("review timestamps must be non-empty")
        references = tuple(
            sorted(
                {
                    _independent_evidence_reference(item, owner_user_id=self.owner_user_id)
                    for item in self.independent_evidence_references
                }
            )
        )
        if len(references) > _MAX_INDEPENDENT_EVIDENCE_REFERENCES:
            raise ValueError("review independent evidence reference count exceeds its bound")
        object.__setattr__(self, "independent_evidence_references", references)
        workflow = MemoryEditReviewWorkflow(self.workflow_kind)
        object.__setattr__(self, "workflow_kind", workflow)
        sources = tuple(self.consolidation_sources)
        if len(sources) > _MAX_CONSOLIDATION_SOURCES:
            raise ValueError("review consolidation source count exceeds its bound")
        source_ids = tuple(source.document_id for source in sources)
        if len(set(source_ids)) != len(source_ids) or self.document_id in source_ids:
            raise ValueError("review consolidation sources must be unique and exclude the target")
        if workflow is MemoryEditReviewWorkflow.CONSOLIDATION:
            if self.edit_kind not in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE} or not sources:
                raise ValueError("consolidation review requires an exact target edit and sources")
        elif sources:
            raise ValueError("ordinary document review cannot bind consolidation sources")
        object.__setattr__(self, "consolidation_sources", sources)
        if self.status == MemoryEditReviewStatus.APPROVED:
            if workflow is MemoryEditReviewWorkflow.DOCUMENT_EDIT:
                if not self.commit_intent_id or self.consolidation_saga_id:
                    raise ValueError("approved document review requires only its commit intent")
            elif not self.consolidation_saga_id or self.commit_intent_id:
                raise ValueError("approved consolidation review requires only its saga")
        elif self.commit_intent_id or self.consolidation_saga_id:
            raise ValueError("only an approved review may name a commit intent or consolidation saga")
        if self.status == MemoryEditReviewStatus.CORRECTED and not self.replacement_proposal_id:
            raise ValueError("corrected review requires its replacement proposal")
        if self.status != MemoryEditReviewStatus.CORRECTED and self.replacement_proposal_id:
            raise ValueError("only a corrected review may name a replacement")
        expected_seal = hashlib.sha256(canonical_json(self.immutable_payload()).encode()).hexdigest()
        if self.sealed_digest is None:
            object.__setattr__(self, "sealed_digest", expected_seal)
        elif self.sealed_digest != expected_seal:
            raise ValueError("review sealed digest does not match its immutable proposal")
        expected_id = f"mdreview_{expected_seal}"
        if self.proposal_id != expected_id:
            raise ValueError("review proposal ID is detached from its sealed proposal")

    def immutable_payload(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "document_id": self.document_id,
            "edit_kind": self.edit_kind.value,
            "expected_state": raw_state_to_dict(self.expected_state),
            "expected_new_state": raw_state_to_dict(self.expected_new_state),
            "relative_path": self.relative_path,
            "new_relative_path": self.new_relative_path,
            "expected_registration_document_id": self.expected_registration_document_id,
            "request_id_digest": self.request_id_digest,
            "evidence_digest": self.evidence_digest,
            "edit_summary": self.edit_summary,
            "after_blob_digest": self.after_blob_digest,
            "proposed_diff_blob_digest": self.proposed_diff_blob_digest,
            "independent_evidence_references": list(self.independent_evidence_references),
            "workflow_kind": self.workflow_kind.value,
            "consolidation_sources": [source.to_dict() for source in self.consolidation_sources],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _REVIEW_SCHEMA,
            "proposal_id": self.proposal_id,
            **self.immutable_payload(),
            "sealed_digest": self.sealed_digest,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "commit_intent_id": self.commit_intent_id,
            "replacement_proposal_id": self.replacement_proposal_id,
            "consolidation_saga_id": self.consolidation_saga_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryEditReviewRecord:
        if payload.get("schema") != _REVIEW_SCHEMA:
            raise MemoryEditReviewIntegrityError("memory edit review schema is unsupported")
        try:
            return cls(
                proposal_id=str(payload["proposal_id"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                edit_kind=DocumentEditKind(str(payload["edit_kind"])),
                expected_state=raw_state_from_dict(_mapping(payload["expected_state"])),
                expected_new_state=raw_state_from_dict(_mapping(payload["expected_new_state"])),
                relative_path=str(payload["relative_path"]),
                new_relative_path=str(payload.get("new_relative_path") or ""),
                expected_registration_document_id=str(payload.get("expected_registration_document_id") or ""),
                request_id_digest=str(payload["request_id_digest"]),
                evidence_digest=str(payload["evidence_digest"]),
                edit_summary=str(payload["edit_summary"]),
                after_blob_digest=str(payload.get("after_blob_digest") or ""),
                proposed_diff_blob_digest=str(payload["proposed_diff_blob_digest"]),
                independent_evidence_references=tuple(
                    str(item) for item in _sequence(payload["independent_evidence_references"])
                ),
                workflow_kind=MemoryEditReviewWorkflow(str(payload["workflow_kind"])),
                consolidation_sources=tuple(
                    ReviewConsolidationSource.from_dict(_mapping(item))
                    for item in _sequence(payload["consolidation_sources"])
                ),
                sealed_digest=str(payload["sealed_digest"]),
                status=MemoryEditReviewStatus(str(payload["status"])),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                commit_intent_id=str(payload.get("commit_intent_id") or ""),
                replacement_proposal_id=str(payload.get("replacement_proposal_id") or ""),
                consolidation_saga_id=str(payload.get("consolidation_saga_id") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryEditReviewIntegrityError("memory edit review record is malformed") from exc



def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MemoryEditReviewIntegrityError("review metadata field must be an object")
    return value


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("review independent evidence references must be an array")
    return value


def _independent_evidence_reference(value: object, *, owner_user_id: str) -> str:
    reference = str(value or "")
    prefix = f"memoryos://user/{owner_user_id}/sessions/history/"
    if (
        not reference.startswith(prefix)
        or len(reference.encode("utf-8")) > 2048
        or any(ord(character) < 32 for character in reference)
    ):
        raise ValueError("review independent evidence reference must be an owner-bound Session archive URI")
    return reference


def _validate_proposal_id(value: str) -> None:
    if not value.startswith("mdreview_") or not _is_sha256(value.removeprefix("mdreview_")):
        raise ValueError("memory edit review proposal ID is invalid")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
