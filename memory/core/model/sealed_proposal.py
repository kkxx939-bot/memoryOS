"""已封存记忆提案及其无正文文档绑定模型。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest
from memory.core.model.proposal import MemoryEditProposal
from memory.core.structure.frontmatter import validate_document_id


class SealedProposalIntegrityError(RuntimeError):
    """包含正文的提案产物或其精确血缘不安全。"""


@dataclass(frozen=True)
class SealedProposalSet:
    task_id: str
    tenant_id: str
    owner_user_id: str
    archive_uri: str
    archive_digest: str
    manifest_digest: str
    proposals: tuple[MemoryEditProposal, ...]
    proposal_set_digest: str


@dataclass(frozen=True)
class ProposalDocumentBinding:
    """一条不含正文的任务到文档副作用指纹。"""

    document_id: str
    change_digest: str

    def __post_init__(self) -> None:
        validate_document_id(self.document_id)
        _require_digest(self.change_digest, "proposal document change digest")

    def to_dict(self) -> dict[str, str]:
        return {"document_id": self.document_id, "change_digest": self.change_digest}

    @classmethod
    def from_dict(cls, payload: object) -> ProposalDocumentBinding:
        if not isinstance(payload, dict) or set(payload) != {"document_id", "change_digest"}:
            raise SealedProposalIntegrityError("proposal document binding is malformed")
        try:
            return cls(
                document_id=str(payload["document_id"]),
                change_digest=str(payload["change_digest"]),
            )
        except (TypeError, ValueError) as exc:
            raise SealedProposalIntegrityError("proposal document binding is invalid") from exc


@dataclass(frozen=True)
class SealedProposalBindingSet:
    """一个已封存 Session 任务的精确无正文血缘。"""

    task_id: str
    tenant_id: str
    owner_user_id: str
    proposal_set_digest: str
    documents: tuple[ProposalDocumentBinding, ...]
    binding_digest: str

    def __post_init__(self) -> None:
        _validate_task_id(self.task_id)
        require_safe_path_segment(self.tenant_id, "proposal binding tenant_id")
        require_safe_path_segment(self.owner_user_id, "proposal binding owner_user_id")
        _require_digest(self.proposal_set_digest, "proposal set digest")
        if not self.documents:
            raise ValueError("proposal binding requires at least one document")
        if tuple(sorted(self.documents, key=lambda item: item.document_id)) != self.documents:
            raise ValueError("proposal bindings must be sorted by document identity")
        if len({item.document_id for item in self.documents}) != len(self.documents):
            raise ValueError("proposal binding repeats a document identity")
        if self.binding_digest != canonical_digest(self._digest_payload()):
            raise ValueError("proposal binding digest is invalid")

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "proposal_set_digest": self.proposal_set_digest,
            "documents": [item.to_dict() for item in self.documents],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._digest_payload(), "binding_digest": self.binding_digest}

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        tenant_id: str,
        owner_user_id: str,
        proposal_set_digest: str,
        documents: Iterable[ProposalDocumentBinding],
    ) -> SealedProposalBindingSet:
        ordered = tuple(sorted(documents, key=lambda item: item.document_id))
        payload = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "proposal_set_digest": proposal_set_digest,
            "documents": [item.to_dict() for item in ordered],
        }
        return cls(
            task_id=task_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            proposal_set_digest=proposal_set_digest,
            documents=ordered,
            binding_digest=canonical_digest(payload),
        )

    @classmethod
    def from_dict(cls, payload: object) -> SealedProposalBindingSet:
        expected = {
            "task_id",
            "tenant_id",
            "owner_user_id",
            "proposal_set_digest",
            "documents",
            "binding_digest",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise SealedProposalIntegrityError("sealed proposal binding is malformed")
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise SealedProposalIntegrityError("sealed proposal documents must be an array")
        try:
            return cls(
                task_id=str(payload["task_id"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                proposal_set_digest=str(payload["proposal_set_digest"]),
                documents=tuple(ProposalDocumentBinding.from_dict(item) for item in raw_documents),
                binding_digest=str(payload["binding_digest"]),
            )
        except (TypeError, ValueError) as exc:
            raise SealedProposalIntegrityError("sealed proposal binding is invalid") from exc



def _validate_task_id(task_id: object) -> str:
    if not isinstance(task_id, str) or not task_id or len(task_id.encode("utf-8")) > 512:
        raise ValueError("proposal task_id is empty or too large")
    if any(ord(character) < 32 for character in task_id):
        raise ValueError("proposal task_id contains control characters")
    return task_id


def _require_digest(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest



__all__ = [
    "ProposalDocumentBinding",
    "SealedProposalBindingSet",
    "SealedProposalIntegrityError",
    "SealedProposalSet",
]
