"""Hard-erasure backend for body-bearing sealed Session proposals."""

from __future__ import annotations

from memoryos.memory.documents.erase import DerivedEraseRequest
from memoryos.memory.evidence.proposal_store import SealedProposalStore


class SealedProposalEraseBackend:
    """Delete proposal sets only through their exact content-free document binding."""

    name = "derived.sealed_proposals"

    def __init__(self, store: SealedProposalStore) -> None:
        self.store = store

    def erase_document(self, request: DerivedEraseRequest) -> bool:
        if request.tenant_id != self.store.tenant_id:
            raise ValueError("sealed proposal cleanup crosses its configured tenant")
        self.store.delete_for_document(
            request.owner_user_id,
            request.document_id,
            erasure_epoch=request.erasure_epoch,
        )
        return True


__all__ = ["SealedProposalEraseBackend"]
