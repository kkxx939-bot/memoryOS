"""对已封存且包含正文的 Session 提案执行硬删除。"""

from __future__ import annotations

from infrastructure.store.memory.evidence.proposal_store import SealedProposalStore
from memory.ports.erase import DerivedEraseRequest


class SealedProposalEraseBackend:
    """只通过精确的无正文文档绑定删除提案集合。"""

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
