"""记忆提案与显著性预约的耐久证据存储。"""

from infrastructure.store.memory.evidence.erase_backend import SealedProposalEraseBackend
from infrastructure.store.memory.evidence.proposal_store import SealedProposalStore
from infrastructure.store.memory.evidence.salience_ledger import (
    DurableSalienceLedger,
    SalienceLedgerIntegrityError,
    SalienceReservationResult,
)
from memory.core.model.sealed_proposal import (
    ProposalDocumentBinding,
    SealedProposalBindingSet,
    SealedProposalIntegrityError,
    SealedProposalSet,
)

__all__ = [
    "DurableSalienceLedger",
    "ProposalDocumentBinding",
    "SalienceLedgerIntegrityError",
    "SalienceReservationResult",
    "SealedProposalBindingSet",
    "SealedProposalEraseBackend",
    "SealedProposalIntegrityError",
    "SealedProposalSet",
    "SealedProposalStore",
]
