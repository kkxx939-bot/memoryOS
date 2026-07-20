"""记忆领域模型的统一导出入口。"""

from memory.core.model.document import MemoryDocument, MemoryDocumentKind
from memory.core.model.edit import DocumentChangeEvent, DocumentEditPlan
from memory.core.model.proposal import MemoryCandidateKind, MemoryEditProposal
from memory.core.model.sealed_proposal import (
    ProposalDocumentBinding,
    SealedProposalBindingSet,
    SealedProposalIntegrityError,
    SealedProposalSet,
)
from memory.core.model.state import (
    ABSENT,
    AbsentPath,
    DocumentEditKind,
    DocumentRegistrationState,
    ManagedDocument,
    PresentPath,
    QuarantinedDocument,
    RawPathState,
    RegistrationStatus,
    ScanGeneration,
    UnmanagedDocument,
    UnsafePath,
    raw_state_from_dict,
    raw_state_to_dict,
)

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
    "ProposalDocumentBinding",
    "QuarantinedDocument",
    "RawPathState",
    "RegistrationStatus",
    "ScanGeneration",
    "SealedProposalBindingSet",
    "SealedProposalIntegrityError",
    "SealedProposalSet",
    "UnmanagedDocument",
    "UnsafePath",
    "raw_state_from_dict",
    "raw_state_to_dict",
]
