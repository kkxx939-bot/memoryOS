"""Immutable SessionArchive evidence used by memory extraction and projection."""

from memoryos.memory.evidence.archive_encoder import SessionEvidenceArchiveEncoder
from memoryos.memory.evidence.episode import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.memory.evidence.erase_backend import SealedProposalEraseBackend
from memoryos.memory.evidence.event import ActorRef, EventEnvelope, OriginContext, SubjectRef
from memoryos.memory.evidence.proposal_store import (
    ProposalDocumentBinding,
    SealedProposalBindingSet,
    SealedProposalIntegrityError,
    SealedProposalSet,
    SealedProposalStore,
)
from memoryos.memory.evidence.salience import EpisodeSalienceGate, SalienceDecision, SalienceFactor
from memoryos.memory.evidence.salience_ledger import (
    DurableSalienceLedger,
    SalienceLedgerIntegrityError,
    SalienceReservationResult,
)
from memoryos.memory.evidence.scope import ScopeRef, ScopeResolutionSource, scope_from_external

__all__ = [
    "ActorRef",
    "DurableSalienceLedger",
    "EpisodeSalienceGate",
    "EvidenceEpisode",
    "EventEnvelope",
    "OriginContext",
    "SalienceDecision",
    "SalienceFactor",
    "SalienceLedgerIntegrityError",
    "SalienceReservationResult",
    "SealedProposalBindingSet",
    "SealedProposalEraseBackend",
    "SealedProposalIntegrityError",
    "SealedProposalSet",
    "SealedProposalStore",
    "ScopeRef",
    "ScopeResolutionSource",
    "SessionEvidenceArchiveEncoder",
    "SessionArchiveEpisodeAdapter",
    "SubjectRef",
    "ProposalDocumentBinding",
    "scope_from_external",
]
