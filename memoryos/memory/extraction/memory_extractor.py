"""Protocol for semantic candidate extraction from immutable Session evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.documents.model import MemoryEditProposal
from memoryos.memory.schema import MemoryCandidateSchema


class MemoryExtractorBackend(Protocol):
    semantic_proposal_backend: bool
    llm_semantic_backend: bool

    @property
    def is_remote(self) -> bool: ...

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
    ) -> Sequence[MemoryEditProposal]: ...


__all__ = ["MemoryExtractorBackend"]
