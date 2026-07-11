"""记忆系统里的记忆提取接口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.schema import MemoryCandidateDraft, MemoryTypeSchema


class MemoryExtractorBackend(Protocol):
    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemoryCandidateDraft | MemorySemanticProposal]: ...
