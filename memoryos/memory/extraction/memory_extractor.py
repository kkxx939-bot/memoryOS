"""记忆系统里的记忆提取接口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.schema import MemoryTypeSchema


class MemoryExtractorBackend(Protocol):
    semantic_proposal_backend: bool
    llm_semantic_backend: bool

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemorySemanticProposal]: ...
