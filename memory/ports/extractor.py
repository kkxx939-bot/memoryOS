"""从不可变 Session 证据提取语义候选的端口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memory.core.formation.schema import MemoryCandidateSchema
from memory.core.model import MemoryEditProposal
from pre.session import SessionArchive


class MemoryExtractionModelProvider(Protocol):
    """记忆提取只需要一个已配置好、接受文本 Prompt 的模型端口。"""

    @property
    def is_remote(self) -> bool: ...

    def complete(self, request: str) -> object: ...


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


__all__ = ["MemoryExtractionModelProvider", "MemoryExtractorBackend"]
