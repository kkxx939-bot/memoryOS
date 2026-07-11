"""记忆系统里的记忆提取接口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.schema import MemoryCandidateDraft, MemoryTypeSchema
from memoryos.operations.model.context_operation import ContextOperation


@dataclass
class ExtractionResult:
    accepted: list[ContextOperation] = field(default_factory=list)
    pending: list[ContextOperation] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    raw_output: str = ""
    extractor_version: str = "context_memory_extractor_v1"

    def to_dict(self) -> dict:
        return {
            "accepted": [operation.to_dict() for operation in self.accepted],
            "pending": [operation.to_dict() for operation in self.pending],
            "rejected": self.rejected,
            "raw_output": self.raw_output,
            "extractor_version": self.extractor_version,
        }


class MemoryExtractor:
    def extract(self, session_archive) -> ExtractionResult:
        raise NotImplementedError


class MemoryExtractorBackend(Protocol):
    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemoryCandidateDraft | MemorySemanticProposal]: ...
