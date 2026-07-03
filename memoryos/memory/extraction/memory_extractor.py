from __future__ import annotations

from dataclasses import dataclass, field

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
