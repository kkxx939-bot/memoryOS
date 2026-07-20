"""记忆执行层对外返回的稳定结果对象。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ForgetMode = Literal["SOFT_FORGET", "HARD_ERASE"]


@dataclass(frozen=True)
class MemoryDocumentCommandResult:
    """一次记忆文档命令的公共结果。"""

    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    document_revision: int
    source_digest: str
    changed: bool
    edit_summary: str
    projection_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RememberResult(MemoryDocumentCommandResult):
    """明确记住一条内容的结果。"""


@dataclass(frozen=True)
class AdoptResult(MemoryDocumentCommandResult):
    """收养一个未受管 Markdown 文档的结果。"""


@dataclass(frozen=True)
class DocumentEditResult(MemoryDocumentCommandResult):
    """编辑、重命名或恢复文档的结果。"""


@dataclass(frozen=True)
class ForgetResult(MemoryDocumentCommandResult):
    """软遗忘或硬擦除的结果。"""

    mode: ForgetMode
    recoverable: bool
    erasure_status: str = ""
    erasure_epoch: str = ""
    pending_backends: tuple[str, ...] = ()
    independent_evidence_retained: tuple[str, ...] = ()
    media_disclaimer: str = ""


@dataclass(frozen=True)
class MemoryRevisionInfo:
    """一个可查询的记忆文档历史版本。"""

    document_revision: int
    projection_generation: int
    edit_kind: str
    relative_path: str
    source_digest: str
    state: str
    created_at: str
    restorable: bool


@dataclass(frozen=True)
class MemoryHistoryResult:
    """记忆文档历史查询结果。"""

    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    revisions: tuple[MemoryRevisionInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["revisions"] = [asdict(revision) for revision in self.revisions]
        return payload


@dataclass(frozen=True)
class MemoryConsolidationProposalResult:
    """多个记忆文档合并前生成的待审核提案。"""

    proposal_id: str
    status: str
    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    source_digest: str
    proposed_source_digest: str
    proposed_diff_digest: str
    proposed_diff: str
    edit_summary: str
    workflow_kind: str
    consolidation_sources: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "AdoptResult",
    "DocumentEditResult",
    "ForgetMode",
    "ForgetResult",
    "MemoryConsolidationProposalResult",
    "MemoryDocumentCommandResult",
    "MemoryHistoryResult",
    "MemoryRevisionInfo",
    "RememberResult",
]
