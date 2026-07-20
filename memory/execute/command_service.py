"""用户记忆命令的统一门面；具体操作由独立模块实现。"""

from __future__ import annotations

from memory.execute.consolidate import ConsolidateOperation
from memory.execute.contracts import (
    AdoptResult,
    DocumentEditResult,
    ForgetMode,
    ForgetResult,
    MemoryConsolidationProposalResult,
    MemoryDocumentCommandResult,
    MemoryHistoryResult,
    MemoryRevisionInfo,
    RememberResult,
)
from memory.execute.edit import EditOperation
from memory.execute.forget import ForgetOperation
from memory.execute.history import HistoryOperation
from memory.execute.remember import RememberOperation


class MemoryCommandService(
    RememberOperation,
    EditOperation,
    ConsolidateOperation,
    ForgetOperation,
    HistoryOperation,
):
    """聚合稳定的公开入口，不再承载任一具体记忆操作的实现。"""


__all__ = [
    "AdoptResult",
    "DocumentEditResult",
    "ForgetMode",
    "ForgetResult",
    "MemoryCommandService",
    "MemoryConsolidationProposalResult",
    "MemoryDocumentCommandResult",
    "MemoryHistoryResult",
    "MemoryRevisionInfo",
    "RememberResult",
]
