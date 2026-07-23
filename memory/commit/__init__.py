"""显式 Memory 提交与 Session 普通派生协调，公开符号按需加载。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.commit.consolidation import (
        ConsolidationInputRequired,
        ConsolidationIntegrityError,
        ConsolidationProjectionReader,
        ConsolidationRecoveryReport,
        ConsolidationResult,
        ConsolidationSagaRecord,
        ConsolidationSource,
        ConsolidationStatus,
        MemoryDocumentConsolidator,
    )
    from memory.commit.document_commit import (
        DocumentCommitConflict,
        DocumentCommitResult,
        DocumentRecoveryReport,
        MemoryDocumentCommitter,
    )
    from memory.commit.erase import (
        DerivedEraseRequest,
        DocumentEraseCleanupBackend,
        DocumentEraseConflict,
        DocumentErasedError,
        DocumentEraseFloorProvider,
        DocumentEraseIntegrityError,
        DocumentEraseRecord,
        DocumentEraseRecoveryReport,
        DocumentEraseResult,
        DocumentEraseStatus,
        EraseBackendProgress,
        MemoryDocumentEraser,
    )
    from memory.commit.remember_plan import ExplicitRememberPlanner, RememberTarget, RememberTargetKind
    from memory.commit.session_commit import DerivedConsumerError, SessionCommitService

__all__ = [
    "ConsolidationInputRequired",
    "ConsolidationIntegrityError",
    "ConsolidationProjectionReader",
    "ConsolidationRecoveryReport",
    "ConsolidationResult",
    "ConsolidationSagaRecord",
    "ConsolidationSource",
    "ConsolidationStatus",
    "DerivedEraseRequest",
    "DerivedConsumerError",
    "DocumentCommitConflict",
    "DocumentCommitResult",
    "DocumentEraseCleanupBackend",
    "DocumentEraseConflict",
    "DocumentErasedError",
    "DocumentEraseFloorProvider",
    "DocumentEraseIntegrityError",
    "DocumentEraseRecord",
    "DocumentEraseRecoveryReport",
    "DocumentEraseResult",
    "DocumentEraseStatus",
    "DocumentRecoveryReport",
    "EraseBackendProgress",
    "ExplicitRememberPlanner",
    "MemoryDocumentCommitter",
    "MemoryDocumentConsolidator",
    "MemoryDocumentEraser",
    "RememberTarget",
    "RememberTargetKind",
    "SessionCommitService",
]

_EXPORTS = {
    "DocumentCommitConflict": ("memory.commit.document_commit", "DocumentCommitConflict"),
    "DocumentCommitResult": ("memory.commit.document_commit", "DocumentCommitResult"),
    "DocumentRecoveryReport": ("memory.commit.document_commit", "DocumentRecoveryReport"),
    "MemoryDocumentCommitter": ("memory.commit.document_commit", "MemoryDocumentCommitter"),
    "ConsolidationInputRequired": ("memory.commit.consolidation", "ConsolidationInputRequired"),
    "ConsolidationIntegrityError": ("memory.commit.consolidation", "ConsolidationIntegrityError"),
    "ConsolidationProjectionReader": ("memory.commit.consolidation", "ConsolidationProjectionReader"),
    "ConsolidationRecoveryReport": ("memory.commit.consolidation", "ConsolidationRecoveryReport"),
    "ConsolidationResult": ("memory.commit.consolidation", "ConsolidationResult"),
    "ConsolidationSagaRecord": ("memory.commit.consolidation", "ConsolidationSagaRecord"),
    "ConsolidationSource": ("memory.commit.consolidation", "ConsolidationSource"),
    "ConsolidationStatus": ("memory.commit.consolidation", "ConsolidationStatus"),
    "MemoryDocumentConsolidator": (
        "memory.commit.consolidation",
        "MemoryDocumentConsolidator",
    ),
    "DerivedEraseRequest": ("memory.commit.erase", "DerivedEraseRequest"),
    "DocumentEraseCleanupBackend": ("memory.commit.erase", "DocumentEraseCleanupBackend"),
    "DocumentEraseConflict": ("memory.commit.erase", "DocumentEraseConflict"),
    "DocumentErasedError": ("memory.commit.erase", "DocumentErasedError"),
    "DocumentEraseFloorProvider": ("memory.commit.erase", "DocumentEraseFloorProvider"),
    "DocumentEraseIntegrityError": ("memory.commit.erase", "DocumentEraseIntegrityError"),
    "DocumentEraseRecord": ("memory.commit.erase", "DocumentEraseRecord"),
    "DocumentEraseRecoveryReport": ("memory.commit.erase", "DocumentEraseRecoveryReport"),
    "DocumentEraseResult": ("memory.commit.erase", "DocumentEraseResult"),
    "DocumentEraseStatus": ("memory.commit.erase", "DocumentEraseStatus"),
    "EraseBackendProgress": ("memory.commit.erase", "EraseBackendProgress"),
    "MemoryDocumentEraser": ("memory.commit.erase", "MemoryDocumentEraser"),
    "ExplicitRememberPlanner": ("memory.commit.remember_plan", "ExplicitRememberPlanner"),
    "RememberTarget": ("memory.commit.remember_plan", "RememberTarget"),
    "RememberTargetKind": ("memory.commit.remember_plan", "RememberTargetKind"),
    "DerivedConsumerError": ("memory.commit.session_commit", "DerivedConsumerError"),
    "SessionCommitService": ("memory.commit.session_commit", "SessionCommitService"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
