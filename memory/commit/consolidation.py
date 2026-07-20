"""多文档记忆合并事务的公开入口。"""

from memory.commit.consolidation_service import MemoryDocumentConsolidator
from memory.ports.consolidation import (
    ConsolidationFaultHook,
    ConsolidationInputRequired,
    ConsolidationIntegrityError,
    ConsolidationProjectionReader,
    ConsolidationRecoveryReport,
    ConsolidationResult,
    ConsolidationSagaRecord,
    ConsolidationSagaStore,
    ConsolidationSource,
    ConsolidationStatus,
    consolidation_identity_digest,
    consolidation_saga_id,
)

__all__ = [
    "ConsolidationFaultHook",
    "ConsolidationInputRequired",
    "ConsolidationIntegrityError",
    "ConsolidationProjectionReader",
    "ConsolidationRecoveryReport",
    "ConsolidationResult",
    "ConsolidationSagaRecord",
    "ConsolidationSagaStore",
    "ConsolidationSource",
    "ConsolidationStatus",
    "MemoryDocumentConsolidator",
    "consolidation_identity_digest",
    "consolidation_saga_id",
]
