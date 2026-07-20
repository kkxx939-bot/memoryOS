"""Markdown 记忆文档控制仓储的组合入口。"""

from __future__ import annotations

from pathlib import Path

from infrastructure.store.memory.control_common import (
    DocumentControlIntegrityError,
    DocumentDeletionStatus,
    DocumentIntentStatus,
)
from infrastructure.store.memory.control_identity import (
    DocumentAdoptionReceipt,
    DocumentRootIdentity,
    DocumentRootIdentityGuard,
    adoption_document_id,
    adoption_request_digest,
)
from infrastructure.store.memory.control_identity_store import ControlIdentityStoreMixin
from infrastructure.store.memory.control_intent import (
    DocumentCommitIntent,
    DocumentPathEffect,
    deletion_event_digest,
    document_intent_id,
    document_intent_identity_digest,
)
from infrastructure.store.memory.control_publication_store import ControlPublicationStoreMixin
from infrastructure.store.memory.control_record import (
    DocumentControlRecord,
    DocumentPublicationBarrier,
)


class MemoryDocumentControlStore(
    ControlIdentityStoreMixin,
    ControlPublicationStoreMixin,
):
    """组合文档身份、提交日志、发布屏障和安全文件操作。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)


__all__ = [
    "DocumentAdoptionReceipt", "DocumentCommitIntent", "DocumentControlIntegrityError",
    "DocumentControlRecord", "DocumentDeletionStatus", "DocumentIntentStatus",
    "DocumentPathEffect", "DocumentPublicationBarrier", "DocumentRootIdentity",
    "DocumentRootIdentityGuard",
    "MemoryDocumentControlStore", "adoption_document_id", "adoption_request_digest",
    "deletion_event_digest", "document_intent_id", "document_intent_identity_digest",
]
