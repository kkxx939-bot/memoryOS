"""Markdown 记忆的文件布局、控制记录和修订持久化。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MemoryDocumentBootstrapper": (
        "infrastructure.store.memory.bootstrap",
        "MemoryDocumentBootstrapper",
    ),
    "DocumentAdoptionReceipt": (
        "infrastructure.store.memory.control_store",
        "DocumentAdoptionReceipt",
    ),
    "DocumentCommitIntent": (
        "infrastructure.store.memory.control_store",
        "DocumentCommitIntent",
    ),
    "DocumentControlIntegrityError": (
        "infrastructure.store.memory.control_store",
        "DocumentControlIntegrityError",
    ),
    "DocumentControlRecord": (
        "infrastructure.store.memory.control_store",
        "DocumentControlRecord",
    ),
    "DocumentDeletionStatus": (
        "infrastructure.store.memory.control_store",
        "DocumentDeletionStatus",
    ),
    "DocumentIntentStatus": (
        "infrastructure.store.memory.control_store",
        "DocumentIntentStatus",
    ),
    "DocumentPathEffect": (
        "infrastructure.store.memory.control_store",
        "DocumentPathEffect",
    ),
    "DocumentPublicationBarrier": (
        "infrastructure.store.memory.control_store",
        "DocumentPublicationBarrier",
    ),
    "DocumentRootIdentity": (
        "infrastructure.store.memory.control_store",
        "DocumentRootIdentity",
    ),
    "MemoryDocumentControlStore": (
        "infrastructure.store.memory.control_store",
        "MemoryDocumentControlStore",
    ),
    "adoption_document_id": (
        "infrastructure.store.memory.control_store",
        "adoption_document_id",
    ),
    "adoption_request_digest": (
        "infrastructure.store.memory.control_store",
        "adoption_request_digest",
    ),
    "deletion_event_digest": (
        "infrastructure.store.memory.control_store",
        "deletion_event_digest",
    ),
    "RUNTIME_LAYOUT_SCHEMA": ("infrastructure.store.memory.layout", "RUNTIME_LAYOUT_SCHEMA"),
    "RuntimeLayout": ("infrastructure.store.memory.layout", "RuntimeLayout"),
    "RuntimeResetRequired": ("infrastructure.store.memory.layout", "RuntimeResetRequired"),
    "UnsupportedRuntimeLayout": (
        "infrastructure.store.memory.layout",
        "UnsupportedRuntimeLayout",
    ),
    "tenant_control_root": ("infrastructure.store.memory.layout", "tenant_control_root"),
    "user_memory_root": ("infrastructure.store.memory.layout", "user_memory_root"),
    "DocumentRevisionIntegrityError": (
        "infrastructure.store.memory.revision_store",
        "DocumentRevisionIntegrityError",
    ),
    "DocumentRevisionRecord": (
        "infrastructure.store.memory.revision_store",
        "DocumentRevisionRecord",
    ),
    "MemoryDocumentRevisionStore": (
        "infrastructure.store.memory.revision_store",
        "MemoryDocumentRevisionStore",
    ),
    "MemoryDocumentEraseStore": (
        "infrastructure.store.memory.erasure_store",
        "MemoryDocumentEraseStore",
    ),
    "MemoryDocumentConsolidationStore": (
        "infrastructure.store.memory.consolidation_store",
        "MemoryDocumentConsolidationStore",
    ),
    "MemoryEditReviewIntegrityError": (
        "infrastructure.store.memory.review",
        "MemoryEditReviewIntegrityError",
    ),
    "MemoryEditReviewRecord": (
        "infrastructure.store.memory.review",
        "MemoryEditReviewRecord",
    ),
    "MemoryEditReviewStatus": (
        "infrastructure.store.memory.review",
        "MemoryEditReviewStatus",
    ),
    "MemoryEditReviewStore": (
        "infrastructure.store.memory.review",
        "MemoryEditReviewStore",
    ),
    "MemoryEditReviewWorkflow": (
        "infrastructure.store.memory.review",
        "MemoryEditReviewWorkflow",
    ),
    "ReviewConsolidationSource": (
        "infrastructure.store.memory.review",
        "ReviewConsolidationSource",
    ),
    "ExternalChangeKind": ("infrastructure.store.memory.scanner", "ExternalChangeKind"),
    "ExternalDocumentChange": (
        "infrastructure.store.memory.scanner",
        "ExternalDocumentChange",
    ),
    "MemoryDocumentScanner": (
        "infrastructure.store.memory.scanner",
        "MemoryDocumentScanner",
    ),
    "ScanReconciliation": ("infrastructure.store.memory.scanner", "ScanReconciliation"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
