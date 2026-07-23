"""单文档提交事务的共享结果、冲突和准备态模型。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from infrastructure.store.memory.control_store import (
    DocumentCommitIntent,
    DocumentControlRecord,
    DocumentIntentStatus,
    DocumentPathEffect,
)
from infrastructure.store.memory.revision_store import (
    DocumentRevisionRecord,
)
from memory.core.model import (
    DocumentChangeEvent,
    DocumentEditKind,
)
from memory.ports.document_store import (
    DocumentConflictError,
)


class DocumentCommitConflict(DocumentConflictError):
    """当前 Markdown 或耐久身份处于第三状态，原状态已被保留。"""

    retryable = False


@dataclass(frozen=True)
class DocumentCommitResult:
    intent_id: str
    status: DocumentIntentStatus
    event: DocumentChangeEvent | None
    control: DocumentControlRecord | None
    revision: DocumentRevisionRecord | None
    no_op: bool = False
    recovered: bool = False


@dataclass(frozen=True)
class DocumentRecoveryReport:
    completed: tuple[DocumentCommitResult, ...]
    conflicted_intent_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedPlan:
    tenant_id: str
    owner_user_id: str
    document_id: str
    edit_kind: DocumentEditKind
    effects: tuple[DocumentPathEffect, ...]
    after_bytes: bytes | None
    revision_bytes: bytes
    after_blob_digest: str
    revision_blob_digest: str
    revision_blob_role: str
    old_relative_path: str
    new_relative_path: str


FaultHook = Callable[[str, DocumentCommitIntent], None]


class _DocumentCommitState:
    """拆分后的文档事务阶段共享同一个显式状态契约。"""

    document_store: Any
    control_store: Any
    revision_store: Any
    projection_queue: Any
    erasure_store: Any
    path_lock: Any
    test_hook: Any
    clock: Any

    def _commit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _resume_existing(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _preflight_new_intent_root(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _verify_existing_root_identity(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _notify(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _bounded_text(value: object, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(character) < 32 and character not in "\t" for character in text):
        raise ValueError(f"{label} is empty, too large or contains control characters")
    return text


def _sha256_digest(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


__all__ = [
    "DocumentCommitConflict",
    "DocumentCommitResult",
    "DocumentRecoveryReport",
    "FaultHook",
]
