"""记忆文档控制记录共享的状态、限制和校验函数。"""

from __future__ import annotations

from enum import Enum
from typing import Any

_INTENT_SCHEMA = "memory_document_intent_v1"
_EVENT_SCHEMA = "memory_document_change_event_v1"
_CONTROL_SCHEMA = "memory_document_control_v1"
_ADOPTION_RECEIPT_SCHEMA = "memory_document_adoption_receipt_v1"
_ADOPTION_IDENTITY_SCHEMA = "memory_document_adoption_identity_v1"
_ROOT_IDENTITY_SCHEMA = "memory_document_root_identity_v1"
_PUBLICATION_BARRIER_SCHEMA = "memory_document_publication_barrier_v1"
_MAX_CONTROL_BYTES = 1024 * 1024
_MAX_LINEAGE_EVENTS = 10_000
_MAX_PUBLICATION_BARRIERS = 10_000
_MAX_DOCUMENT_CONTROLS = 10_000
_MAX_ADOPTION_RECEIPTS = 10_000


class DocumentControlIntegrityError(RuntimeError):
    """耐久控制记录损坏，或记录身份与存储路径不一致。"""


class DocumentIntentStatus(str, Enum):
    PREPARED = "PREPARED"
    INSTALLED = "INSTALLED"
    EVENT_APPENDED = "EVENT_APPENDED"
    PROJECTION_ENQUEUED = "PROJECTION_ENQUEUED"
    COMPLETED = "COMPLETED"
    CONFLICTED = "CONFLICTED"


class DocumentDeletionStatus(str, Enum):
    """比可重建检索数据保存更久的删除发布状态。"""

    SOFT_FORGOTTEN = "SOFT_FORGOTTEN"
    HARD_ERASED = "HARD_ERASED"


def mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DocumentControlIntegrityError(f"{label} must be a JSON object")
    return value


def is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def validate_prefixed_digest(value: str, prefix: str, label: str) -> None:
    if not value.startswith(prefix) or not is_hex(value.removeprefix(prefix), 64):
        raise ValueError(f"{label} is invalid")
