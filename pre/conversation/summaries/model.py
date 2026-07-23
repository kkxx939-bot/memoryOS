"""Conversation 语义摘要的数据契约；不负责生成摘要。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonicalize
from pre.conversation.messages.model import conversation_datetime


class ConversationSummaryError(ValueError):
    """Conversation summary 与其原始消息批次不一致。"""


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    source_message_digest: str
    content: str
    started_at: datetime
    ended_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        try:
            identifier = require_safe_path_segment(self.conversation_id, "conversation_id")
        except ValueError as exc:
            raise ConversationSummaryError(str(exc)) from exc
        object.__setattr__(self, "conversation_id", identifier)
        if not isinstance(self.source_message_digest, str):
            raise ConversationSummaryError("source_message_digest must be a string")
        digest = self.source_message_digest
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ConversationSummaryError("source_message_digest must be a SHA-256 hex digest")
        object.__setattr__(self, "source_message_digest", digest)
        if not isinstance(self.content, str) or not self.content.strip():
            raise ConversationSummaryError("conversation summary content must be non-empty text")
        started_at = conversation_datetime(self.started_at, "summary started_at")
        ended_at = conversation_datetime(self.ended_at, "summary ended_at")
        updated_at = conversation_datetime(self.updated_at, "summary updated_at")
        if started_at > ended_at or updated_at < ended_at:
            raise ConversationSummaryError("conversation summary time range is invalid")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "updated_at", updated_at)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": "conversation_summary_v1",
                "conversation_id": self.conversation_id,
                "source_message_digest": self.source_message_digest,
                "content": self.content,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "updated_at": self.updated_at,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationSummary:
        allowed = {
            "schema_version",
            "conversation_id",
            "source_message_digest",
            "content",
            "started_at",
            "ended_at",
            "updated_at",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationSummaryError(f"conversation summary contains unknown fields: {sorted(unknown)}")
        if str(payload.get("schema_version") or "") != "conversation_summary_v1":
            raise ConversationSummaryError("unsupported conversation summary schema")
        required = allowed - {"schema_version"}
        missing = required - set(payload)
        if missing:
            raise ConversationSummaryError(f"conversation summary is missing fields: {sorted(missing)}")
        for field_name in ("conversation_id", "source_message_digest", "content"):
            if not isinstance(payload[field_name], str):
                raise ConversationSummaryError(f"conversation summary {field_name} must be text")
        return cls(
            conversation_id=payload["conversation_id"],
            source_message_digest=payload["source_message_digest"],
            content=payload["content"],
            started_at=conversation_datetime(str(payload["started_at"]), "summary started_at"),
            ended_at=conversation_datetime(str(payload["ended_at"]), "summary ended_at"),
            updated_at=conversation_datetime(str(payload["updated_at"]), "summary updated_at"),
        )


__all__ = ["ConversationSummary", "ConversationSummaryError"]
