"""Conversation 归档片段的宽语义历史过程摘要 Schema。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonicalize
from pre.conversation.messages.model import (
    ConversationSegment,
    conversation_datetime,
    require_sha256,
)


class ConversationSummarySchemaError(ValueError):
    """Conversation 片段摘要不满足来源或内容契约。"""


def _safe_identifier(value: object, label: str) -> str:
    try:
        identifier = require_safe_path_segment(value, label)
    except ValueError as exc:
        raise ConversationSummarySchemaError(str(exc)) from exc
    if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
        raise ConversationSummarySchemaError(f"{label} contains unsafe characters")
    return identifier


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationSummarySchemaError(f"{label} must be non-empty text")
    return value


def _text_sequence(value: object, label: str, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise ConversationSummarySchemaError(f"{label} must be a list of text items")
    resolved = tuple(_required_text(item, f"{label} item") for item in value)
    if required and not resolved:
        raise ConversationSummarySchemaError(f"{label} must contain at least one item")
    return resolved


def _summary_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, datetime | str):
        raise ConversationSummarySchemaError(f"{label} must be a datetime or ISO-8601 string")
    try:
        return conversation_datetime(value, label)
    except ValueError as exc:
        raise ConversationSummarySchemaError(str(exc)) from exc


@dataclass(frozen=True)
class ConversationSegmentSummary:
    """描述一个不可变归档片段中完整历史过程的派生语义。"""

    conversation_id: str
    segment_id: str
    source_message_digest: str
    start_sequence: int
    end_sequence: int
    started_at: datetime
    ended_at: datetime
    generated_at: datetime
    overview: str
    chronology: tuple[str, ...]
    corrections: tuple[str, ...]
    ending_state: str
    open_threads: tuple[str, ...]

    SCHEMA_VERSION = "conversation_segment_summary_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            _safe_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(self, "segment_id", _safe_identifier(self.segment_id, "segment_id"))
        try:
            source_digest = require_sha256(self.source_message_digest, "source_message_digest")
        except ValueError as exc:
            raise ConversationSummarySchemaError(str(exc)) from exc
        object.__setattr__(self, "source_message_digest", source_digest)

        for label, value in (
            ("start_sequence", self.start_sequence),
            ("end_sequence", self.end_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationSummarySchemaError(f"{label} must be a non-negative integer")
        if self.start_sequence > self.end_sequence:
            raise ConversationSummarySchemaError("summary sequence range is invalid")

        started_at = _summary_datetime(self.started_at, "summary started_at")
        ended_at = _summary_datetime(self.ended_at, "summary ended_at")
        generated_at = _summary_datetime(self.generated_at, "summary generated_at")
        if started_at > ended_at or generated_at < ended_at:
            raise ConversationSummarySchemaError("summary time range is invalid")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "generated_at", generated_at)

        object.__setattr__(self, "overview", _required_text(self.overview, "overview"))
        object.__setattr__(
            self,
            "chronology",
            _text_sequence(self.chronology, "chronology", required=True),
        )
        object.__setattr__(
            self,
            "corrections",
            _text_sequence(self.corrections, "corrections", required=False),
        )
        object.__setattr__(self, "ending_state", _required_text(self.ending_state, "ending_state"))
        object.__setattr__(
            self,
            "open_threads",
            _text_sequence(self.open_threads, "open_threads", required=False),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def require_matches_source(self, segment: ConversationSegment) -> None:
        """确认摘要只绑定其声明的不可变消息片段。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        expected = (
            segment.conversation_id,
            segment.segment_id,
            segment.digest,
            segment.start_sequence,
            segment.end_sequence,
            segment.started_at,
            segment.ended_at,
        )
        actual = (
            self.conversation_id,
            self.segment_id,
            self.source_message_digest,
            self.start_sequence,
            self.end_sequence,
            self.started_at,
            self.ended_at,
        )
        if actual != expected:
            raise ConversationSummarySchemaError("conversation summary does not match its source segment")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": self.SCHEMA_VERSION,
                "conversation_id": self.conversation_id,
                "segment_id": self.segment_id,
                "source_message_digest": self.source_message_digest,
                "start_sequence": self.start_sequence,
                "end_sequence": self.end_sequence,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "generated_at": self.generated_at,
                "overview": self.overview,
                "chronology": self.chronology,
                "corrections": self.corrections,
                "ending_state": self.ending_state,
                "open_threads": self.open_threads,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationSegmentSummary:
        allowed = {
            "schema_version",
            "conversation_id",
            "segment_id",
            "source_message_digest",
            "start_sequence",
            "end_sequence",
            "started_at",
            "ended_at",
            "generated_at",
            "overview",
            "chronology",
            "corrections",
            "ending_state",
            "open_threads",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationSummarySchemaError(
                f"conversation summary contains unknown fields: {sorted(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ConversationSummarySchemaError("unsupported conversation summary schema")
        required = allowed - {"schema_version"}
        missing = required - set(payload)
        if missing:
            raise ConversationSummarySchemaError(f"conversation summary is missing fields: {sorted(missing)}")
        return cls(
            conversation_id=payload["conversation_id"],
            segment_id=payload["segment_id"],
            source_message_digest=payload["source_message_digest"],
            start_sequence=payload["start_sequence"],
            end_sequence=payload["end_sequence"],
            started_at=_summary_datetime(payload["started_at"], "summary started_at"),
            ended_at=_summary_datetime(payload["ended_at"], "summary ended_at"),
            generated_at=_summary_datetime(payload["generated_at"], "summary generated_at"),
            overview=payload["overview"],
            chronology=payload["chronology"],
            corrections=payload["corrections"],
            ending_state=payload["ending_state"],
            open_threads=payload["open_threads"],
        )


__all__ = ["ConversationSegmentSummary", "ConversationSummarySchemaError"]
