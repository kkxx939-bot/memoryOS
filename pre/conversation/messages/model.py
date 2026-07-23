"""Conversation 原始消息、追加批次与不可变归档片段的严格 Schema。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonicalize, immutable_snapshot


class ConversationMessageSchemaError(ValueError):
    """Conversation 消息、批次或归档片段不满足数据契约。"""


class ConversationMessageRole(str, Enum):
    """持久化时严格区分的会话角色。"""

    PROMPT = "prompt"
    COMPLETION = "completion"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ConversationToolResultStatus(str, Enum):
    """工具结果的最终执行状态。"""

    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class ConversationToolResultContentMode(str, Enum):
    """工具结果在 Conversation 中保留到什么程度。"""

    INLINE = "inline"
    SUMMARIZED = "summarized"
    OMITTED = "omitted"


def conversation_datetime(value: datetime | str, label: str) -> datetime:
    """解析带时区时间并规范为 UTC。"""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConversationMessageSchemaError(f"{label} must be ISO-8601") from exc
    else:
        raise ConversationMessageSchemaError(f"{label} must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None:
        raise ConversationMessageSchemaError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def require_sha256(value: object, label: str) -> str:
    """校验小写 SHA-256 十六进制摘要。"""

    if not isinstance(value, str) or len(value) != 64:
        raise ConversationMessageSchemaError(f"{label} must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ConversationMessageSchemaError(f"{label} must be a SHA-256 hex digest")
    return value


def _clean_identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ConversationMessageSchemaError(f"{label} must be a string")
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ConversationMessageSchemaError(f"{label} must be a non-empty clean string")
    return value


def _optional_clean_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _clean_identifier(value, label)


def _safe_path_identifier(value: object, label: str) -> str:
    try:
        identifier = require_safe_path_segment(value, label)
    except ValueError as exc:
        raise ConversationMessageSchemaError(str(exc)) from exc
    if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
        raise ConversationMessageSchemaError(f"{label} contains unsafe characters")
    return identifier


@dataclass(frozen=True)
class ConversationMessage:
    """一条可独立校验的原始会话事实。"""

    message_id: str
    sequence: int
    role: ConversationMessageRole
    occurred_at: datetime
    content: Any
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_status: ConversationToolResultStatus | None = None
    content_mode: ConversationToolResultContentMode | None = None
    source_ref: str | None = None
    original_size_bytes: int | None = None
    original_sha256: str | None = None

    SCHEMA_VERSION = "conversation_message_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _clean_identifier(self.message_id, "message_id"))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ConversationMessageSchemaError("message sequence must be a non-negative integer")
        try:
            role = ConversationMessageRole(self.role)
        except ValueError as exc:
            raise ConversationMessageSchemaError("unsupported conversation message role") from exc
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "occurred_at", conversation_datetime(self.occurred_at, "occurred_at"))
        try:
            normalized_content = canonicalize(self.content)
        except ValueError as exc:
            raise ConversationMessageSchemaError(f"message content is not canonical JSON: {exc}") from exc

        tool_call_id = _optional_clean_text(self.tool_call_id, "tool_call_id")
        tool_name = _optional_clean_text(self.tool_name, "tool_name")
        source_ref = _optional_clean_text(self.source_ref, "source_ref")
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "source_ref", source_ref)

        if self.original_size_bytes is not None:
            if (
                isinstance(self.original_size_bytes, bool)
                or not isinstance(self.original_size_bytes, int)
                or self.original_size_bytes < 0
            ):
                raise ConversationMessageSchemaError("original_size_bytes must be a non-negative integer")
        if self.original_sha256 is not None:
            object.__setattr__(
                self,
                "original_sha256",
                require_sha256(self.original_sha256, "original_sha256"),
            )

        if role in {ConversationMessageRole.PROMPT, ConversationMessageRole.COMPLETION}:
            self._validate_text_message(normalized_content)
        elif role is ConversationMessageRole.TOOL_CALL:
            self._validate_tool_call(normalized_content)
        else:
            self._validate_tool_result(normalized_content)
        object.__setattr__(self, "content", immutable_snapshot(normalized_content))

    def _validate_text_message(self, content: Any) -> None:
        if not isinstance(content, str) or not content:
            raise ConversationMessageSchemaError("prompt and completion content must be non-empty text")
        if any(
            value is not None
            for value in (
                self.tool_call_id,
                self.tool_name,
                self.tool_status,
                self.content_mode,
                self.source_ref,
                self.original_size_bytes,
                self.original_sha256,
            )
        ):
            raise ConversationMessageSchemaError("prompt and completion cannot carry tool-result fields")

    def _validate_tool_call(self, content: Any) -> None:
        if not isinstance(content, dict | str):
            raise ConversationMessageSchemaError("tool_call content must be an object or exact argument string")
        if isinstance(content, str) and not content:
            raise ConversationMessageSchemaError("tool_call argument string must be non-empty")
        if self.tool_call_id is None or self.tool_name is None:
            raise ConversationMessageSchemaError("tool_call requires tool_call_id and tool_name")
        if any(
            value is not None
            for value in (
                self.tool_status,
                self.content_mode,
                self.source_ref,
                self.original_size_bytes,
                self.original_sha256,
            )
        ):
            raise ConversationMessageSchemaError("tool_call cannot carry tool-result fields")

    def _validate_tool_result(self, content: Any) -> None:
        if self.tool_call_id is None or self.tool_name is None:
            raise ConversationMessageSchemaError("tool_result requires tool_call_id and tool_name")
        if self.tool_status is None:
            raise ConversationMessageSchemaError("tool_result requires tool_status")
        if self.content_mode is None:
            raise ConversationMessageSchemaError("tool_result requires content_mode")
        try:
            status = ConversationToolResultStatus(self.tool_status)
            content_mode = ConversationToolResultContentMode(self.content_mode)
        except ValueError as exc:
            raise ConversationMessageSchemaError("tool_result status or content mode is unsupported") from exc
        object.__setattr__(self, "tool_status", status)
        object.__setattr__(self, "content_mode", content_mode)
        if content_mode is ConversationToolResultContentMode.SUMMARIZED:
            if not isinstance(content, str) or not content.strip():
                raise ConversationMessageSchemaError("summarized tool_result content must be non-empty text")
        if content_mode is ConversationToolResultContentMode.OMITTED:
            has_description = isinstance(content, str) and bool(content.strip())
            if not any(
                value is not None
                for value in (
                    self.source_ref,
                    self.original_size_bytes,
                    self.original_sha256,
                )
            ) and not has_description:
                raise ConversationMessageSchemaError(
                    "omitted tool_result requires a description or original-result metadata"
                )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "message_id": self.message_id,
            "sequence": self.sequence,
            "role": self.role.value,
            "occurred_at": self.occurred_at,
            "content": canonicalize(self.content),
        }
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.tool_status is not None:
            payload["tool_status"] = self.tool_status.value
        if self.content_mode is not None:
            payload["content_mode"] = self.content_mode.value
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        if self.original_size_bytes is not None:
            payload["original_size_bytes"] = self.original_size_bytes
        if self.original_sha256 is not None:
            payload["original_sha256"] = self.original_sha256
        return canonicalize(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationMessage:
        allowed = {
            "schema_version",
            "message_id",
            "sequence",
            "role",
            "occurred_at",
            "content",
            "tool_call_id",
            "tool_name",
            "tool_status",
            "content_mode",
            "source_ref",
            "original_size_bytes",
            "original_sha256",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationMessageSchemaError(f"conversation message contains unknown fields: {sorted(unknown)}")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ConversationMessageSchemaError("unsupported conversation message schema")
        required = {"message_id", "sequence", "role", "occurred_at", "content"}
        missing = required - set(payload)
        if missing:
            raise ConversationMessageSchemaError(f"conversation message is missing fields: {sorted(missing)}")
        try:
            role = ConversationMessageRole(payload["role"])
        except (TypeError, ValueError) as exc:
            raise ConversationMessageSchemaError("unsupported conversation message role") from exc
        tool_status_value = payload.get("tool_status")
        content_mode_value = payload.get("content_mode")
        try:
            tool_status = (
                ConversationToolResultStatus(tool_status_value)
                if tool_status_value is not None
                else None
            )
            content_mode = (
                ConversationToolResultContentMode(content_mode_value)
                if content_mode_value is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ConversationMessageSchemaError("tool_result status or content mode is unsupported") from exc
        return cls(
            message_id=payload["message_id"],
            sequence=payload["sequence"],
            role=role,
            occurred_at=conversation_datetime(payload["occurred_at"], "occurred_at"),
            content=payload["content"],
            tool_call_id=payload.get("tool_call_id"),
            tool_name=payload.get("tool_name"),
            tool_status=tool_status,
            content_mode=content_mode,
            source_ref=payload.get("source_ref"),
            original_size_bytes=payload.get("original_size_bytes"),
            original_sha256=payload.get("original_sha256"),
        )


def _validated_messages(messages: object) -> tuple[ConversationMessage, ...]:
    if not isinstance(messages, tuple | list):
        raise ConversationMessageSchemaError("messages must be a list or tuple")
    resolved = tuple(messages)
    if not resolved:
        raise ConversationMessageSchemaError("conversation messages must not be empty")
    if any(not isinstance(item, ConversationMessage) for item in resolved):
        raise ConversationMessageSchemaError("conversation contains an invalid message")
    expected_sequences = list(range(resolved[0].sequence, resolved[0].sequence + len(resolved)))
    if [item.sequence for item in resolved] != expected_sequences:
        raise ConversationMessageSchemaError("conversation message sequence must be globally contiguous and ordered")
    if len({item.message_id for item in resolved}) != len(resolved):
        raise ConversationMessageSchemaError("conversation message IDs must be unique")
    if any(
        current.occurred_at > following.occurred_at
        for current, following in zip(resolved, resolved[1:], strict=False)
    ):
        raise ConversationMessageSchemaError("conversation messages must be chronological")
    tool_call_ids = [
        item.tool_call_id
        for item in resolved
        if item.role is ConversationMessageRole.TOOL_CALL
    ]
    if len(set(tool_call_ids)) != len(tool_call_ids):
        raise ConversationMessageSchemaError("tool_call IDs must be unique within a message collection")
    return resolved


@dataclass(frozen=True)
class ConversationBatch:
    """一次追加操作携带的连续消息；首序号不必从零开始。"""

    conversation_id: str
    messages: tuple[ConversationMessage, ...]

    SCHEMA_VERSION = "conversation_batch_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            _safe_path_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(self, "messages", _validated_messages(self.messages))

    @property
    def start_sequence(self) -> int:
        return self.messages[0].sequence

    @property
    def end_sequence(self) -> int:
        return self.messages[-1].sequence

    @property
    def started_at(self) -> datetime:
        return self.messages[0].occurred_at

    @property
    def ended_at(self) -> datetime:
        return self.messages[-1].occurred_at

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": self.SCHEMA_VERSION,
                "conversation_id": self.conversation_id,
                "messages": [message.to_dict() for message in self.messages],
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationBatch:
        allowed = {"schema_version", "conversation_id", "messages"}
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationMessageSchemaError(f"conversation batch contains unknown fields: {sorted(unknown)}")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ConversationMessageSchemaError("unsupported conversation batch schema")
        missing = {"conversation_id", "messages"} - set(payload)
        if missing:
            raise ConversationMessageSchemaError(f"conversation batch is missing fields: {sorted(missing)}")
        raw_messages = payload["messages"]
        if not isinstance(raw_messages, list | tuple):
            raise ConversationMessageSchemaError("conversation batch messages must be a list")
        if any(not isinstance(item, Mapping) for item in raw_messages):
            raise ConversationMessageSchemaError("conversation batch contains a non-object message")
        return cls(
            conversation_id=payload["conversation_id"],
            messages=tuple(ConversationMessage.from_dict(item) for item in raw_messages),
        )


@dataclass(frozen=True)
class ConversationSegment:
    """从 live 会话封存得到的不可变、可摘要消息片段。"""

    conversation_id: str
    segment_id: str
    messages: tuple[ConversationMessage, ...]

    SCHEMA_VERSION = "conversation_segment_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            _safe_path_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(self, "segment_id", _safe_path_identifier(self.segment_id, "segment_id"))
        object.__setattr__(self, "messages", _validated_messages(self.messages))

    @property
    def start_sequence(self) -> int:
        return self.messages[0].sequence

    @property
    def end_sequence(self) -> int:
        return self.messages[-1].sequence

    @property
    def started_at(self) -> datetime:
        return self.messages[0].occurred_at

    @property
    def ended_at(self) -> datetime:
        return self.messages[-1].occurred_at

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": self.SCHEMA_VERSION,
                "conversation_id": self.conversation_id,
                "segment_id": self.segment_id,
                "messages": [message.to_dict() for message in self.messages],
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationSegment:
        allowed = {"schema_version", "conversation_id", "segment_id", "messages"}
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationMessageSchemaError(
                f"conversation segment contains unknown fields: {sorted(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ConversationMessageSchemaError("unsupported conversation segment schema")
        missing = {"conversation_id", "segment_id", "messages"} - set(payload)
        if missing:
            raise ConversationMessageSchemaError(f"conversation segment is missing fields: {sorted(missing)}")
        raw_messages = payload["messages"]
        if not isinstance(raw_messages, list | tuple):
            raise ConversationMessageSchemaError("conversation segment messages must be a list")
        if any(not isinstance(item, Mapping) for item in raw_messages):
            raise ConversationMessageSchemaError("conversation segment contains a non-object message")
        return cls(
            conversation_id=payload["conversation_id"],
            segment_id=payload["segment_id"],
            messages=tuple(ConversationMessage.from_dict(item) for item in raw_messages),
        )


__all__ = [
    "ConversationBatch",
    "ConversationMessage",
    "ConversationMessageRole",
    "ConversationMessageSchemaError",
    "ConversationSegment",
    "ConversationToolResultContentMode",
    "ConversationToolResultStatus",
    "conversation_datetime",
    "require_sha256",
]
