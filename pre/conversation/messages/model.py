"""Conversation messages 的严格数据契约。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonical_json, canonicalize, immutable_snapshot


class ConversationMessageError(ValueError):
    """原始会话消息或批次不满足无损、严格角色约束。"""


class ConversationMessageRole(str, Enum):
    PROMPT = "prompt"
    COMPLETION = "completion"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


def conversation_datetime(value: datetime | str, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConversationMessageError(f"{label} must be ISO-8601") from exc
    else:
        raise ConversationMessageError(f"{label} must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None:
        raise ConversationMessageError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ConversationMessageError(f"{label} must be a string")
    text = value
    if not text or text != text.strip() or any(ord(character) < 32 for character in text):
        raise ConversationMessageError(f"{label} must be a non-empty clean string")
    return text


@dataclass(frozen=True)
class ConversationMessage:
    message_id: str
    role: ConversationMessageRole
    content: Any
    occurred_at: datetime
    sequence: int
    tool_call_id: str | None = None
    tool_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _identifier(self.message_id, "message_id"))
        role = ConversationMessageRole(self.role)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "occurred_at", conversation_datetime(self.occurred_at, "occurred_at"))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ConversationMessageError("message sequence must be a non-negative integer")
        normalized = canonicalize(self.content)
        if role in {ConversationMessageRole.PROMPT, ConversationMessageRole.COMPLETION}:
            if not isinstance(normalized, str) or not normalized:
                raise ConversationMessageError("prompt and completion content must be non-empty text")
            if self.tool_call_id is not None or self.tool_name is not None:
                raise ConversationMessageError("prompt and completion cannot carry tool identity")
        elif role is ConversationMessageRole.TOOL_CALL:
            if not isinstance(normalized, dict | str):
                raise ConversationMessageError("tool_call content must be an object or exact argument string")
            object.__setattr__(self, "tool_call_id", _identifier(self.tool_call_id, "tool_call_id"))
            object.__setattr__(self, "tool_name", _identifier(self.tool_name, "tool_name"))
        else:
            object.__setattr__(self, "tool_call_id", _identifier(self.tool_call_id, "tool_call_id"))
            if self.tool_name is not None:
                object.__setattr__(self, "tool_name", _identifier(self.tool_name, "tool_name"))
        object.__setattr__(self, "content", immutable_snapshot(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": canonicalize(self.content),
            "occurred_at": self.occurred_at.isoformat(),
            "sequence": self.sequence,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationMessage:
        allowed = {
            "message_id",
            "role",
            "content",
            "occurred_at",
            "sequence",
            "tool_call_id",
            "tool_name",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationMessageError(f"conversation message contains unknown fields: {sorted(unknown)}")
        required = {"message_id", "role", "content", "occurred_at", "sequence"}
        missing = required - set(payload)
        if missing:
            raise ConversationMessageError(f"conversation message is missing fields: {sorted(missing)}")
        return cls(
            message_id=payload["message_id"],
            role=ConversationMessageRole(str(payload["role"])),
            content=payload["content"],
            occurred_at=conversation_datetime(str(payload["occurred_at"]), "occurred_at"),
            sequence=payload["sequence"],
            tool_call_id=payload.get("tool_call_id"),
            tool_name=payload.get("tool_name"),
        )


@dataclass(frozen=True)
class ConversationBatch:
    conversation_id: str
    messages: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        try:
            identifier = require_safe_path_segment(self.conversation_id, "conversation_id")
        except ValueError as exc:
            raise ConversationMessageError(str(exc)) from exc
        if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
            raise ConversationMessageError("conversation_id contains unsafe characters")
        object.__setattr__(self, "conversation_id", identifier)
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ConversationMessageError("conversation batch must contain at least one message")
        if any(not isinstance(item, ConversationMessage) for item in self.messages):
            raise ConversationMessageError("conversation batch contains an invalid message")
        if [item.sequence for item in self.messages] != list(range(len(self.messages))):
            raise ConversationMessageError("conversation message sequence must be contiguous and ordered")
        if len({item.message_id for item in self.messages}) != len(self.messages):
            raise ConversationMessageError("conversation message IDs must be unique")
        if any(
            current.occurred_at > following.occurred_at
            for current, following in zip(self.messages, self.messages[1:], strict=False)
        ):
            raise ConversationMessageError("conversation messages must be chronological")
        calls: set[str] = set()
        for item in self.messages:
            if item.role is ConversationMessageRole.TOOL_CALL:
                assert item.tool_call_id is not None
                if item.tool_call_id in calls:
                    raise ConversationMessageError("tool_call IDs must be unique")
                calls.add(item.tool_call_id)
            elif item.role is ConversationMessageRole.TOOL_RESULT:
                if item.tool_call_id not in calls:
                    raise ConversationMessageError("tool_result must reference a preceding tool_call")

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
        return {
            "conversation_id": self.conversation_id,
            "messages": [message.to_dict() for message in self.messages],
        }

    def to_jsonl(self) -> str:
        return "".join(
            canonical_json({"conversation_id": self.conversation_id, **message.to_dict()}) + "\n"
            for message in self.messages
        )

    @classmethod
    def from_jsonl(cls, source: str) -> ConversationBatch:
        conversation_id = ""
        messages: list[ConversationMessage] = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversationMessageError(f"invalid conversation JSONL at line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ConversationMessageError(f"conversation JSONL line {line_number} must be an object")
            row_id = str(raw.pop("conversation_id", ""))
            if not row_id or (conversation_id and row_id != conversation_id):
                raise ConversationMessageError("conversation JSONL contains mixed conversation IDs")
            conversation_id = row_id
            messages.append(ConversationMessage.from_dict(raw))
        return cls(conversation_id=conversation_id, messages=tuple(messages))


__all__ = [
    "ConversationBatch",
    "ConversationMessage",
    "ConversationMessageError",
    "ConversationMessageRole",
    "conversation_datetime",
]
