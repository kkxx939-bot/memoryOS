"""把 SessionArchive 确定性转换为严格角色化 ConversationBatch。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from foundation.ids import require_safe_path_segment, stable_hash
from pre.conversation.messages.model import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageError,
    ConversationMessageRole,
    conversation_datetime,
)
from pre.session import SessionArchive


class ConversationProjectionError(ValueError):
    """SessionArchive 不能无失真转换为严格 Conversation。"""


@dataclass(frozen=True)
class _ProjectedMessage:
    message: ConversationMessage
    source_order: int
    intra_order: int


class SessionArchiveConversationProjector:
    """只转换 messages/tool_results，不读取摘要或调用模型。"""

    def project(self, archive: SessionArchive) -> ConversationBatch:
        conversation_id = self._conversation_id(archive)
        fallback_time = self._time(archive.created_at, datetime(1970, 1, 1, tzinfo=timezone.utc))
        projected: list[_ProjectedMessage] = []
        known_calls: set[str] = set()
        source_order = 0

        for raw in archive.messages:
            if not isinstance(raw, Mapping):
                raise ConversationProjectionError("SessionArchive messages must contain objects")
            row = dict(raw)
            role = str(row.get("role") or "").strip().casefold()
            occurred_at = self._time(
                row.get("occurred_at") or row.get("created_at") or row.get("timestamp"),
                fallback_time,
            )
            source_id = self._source_id(row, "message", source_order)
            if role == "system":
                source_order += 1
                continue
            if role in {"user", "prompt"}:
                projected.append(
                    self._message(
                        source_id,
                        ConversationMessageRole.PROMPT,
                        self._text(row, ("content", "text", "prompt")),
                        occurred_at,
                        source_order,
                    )
                )
            elif role in {"assistant", "completion"}:
                content = self._optional_text(row, ("content", "text", "completion"))
                intra_order = 0
                if content:
                    projected.append(
                        self._message(
                            source_id,
                            ConversationMessageRole.COMPLETION,
                            content,
                            occurred_at,
                            source_order,
                            intra_order=intra_order,
                        )
                    )
                    intra_order += 1
                tool_calls = row.get("tool_calls") or []
                if not isinstance(tool_calls, list):
                    raise ConversationProjectionError("assistant tool_calls must be a list")
                for index, call in enumerate(tool_calls):
                    item = self._tool_call(
                        call,
                        occurred_at=occurred_at,
                        source_id=f"{source_id}:tool_call:{index}",
                        source_order=source_order,
                        intra_order=intra_order + index,
                    )
                    assert item.message.tool_call_id is not None
                    if item.message.tool_call_id in known_calls:
                        raise ConversationProjectionError("duplicate tool_call_id in SessionArchive messages")
                    known_calls.add(item.message.tool_call_id)
                    projected.append(item)
                if not content and not tool_calls:
                    raise ConversationProjectionError("assistant message has neither completion nor tool_call")
            elif role in {"tool_call", "tool"}:
                added = self._tool_row(
                    row,
                    occurred_at=occurred_at,
                    source_id=source_id,
                    source_order=source_order,
                    known_calls=known_calls,
                    tool_role_is_result=role == "tool",
                )
                projected.extend(added)
            else:
                raise ConversationProjectionError(f"unsupported SessionArchive message role: {role or '<missing>'}")
            source_order += 1

        for raw in archive.tool_results:
            if not isinstance(raw, Mapping):
                raise ConversationProjectionError("SessionArchive tool_results must contain objects")
            row = dict(raw)
            occurred_at = self._time(
                row.get("occurred_at") or row.get("created_at") or row.get("timestamp"),
                fallback_time,
            )
            source_id = self._source_id(row, "tool", source_order)
            projected.extend(
                self._tool_row(
                    row,
                    occurred_at=occurred_at,
                    source_id=source_id,
                    source_order=source_order,
                    known_calls=known_calls,
                    tool_role_is_result=True,
                )
            )
            source_order += 1

        ordered = sorted(
            projected,
            key=lambda item: (
                item.message.occurred_at,
                item.source_order,
                item.intra_order,
                item.message.message_id,
            ),
        )
        try:
            messages = tuple(replace(item.message, sequence=index) for index, item in enumerate(ordered))
            return ConversationBatch(conversation_id=conversation_id, messages=messages)
        except ConversationMessageError as exc:
            raise ConversationProjectionError(str(exc)) from exc

    def _tool_row(
        self,
        row: Mapping[str, Any],
        *,
        occurred_at: datetime,
        source_id: str,
        source_order: int,
        known_calls: set[str],
        tool_role_is_result: bool,
    ) -> list[_ProjectedMessage]:
        call_id = str(row.get("tool_call_id") or row.get("call_id") or row.get("event_id") or row.get("id") or "")
        if not call_id:
            call_id = f"call-{stable_hash({'source_id': source_id, 'tool_name': row.get('tool_name') or row.get('name')})}"
        tool_name = str(row.get("tool_name") or row.get("name") or "").strip()
        result: list[_ProjectedMessage] = []
        if call_id not in known_calls:
            if not tool_name:
                raise ConversationProjectionError("tool result without a preceding call must include tool_name")
            arguments = self._first_present(row, ("input", "tool_input", "arguments", "params"), default={})
            result.append(
                self._message(
                    f"{source_id}:call",
                    ConversationMessageRole.TOOL_CALL,
                    arguments,
                    occurred_at,
                    source_order,
                    intra_order=0,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                )
            )
            known_calls.add(call_id)
        if tool_role_is_result or any(
            key in row for key in ("output", "tool_output", "result", "tool_result", "content")
        ):
            output = self._first_present(
                row,
                ("output", "tool_output", "result", "tool_result", "content"),
                default=None,
            )
            result.append(
                self._message(
                    f"{source_id}:result",
                    ConversationMessageRole.TOOL_RESULT,
                    output,
                    occurred_at,
                    source_order,
                    intra_order=1,
                    tool_call_id=call_id,
                    tool_name=tool_name or None,
                )
            )
        return result

    def _tool_call(
        self,
        raw: Any,
        *,
        occurred_at: datetime,
        source_id: str,
        source_order: int,
        intra_order: int,
    ) -> _ProjectedMessage:
        if not isinstance(raw, Mapping):
            raise ConversationProjectionError("assistant tool_calls must contain objects")
        call = dict(raw)
        function = call.get("function")
        function_map = dict(function) if isinstance(function, Mapping) else {}
        tool_name = str(call.get("tool_name") or call.get("name") or function_map.get("name") or "").strip()
        if not tool_name:
            raise ConversationProjectionError("tool_call is missing tool_name")
        call_id = str(call.get("tool_call_id") or call.get("id") or "")
        if not call_id:
            call_id = f"call-{stable_hash({'source_id': source_id, 'tool_name': tool_name})}"
        arguments = self._first_present(
            call,
            ("input", "arguments", "params"),
            default=function_map.get("arguments", {}),
        )
        return self._message(
            source_id,
            ConversationMessageRole.TOOL_CALL,
            arguments,
            occurred_at,
            source_order,
            intra_order=intra_order,
            tool_call_id=call_id,
            tool_name=tool_name,
        )

    @staticmethod
    def _message(
        message_id: str,
        role: ConversationMessageRole,
        content: Any,
        occurred_at: datetime,
        source_order: int,
        *,
        intra_order: int = 0,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> _ProjectedMessage:
        return _ProjectedMessage(
            ConversationMessage(
                message_id=message_id,
                role=role,
                content=content,
                occurred_at=occurred_at,
                sequence=0,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            ),
            source_order=source_order,
            intra_order=intra_order,
        )

    @staticmethod
    def _source_id(row: Mapping[str, Any], prefix: str, index: int) -> str:
        return str(row.get("message_id") or row.get("event_id") or row.get("id") or f"{prefix}:{index}")

    @staticmethod
    def _first_present(row: Mapping[str, Any], keys: tuple[str, ...], *, default: Any) -> Any:
        for key in keys:
            if key in row:
                return row[key]
        return default

    @classmethod
    def _text(cls, row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        value = cls._optional_text(row, keys)
        if not value:
            raise ConversationProjectionError("prompt message is missing text content")
        return value

    @staticmethod
    def _optional_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            if key in row:
                value = row[key]
                if value in (None, ""):
                    return ""
                if not isinstance(value, str):
                    raise ConversationProjectionError("prompt and completion content must be text")
                return value
        return ""

    @staticmethod
    def _time(value: Any, fallback: datetime) -> datetime:
        if value in (None, ""):
            return fallback
        try:
            return conversation_datetime(value, "conversation message time")
        except ConversationMessageError:
            return fallback

    @staticmethod
    def _conversation_id(archive: SessionArchive) -> str:
        try:
            return require_safe_path_segment(archive.session_id, "conversation_id")
        except ValueError:
            return "conversation-" + stable_hash(
                {"session_id": archive.session_id, "archive_uri": archive.archive_uri},
                length=32,
            )


__all__ = ["ConversationProjectionError", "SessionArchiveConversationProjector"]
