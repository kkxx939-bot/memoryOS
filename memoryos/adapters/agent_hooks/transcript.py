from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TranscriptCursor:
    offset: int = 0
    inode: int | None = None


@dataclass(frozen=True)
class TranscriptDelta:
    messages: list[dict[str, Any]]
    cursor: TranscriptCursor
    truncated: bool = False
    parse_failed: bool = False


class TranscriptReader(Protocol):
    def read_since(self, transcript_path: str, cursor: TranscriptCursor | None) -> TranscriptDelta: ...


class GenericJsonlTranscriptReader:
    def __init__(self, max_bytes: int = 2_000_000) -> None:
        self.max_bytes = max_bytes

    def read_since(self, transcript_path: str, cursor: TranscriptCursor | None) -> TranscriptDelta:
        path = Path(transcript_path)
        stat = path.stat()
        previous = cursor or TranscriptCursor()
        offset = previous.offset
        truncated = stat.st_size < offset or (previous.inode is not None and previous.inode != stat.st_ino)
        if truncated:
            offset = 0
        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(self.max_bytes)
            new_offset = handle.tell()
        at_eof = new_offset >= stat.st_size
        if raw and not at_eof and not raw.endswith((b"\n", b"\r")):
            line_end = max(raw.rfind(b"\n"), raw.rfind(b"\r"))
            if line_end < 0:
                return TranscriptDelta([], previous, truncated, parse_failed=True)
            raw = raw[: line_end + 1]
            new_offset = offset + line_end + 1
        messages: list[dict[str, Any]] = []
        parse_failed = False
        for line in raw.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parse_failed = True
                continue
            if isinstance(item, dict):
                messages.append(self._normalize_item(item))
        next_cursor = previous if parse_failed else TranscriptCursor(new_offset, stat.st_ino)
        return TranscriptDelta(messages, next_cursor, truncated, parse_failed)

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return item


class ClaudeCodeTranscriptReader(GenericJsonlTranscriptReader):
    """Normalize Claude Code's message envelope while retaining unknown records."""

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        message = item.get("message")
        if isinstance(message, dict) and message.get("role"):
            return {
                "id": str(item.get("uuid") or message.get("id") or ""),
                "role": str(message["role"]),
                "content": message.get("content", ""),
            }
        return item


class CodexTranscriptReader(GenericJsonlTranscriptReader):
    """Normalize Codex rollout response_item message records."""

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = item.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "message" and payload.get("role"):
            return {
                "id": str(payload.get("id") or item.get("id") or ""),
                "role": str(payload["role"]),
                "content": payload.get("content", ""),
            }
        return item
