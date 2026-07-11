"""适配器里的会话服务。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memoryos.adapters.agent_hooks.events import AgentEventType, NormalizedAgentEvent
from memoryos.adapters.agent_hooks.sanitizer import sanitize_error_text, sanitize_payload
from memoryos.adapters.agent_hooks.transcript import (
    ClaudeCodeTranscriptReader,
    CodexTranscriptReader,
    GenericJsonlTranscriptReader,
    TranscriptCursor,
)
from memoryos.core.time import utc_now

logger = logging.getLogger(__name__)


class AgentSessionService:
    """负责 AgentSessionService 这部分逻辑。"""

    def __init__(self, root: str) -> None:
        self.root = Path(root) / "agent-sessions" / "live"
        self.root.mkdir(parents=True, exist_ok=True)
        self.readers = {
            "claude_code": ClaudeCodeTranscriptReader(),
            "codex": CodexTranscriptReader(),
        }
        self.default_reader = GenericJsonlTranscriptReader()

    def append_event(self, event: NormalizedAgentEvent) -> bool:
        state = self._state(event.session_key)
        seen = set(state.get("event_ids", []))
        if event.event_id in seen:
            return False
        payload = asdict(event)
        payload["event_type"] = event.event_type.value
        with self._events_path(event.session_key).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitize_payload(payload), ensure_ascii=False, default=str) + "\n")
        seen.add(event.event_id)
        state.update(
            {
                "event_ids": sorted(seen),
                "project_id": event.project_id,
                "adapter_id": event.adapter_id,
                "native_session_id": event.native_session_id,
                "updated_at": utc_now(),
                "status": "ARCHIVED",
            }
        )
        self._write_state(event.session_key, state)
        return True

    def append_transcript(self, event: NormalizedAgentEvent) -> int:
        if not event.transcript_path:
            return 0
        state = self._state(event.session_key)
        raw_cursor = state.get("transcript_cursor") or {}
        cursor = TranscriptCursor(int(raw_cursor.get("offset", 0)), raw_cursor.get("inode"))
        reader = self.readers.get(event.adapter_id, self.default_reader)
        try:
            delta = reader.read_since(event.transcript_path, cursor)
        except OSError as exc:
            error_message = sanitize_error_text(str(exc))
            state["transcript_error"] = {
                "code": "TRANSCRIPT_UNAVAILABLE",
                "message": error_message,
                "retryable": True,
                "operation": "read_transcript",
                "updated_at": utc_now(),
            }
            self._write_state(event.session_key, state)
            logger.warning(
                "transcript unavailable; event remains archived",
                extra={"operation": "read_transcript", "retryable": True, "error": error_message},
            )
            return 0
        if delta.parse_failed:
            state["transcript_error"] = "parse_failed"
            self._write_state(event.session_key, state)
            return 0
        appended = 0
        for index, message in enumerate(delta.messages):
            child = NormalizedAgentEvent(
                **{
                    **asdict(event),
                    "event_id": f"{event.event_id}:transcript:{cursor.offset}:{index}",
                    "event_type": AgentEventType.TURN_END,
                    "messages": [message],
                    "prompt": None,
                    "tool_call": None,
                }
            )
            appended += int(self.append_event(child))
        state = self._state(event.session_key)
        state["transcript_cursor"] = asdict(delta.cursor)
        state["transcript_truncated"] = delta.truncated
        self._write_state(event.session_key, state)
        return appended

    def events(self, session_key: str) -> list[dict[str, Any]]:
        path = self._events_path(session_key)
        if not path.exists():
            return []
        result: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    result.append(item)
        return result

    def commit_payload(self, event: NormalizedAgentEvent) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for item in self.events(event.session_key):
            prompt = item.get("prompt")
            if prompt:
                messages.append({"id": item.get("event_id"), "role": "user", "content": prompt})
            messages.extend(row for row in item.get("messages", []) if isinstance(row, dict))
            if isinstance(item.get("tool_call"), dict):
                tool_results.append({"event_id": item.get("event_id"), **item["tool_call"]})
        scope = {
            "user_id": event.user_id,
            "project_id": event.project_id,
            "branch": event.branch or "",
            "worktree_id": event.worktree_id or "",
            "session_key": event.session_key,
        }
        provenance = {
            "native_session_id": event.native_session_id,
            "event_id": event.event_id,
            "agent_name": event.metadata.get("agent_name", event.adapter_id),
            "source_uri": event.transcript_path or "",
            "captured_at": event.timestamp,
        }
        return {
            "user_id": event.user_id,
            "session_id": event.native_session_id,
            "session_key": event.session_key,
            "project_id": event.project_id,
            "messages": sanitize_payload(messages),
            "tool_results": sanitize_payload(tool_results),
            "used_contexts": sanitize_payload(event.metadata.get("used_contexts", [])),
            "scope": scope,
            "provenance": provenance,
        }

    def record_recall(self, session_key: str, trace: dict[str, Any]) -> None:
        with (self.root / f"{session_key}.recall.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitize_payload(trace), ensure_ascii=False) + "\n")

    def checkpoint(self, session_key: str) -> dict[str, Any]:
        state = self._state(session_key)
        state.update({"status": "ARCHIVED", "checkpointed_at": utc_now()})
        self._write_state(session_key, state)
        return {"session_key": session_key, "event_count": len(self.events(session_key)), "status": "ARCHIVED"}

    def finalize(self, session_key: str, *, commit_state: str = "COMMITTED") -> dict[str, Any]:
        state = self._state(session_key)
        if state.get("status") == "COMMITTED":
            return {"session_key": session_key, "event_count": len(self.events(session_key)), "status": "COMMITTED"}
        state.update({"status": commit_state, "finalized_at": utc_now()})
        self._write_state(session_key, state)
        return {"session_key": session_key, "event_count": len(self.events(session_key)), "status": commit_state}

    def recover_orphan_sessions(self) -> list[str]:
        return [path.name.removesuffix(".state.json") for path in self.root.glob("*.state.json") if self._read_json(path).get("status") != "COMMITTED"]

    def _events_path(self, session_key: str) -> Path:
        return self.root / f"{session_key}.jsonl"

    def _state_path(self, session_key: str) -> Path:
        return self.root / f"{session_key}.state.json"

    def _state(self, session_key: str) -> dict[str, Any]:
        return self._read_json(self._state_path(session_key))

    def _write_state(self, session_key: str, state: dict[str, Any]) -> None:
        path = self._state_path(session_key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
