"""适配器里的会话服务。"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
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

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows only.
    _fcntl = None

_msvcrt: Any
try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX only.
    _msvcrt = None


class AgentSessionService:
    """负责 AgentSessionService 这部分逻辑。"""

    def __init__(
        self,
        root: str,
        tenant_id: str = "default",
        *,
        transcript_roots: tuple[str, ...] = (),
        max_transcript_bytes: int = 20_000_000,
    ) -> None:
        if (
            not isinstance(tenant_id, str)
            or not tenant_id.strip()
            or tenant_id in {".", ".."}
            or "/" in tenant_id
            or "\\" in tenant_id
        ):
            raise ValueError("tenant_id must be one safe non-empty path segment")
        self.tenant_id = tenant_id
        self.root = (
            Path(root) / "agent-sessions" / "live"
            if tenant_id == "default"
            else Path(root) / "tenants" / tenant_id / "agent-sessions" / "live"
        )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.transcript_roots = tuple(Path(item).expanduser().resolve() for item in transcript_roots)
        self.readers = {
            "claude_code": ClaudeCodeTranscriptReader(max_file_bytes=max_transcript_bytes),
            "codex": CodexTranscriptReader(max_file_bytes=max_transcript_bytes),
        }
        self.default_reader = GenericJsonlTranscriptReader(max_file_bytes=max_transcript_bytes)

    def append_event(self, event: NormalizedAgentEvent) -> bool:
        self._require_event_tenant(event)
        with self._session_guard(event.session_key):
            return self._append_event_locked(event)

    def _append_event_locked(self, event: NormalizedAgentEvent) -> bool:
        rows = self._events_locked(event.session_key)
        deduplicated_rows: list[dict[str, Any]] = []
        row_event_ids: set[str] = set()
        for row in rows:
            self._require_row_boundary(row, event)
            row_event_id = row.get("event_id")
            if not isinstance(row_event_id, str) or not row_event_id:
                raise PermissionError("live session event is missing event_id")
            if row_event_id in row_event_ids:
                continue
            row_event_ids.add(row_event_id)
            deduplicated_rows.append(row)
        if len(deduplicated_rows) != len(rows):
            self._write_events(event.session_key, deduplicated_rows)
        state = self._state(event.session_key)
        if event.event_id in row_event_ids:
            if not self._state_matches_rows(state, event, row_event_ids):
                self._write_state(
                    event.session_key,
                    self._event_state(state, event, row_event_ids, mark_archived=True),
                )
            return False
        payload = asdict(event)
        payload["event_type"] = event.event_type.value
        sanitized = sanitize_payload(payload)
        if not isinstance(sanitized, dict):
            raise ValueError("live session event must serialize to an object")
        deduplicated_rows.append(sanitized)
        row_event_ids.add(event.event_id)
        self._write_events(event.session_key, deduplicated_rows)
        self._write_state(
            event.session_key,
            self._event_state(state, event, row_event_ids, mark_archived=True),
        )
        return True

    def append_transcript(self, event: NormalizedAgentEvent) -> int:
        self._require_event_tenant(event)
        with self._session_guard(event.session_key):
            self._require_state_boundary(self._state(event.session_key), event)
            if not event.transcript_path:
                return 0
            state = self._state(event.session_key)
            raw_cursor = state.get("transcript_cursor") or {}
            cursor = TranscriptCursor(int(raw_cursor.get("offset", 0)), raw_cursor.get("inode"))
            reader = self.readers.get(event.adapter_id, self.default_reader)
            try:
                workspace_roots = tuple(
                    dict.fromkeys(
                        str(item)
                        for item in (
                            event.repo_root,
                            event.cwd,
                            *self.transcript_roots,
                        )
                        if item
                    )
                )
                delta = reader.read_since(
                    event.transcript_path,
                    cursor,
                    allowed_roots=workspace_roots,
                )
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
                appended += int(self._append_event_locked(child))
            state = self._state(event.session_key)
            state["transcript_cursor"] = asdict(delta.cursor)
            state["transcript_truncated"] = delta.truncated
            self._write_state(event.session_key, state)
            return appended

    def events(self, session_key: str) -> list[dict[str, Any]]:
        with self._session_guard(session_key):
            return self._events_locked(session_key)

    def _events_locked(self, session_key: str) -> list[dict[str, Any]]:
        path = self._events_path(session_key)
        if not path.exists():
            return []
        result: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"live session event log contains invalid JSON at line {line_number}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"live session event log contains a non-object row at line {line_number}")
                result.append(item)
        return result

    def commit_payload(self, event: NormalizedAgentEvent) -> dict[str, Any]:
        self._require_event_tenant(event)
        with self._session_guard(event.session_key):
            return self._commit_payload_locked(event)

    def _commit_payload_locked(self, event: NormalizedAgentEvent) -> dict[str, Any]:
        self._require_state_boundary(self._state(event.session_key), event)
        messages: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        used_contexts: list[dict[str, Any]] = []
        used_skills: list[dict[str, Any]] = []
        explicit_scope: dict[str, Any] = {}
        explicit_provenance: dict[str, Any] = {}
        for item in self._events_locked(event.session_key):
            self._require_row_boundary(item, event)
            prompt = item.get("prompt")
            if prompt:
                messages.append({"id": item.get("event_id"), "role": "user", "content": prompt})
            messages.extend(row for row in item.get("messages", []) if isinstance(row, dict))
            if isinstance(item.get("tool_call"), dict):
                tool_results.append({"event_id": item.get("event_id"), **item["tool_call"]})
            metadata = dict(item.get("metadata", {}) or {})
            used_contexts.extend(
                dict(value)
                for value in metadata.get("used_contexts", []) or []
                if isinstance(value, dict)
            )
            used_skills.extend(
                dict(value)
                for value in metadata.get("used_skills", []) or []
                if isinstance(value, dict)
            )
            tool_results.extend(
                dict(value)
                for value in metadata.get("tool_results", []) or []
                if isinstance(value, dict)
            )
            if isinstance(metadata.get("scope"), dict):
                explicit_scope.update(metadata["scope"])
            if isinstance(metadata.get("provenance"), dict):
                explicit_provenance.update(metadata["provenance"])
        scope = {
            **explicit_scope,
            "user_id": event.user_id,
            "tenant_id": event.tenant_id,
            "project_id": event.project_id,
            "branch": event.branch or "",
            "worktree_id": event.worktree_id or "",
            "session_key": event.session_key,
        }
        provenance = {
            **explicit_provenance,
            "native_session_id": event.native_session_id,
            "event_id": event.event_id,
            "agent_name": event.metadata.get("agent_name", event.adapter_id),
            "source_uri": event.transcript_path or "",
            "captured_at": event.timestamp,
        }
        return {
            "user_id": event.user_id,
            "tenant_id": event.tenant_id,
            "session_id": event.native_session_id,
            "session_key": event.session_key,
            "project_id": event.project_id,
            "messages": sanitize_payload(messages),
            "tool_results": sanitize_payload(tool_results),
            "used_contexts": sanitize_payload(used_contexts),
            "used_skills": sanitize_payload(used_skills),
            "scope": scope,
            "provenance": provenance,
        }

    def record_recall(self, session_key: str, trace: dict[str, Any]) -> None:
        with self._session_guard(session_key):
            with (self.root / f"{session_key}.recall.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sanitize_payload(trace), ensure_ascii=False) + "\n")

    def checkpoint(self, session_key: str) -> dict[str, Any]:
        with self._session_guard(session_key):
            state = self._state(session_key)
            self._require_state_tenant(state)
            if str(state.get("status") or "OPEN") in {"OPEN", "ARCHIVED"}:
                state["status"] = "ARCHIVED"
            state["checkpointed_at"] = utc_now()
            self._write_state(session_key, state)
            return {
                "session_key": session_key,
                "event_count": len(self._events_locked(session_key)),
                "status": "ARCHIVED",
            }

    def finalize(self, session_key: str, *, commit_state: str = "COMMITTED") -> dict[str, Any]:
        with self._session_guard(session_key):
            state = self._state(session_key)
            self._require_state_tenant(state)
            event_count = len(self._events_locked(session_key))
            current = str(state.get("status") or "ARCHIVED")
            requested = str(commit_state).upper()
            allowed = {
                "OPEN": {"ARCHIVED"},
                "ARCHIVED": {"ARCHIVED", "QUEUED", "PROCESSING", "COMMITTED"},
                "QUEUED": {"QUEUED", "PROCESSING", "COMMITTED", "FAILED_RETRYABLE", "DEAD_LETTER"},
                "PROCESSING": {"PROCESSING", "COMMITTED", "FAILED_RETRYABLE", "DEAD_LETTER"},
                "FAILED_RETRYABLE": {"FAILED_RETRYABLE", "QUEUED", "PROCESSING", "DEAD_LETTER"},
                "COMMITTED": {"COMMITTED"},
                "DEAD_LETTER": {"DEAD_LETTER"},
            }
            if requested not in allowed.get(current, set()):
                raise ValueError(f"illegal session state transition: {current} -> {requested}")
            state.update({"status": commit_state, "finalized_at": utc_now()})
            self._write_state(session_key, state)
            return {"session_key": session_key, "event_count": event_count, "status": commit_state}

    def recover_orphan_sessions(self) -> list[str]:
        sessions = []
        for path in self.root.glob("*.state.json"):
            session_key = path.name.removesuffix(".state.json")
            with self._session_guard(session_key):
                if self._read_json(path).get("status") != "COMMITTED":
                    sessions.append(session_key)
        return sessions

    def _events_path(self, session_key: str) -> Path:
        return self.root / f"{session_key}.jsonl"

    def _state_path(self, session_key: str) -> Path:
        return self.root / f"{session_key}.state.json"

    def _lock_path(self, session_key: str) -> Path:
        return self.root / f"{session_key}.lock"

    @contextmanager
    def _session_guard(self, session_key: str) -> Iterator[None]:
        backend = _lock_backend()
        if backend is None:
            raise RuntimeError("AgentSessionService requires fcntl or msvcrt file locking")
        lock_path = self._lock_path(session_key)
        if not lock_path.exists():
            lock_path.touch(mode=0o600)
        lock_path.chmod(0o600)
        with lock_path.open("a+b") as lock_file:
            _lock_file(lock_file, backend)
            try:
                yield
            finally:
                _unlock_file(lock_file, backend)

    def _state(self, session_key: str) -> dict[str, Any]:
        return self._read_json(self._state_path(session_key))

    def _require_event_tenant(self, event: NormalizedAgentEvent) -> None:
        if event.tenant_id != self.tenant_id:
            raise PermissionError("live session event tenant does not match the session store")

    def _require_state_tenant(self, state: dict[str, Any]) -> None:
        if state and str(state.get("tenant_id") or "default") != self.tenant_id:
            raise PermissionError("live session state tenant does not match the session store")

    def _require_state_boundary(self, state: dict[str, Any], event: NormalizedAgentEvent) -> None:
        self._require_state_tenant(state)
        if not state:
            return
        expected = {
            "user_id": event.user_id,
            "project_id": event.project_id,
            "adapter_id": event.adapter_id,
            "native_session_id": event.native_session_id,
        }
        for field_name, value in expected.items():
            persisted = state.get(field_name)
            if persisted not in {None, ""} and str(persisted) != value:
                raise PermissionError(f"live session state {field_name} boundary mismatch")

    def _require_row_boundary(self, row: dict[str, Any], event: NormalizedAgentEvent) -> None:
        expected = {
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "project_id": event.project_id,
            "adapter_id": event.adapter_id,
            "native_session_id": event.native_session_id,
            "session_key": event.session_key,
        }
        for field_name, value in expected.items():
            persisted = row.get(field_name)
            if field_name == "tenant_id" and persisted in {None, ""}:
                persisted = "default"
            if str(persisted or "") != value:
                raise PermissionError(f"live session event {field_name} boundary mismatch")

    def _state_matches_rows(
        self,
        state: dict[str, Any],
        event: NormalizedAgentEvent,
        event_ids: set[str],
    ) -> bool:
        raw_event_ids = state.get("event_ids")
        if (
            not isinstance(raw_event_ids, list)
            or any(not isinstance(item, str) or not item for item in raw_event_ids)
            or set(raw_event_ids) != event_ids
        ):
            return False
        expected = {
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "project_id": event.project_id,
            "adapter_id": event.adapter_id,
            "native_session_id": event.native_session_id,
            "session_key": event.session_key,
        }
        return all(str(state.get(field_name) or "") == value for field_name, value in expected.items())

    def _event_state(
        self,
        state: dict[str, Any],
        event: NormalizedAgentEvent,
        event_ids: set[str],
        *,
        mark_archived: bool,
    ) -> dict[str, Any]:
        updated = {
            **state,
            "event_ids": sorted(event_ids),
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "project_id": event.project_id,
            "adapter_id": event.adapter_id,
            "native_session_id": event.native_session_id,
            "session_key": event.session_key,
            "updated_at": utc_now(),
        }
        if mark_archived:
            current = str(updated.get("status") or "OPEN")
            if current in {"OPEN", "ARCHIVED"}:
                updated["status"] = "ARCHIVED"
        return updated

    def _write_events(self, session_key: str, rows: list[dict[str, Any]]) -> None:
        path = self._events_path(session_key)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        text = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(text + ("\n" if text else ""))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_state(self, session_key: str, state: dict[str, Any]) -> None:
        path = self._state_path(session_key)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(json.dumps(state, ensure_ascii=False, indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}


def _lock_backend() -> str | None:
    if _fcntl is not None:
        return "fcntl"
    if _msvcrt is not None:
        return "msvcrt"
    return None


def _lock_file(lock_file: Any, backend: str) -> None:
    lock_file.seek(0)
    if backend == "fcntl":
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
        return
    if not lock_file.read(1):
        lock_file.seek(0)
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_LOCK, 1)


def _unlock_file(lock_file: Any, backend: str) -> None:
    lock_file.seek(0)
    if backend == "fcntl":
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        return
    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
