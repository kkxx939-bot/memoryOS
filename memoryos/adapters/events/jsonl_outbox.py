from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from memoryos.domain.memory.memory_item import utc_now
from memoryos.security.path_safety import validate_identifier


class FeedbackEventStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def append_feedback_event(self, user_id: str, episode_id: str, payload: dict) -> dict:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        event = {
            "event_id": self._event_id(user_id, episode_id, payload),
            "event_type": "FeedbackRecorded",
            "user_id": user_id,
            "episode_id": episode_id,
            "created_at": utc_now(),
            "payload": payload,
        }
        with self._locked_user_events(user_id):
            self._append_unlocked(self._feedback_path(user_id), event)
        return event

    def append_outbox_event(self, user_id: str, event: dict) -> dict:
        validate_identifier(user_id, "user_id")
        with self._locked_user_events(user_id):
            return self._append_outbox_event_unlocked(user_id, event)

    def append_feedback_and_outbox(self, user_id: str, episode_id: str, payload: dict) -> tuple[dict, dict]:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        event = {
            "event_id": self._event_id(user_id, episode_id, payload),
            "event_type": "FeedbackRecorded",
            "user_id": user_id,
            "episode_id": episode_id,
            "created_at": utc_now(),
            "payload": payload,
        }
        with self._locked_user_events(user_id):
            existing_event = self._latest_feedback_event_unlocked(user_id, str(event["event_id"]))
            if existing_event:
                event = existing_event
            else:
                self._append_unlocked(self._feedback_path(user_id), event)
            outbox = self._append_outbox_event_unlocked(user_id, event)
        return event, outbox

    def latest_outbox_event(self, user_id: str, outbox_id: str) -> dict | None:
        validate_identifier(user_id, "user_id")
        with self._locked_user_events(user_id):
            return self._latest_outbox_event_unlocked(user_id, outbox_id)

    def mark_outbox_applied(self, outbox_event: dict) -> dict:
        updated = dict(outbox_event)
        updated["status"] = "applied"
        updated["applied_at"] = utc_now()
        user_id = str(updated.get("user_id", ""))
        if user_id:
            with self._locked_user_events(user_id):
                self._append_unlocked(self._outbox_path(user_id), updated)
        return updated

    def mark_outbox_processing(self, outbox_event: dict) -> dict:
        updated = dict(outbox_event)
        updated["status"] = "processing"
        updated["processing_started_at"] = utc_now()
        updated["retry_count"] = int(updated.get("retry_count", 0))
        user_id = str(updated.get("user_id", ""))
        if user_id:
            with self._locked_user_events(user_id):
                self._append_unlocked(self._outbox_path(user_id), updated)
        return updated

    def mark_outbox_failed(self, outbox_event: dict, error: str, max_retries: int = 3) -> dict:
        updated = dict(outbox_event)
        retry_count = int(updated.get("retry_count", 0)) + 1
        updated["retry_count"] = retry_count
        updated["last_error"] = str(error)[:1000]
        updated["failed_at"] = utc_now()
        updated["status"] = "dead_letter" if retry_count >= max_retries else "failed"
        user_id = str(updated.get("user_id", ""))
        if user_id:
            with self._locked_user_events(user_id):
                self._append_unlocked(self._outbox_path(user_id), updated)
        return updated

    def claim_pending_outbox_events(
        self,
        user_id: str | None = None,
        limit: int | None = None,
        max_retries: int = 3,
        worker_id: str = "",
        lease_seconds: int = 60,
    ) -> list[dict]:
        user_ids = [user_id] if user_id else self._user_ids()
        claimed: list[dict] = []
        for current_user_id in user_ids:
            if not current_user_id:
                continue
            validate_identifier(current_user_id, "user_id")
            with self._locked_user_events(current_user_id):
                latest = self._latest_outbox_events_unlocked(current_user_id)
                rows = [
                    event
                    for event in latest.values()
                    if self._claimable(event, max_retries=max_retries)
                ]
                rows.sort(key=lambda item: str(item.get("created_at", "")))
                remaining = None if limit is None else max(0, limit - len(claimed))
                if remaining == 0:
                    break
                for event in rows[:remaining]:
                    claimed_event = dict(event)
                    claimed_event["status"] = "processing"
                    claimed_event["processing_started_at"] = utc_now()
                    claimed_event["locked_by"] = worker_id or f"pid:{os.getpid()}"
                    claimed_event["locked_until"] = self._lease_until(lease_seconds)
                    claimed_event["retry_count"] = int(claimed_event.get("retry_count", 0))
                    self._append_unlocked(self._outbox_path(current_user_id), claimed_event)
                    claimed.append(claimed_event)
        return claimed

    def pending_outbox_events(
        self,
        user_id: str | None = None,
        limit: int | None = None,
        max_retries: int = 3,
    ) -> list[dict]:
        user_ids = [user_id] if user_id else self._user_ids()
        latest: dict[str, dict] = {}
        for current_user_id in user_ids:
            if not current_user_id:
                continue
            with self._locked_user_events(current_user_id):
                latest.update(self._latest_outbox_events_unlocked(current_user_id))
        rows = [
            event
            for event in latest.values()
            if self._claimable(event, max_retries=max_retries)
        ]
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        return rows[:limit] if limit is not None else rows

    def feedback_events(self, user_id: str, limit: int | None = None) -> list[dict]:
        validate_identifier(user_id, "user_id")
        with self._locked_user_events(user_id):
            rows = self._read_jsonl_unlocked(self._feedback_path(user_id))
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        return rows[-limit:] if limit is not None else rows

    def _feedback_path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "events" / "feedback_events.jsonl"

    def _outbox_path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "events" / "outbox_events.jsonl"

    def _append(self, path: Path, payload: dict) -> None:
        with self._locked_path(path.parent / ".events.lock"):
            self._append_unlocked(path, payload)

    def _append_unlocked(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        with self._locked_path(path.parent / ".events.lock"):
            return self._read_jsonl_unlocked(path)

    def _read_jsonl_unlocked(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    def _user_ids(self) -> list[str]:
        user_root = self.root / "user"
        if not user_root.exists():
            return []
        return sorted(path.name for path in user_root.iterdir() if path.is_dir())

    def _event_id(self, user_id: str, episode_id: str, payload: dict) -> str:
        stable_payload = {key: value for key, value in payload.items() if key != "created_at"}
        material = json.dumps(
            {"user_id": user_id, "episode_id": episode_id, "payload": stable_payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def _append_outbox_event_unlocked(self, user_id: str, event: dict) -> dict:
        existing = self._latest_outbox_event_unlocked(user_id, str(event["event_id"]))
        if existing:
            return existing
        outbox = {
            "outbox_id": event["event_id"],
            "event_type": event["event_type"],
            "user_id": user_id,
            "episode_id": event.get("episode_id", ""),
            "created_at": utc_now(),
            "status": "pending",
            "payload": event,
        }
        self._append_unlocked(self._outbox_path(user_id), outbox)
        return outbox

    def _latest_feedback_event_unlocked(self, user_id: str, event_id: str) -> dict | None:
        latest = None
        for event in self._read_jsonl_unlocked(self._feedback_path(user_id)):
            if str(event.get("event_id", "")) == event_id:
                latest = event
        return latest

    def _latest_outbox_event_unlocked(self, user_id: str, outbox_id: str) -> dict | None:
        return self._latest_outbox_events_unlocked(user_id).get(outbox_id)

    def _latest_outbox_events_unlocked(self, user_id: str) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for event in self._read_jsonl_unlocked(self._outbox_path(user_id)):
            outbox_id = str(event.get("outbox_id", ""))
            if outbox_id:
                latest[outbox_id] = event
        return latest

    def _claimable(self, event: dict, max_retries: int) -> bool:
        status = str(event.get("status", ""))
        if int(event.get("retry_count", 0)) >= max_retries:
            return False
        if status in {"pending", "failed"}:
            return True
        if status == "processing":
            return self._lease_expired(str(event.get("locked_until", "")))
        return False

    def _lease_until(self, lease_seconds: int) -> str:
        now = datetime.now(timezone.utc).timestamp()
        return datetime.fromtimestamp(now + max(1, lease_seconds), tz=timezone.utc).isoformat()

    def _lease_expired(self, value: str) -> bool:
        if not value:
            return True
        try:
            expires_at = datetime.fromisoformat(value)
        except ValueError:
            return True
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at <= datetime.now(timezone.utc)

    @contextmanager
    def _locked_user_events(self, user_id: str) -> Iterator[None]:
        with self._locked_path(self.root / "user" / user_id / "events" / ".events.lock"):
            yield

    @contextmanager
    def _locked_path(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
