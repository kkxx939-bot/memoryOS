from __future__ import annotations

import hashlib
import json
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
        self._append(self._feedback_path(user_id), event)
        return event

    def append_outbox_event(self, user_id: str, event: dict) -> dict:
        validate_identifier(user_id, "user_id")
        existing = self.latest_outbox_event(user_id, str(event["event_id"]))
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
        self._append(self._outbox_path(user_id), outbox)
        return outbox

    def latest_outbox_event(self, user_id: str, outbox_id: str) -> dict | None:
        validate_identifier(user_id, "user_id")
        latest = None
        for event in self._read_jsonl(self._outbox_path(user_id)):
            if str(event.get("outbox_id", "")) == outbox_id:
                latest = event
        return latest

    def mark_outbox_applied(self, outbox_event: dict) -> dict:
        updated = dict(outbox_event)
        updated["status"] = "applied"
        updated["applied_at"] = utc_now()
        user_id = str(updated.get("user_id", ""))
        if user_id:
            self._append(self._outbox_path(user_id), updated)
        return updated

    def mark_outbox_processing(self, outbox_event: dict) -> dict:
        updated = dict(outbox_event)
        updated["status"] = "processing"
        updated["processing_started_at"] = utc_now()
        updated["retry_count"] = int(updated.get("retry_count", 0))
        user_id = str(updated.get("user_id", ""))
        if user_id:
            self._append(self._outbox_path(user_id), updated)
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
            self._append(self._outbox_path(user_id), updated)
        return updated

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
            for event in self._read_jsonl(self._outbox_path(current_user_id)):
                outbox_id = str(event.get("outbox_id", ""))
                if not outbox_id:
                    continue
                latest[outbox_id] = event
        rows = [
            event
            for event in latest.values()
            if event.get("status") in {"pending", "failed"}
            and int(event.get("retry_count", 0)) < max_retries
        ]
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        return rows[:limit] if limit is not None else rows

    def feedback_events(self, user_id: str, limit: int | None = None) -> list[dict]:
        validate_identifier(user_id, "user_id")
        rows = self._read_jsonl(self._feedback_path(user_id))
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        return rows[-limit:] if limit is not None else rows

    def _feedback_path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "events" / "feedback_events.jsonl"

    def _outbox_path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "events" / "outbox_events.jsonl"

    def _append(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
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
