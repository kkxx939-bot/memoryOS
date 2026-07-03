from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memoryos.domain.memory.memory_item import utc_now
from memoryos.infrastructure.safety.path_safety import validate_identifier


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

    def mark_outbox_applied(self, outbox_event: dict) -> dict:
        updated = dict(outbox_event)
        updated["status"] = "applied"
        updated["applied_at"] = utc_now()
        user_id = str(updated.get("user_id", ""))
        if user_id:
            self._append(self._outbox_path(user_id), updated)
        return updated

    def pending_outbox_events(self, user_id: str | None = None, limit: int | None = None) -> list[dict]:
        user_ids = [user_id] if user_id else self._user_ids()
        pending: dict[str, dict] = {}
        applied: set[str] = set()
        for current_user_id in user_ids:
            if not current_user_id:
                continue
            for event in self._read_jsonl(self._outbox_path(current_user_id)):
                outbox_id = str(event.get("outbox_id", ""))
                if not outbox_id:
                    continue
                if event.get("status") == "applied":
                    applied.add(outbox_id)
                    pending.pop(outbox_id, None)
                    continue
                if event.get("status") == "pending" and outbox_id not in applied:
                    pending.setdefault(outbox_id, event)
        rows = list(pending.values())
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
