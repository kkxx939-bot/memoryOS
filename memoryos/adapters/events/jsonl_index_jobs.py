from __future__ import annotations

import json
from pathlib import Path

from memoryos.domain.memory.memory_item import utc_now
from memoryos.ports.repositories.index_job_repository import INDEX_JOB_STATES, IndexJob
from memoryos.security.path_safety import validate_identifier


class JsonlIndexJobRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    def enqueue(self, job: IndexJob) -> dict:
        validate_identifier(job.user_id, "user_id")
        row = {**job.to_dict(), "created_at": utc_now(), "updated_at": utc_now()}
        self._append(self._path(job.user_id), row)
        return row

    def pending(self, user_id: str | None = None, limit: int = 50) -> list[dict]:
        user_ids = [user_id] if user_id else self._user_ids()
        latest: dict[str, dict] = {}
        for current_user_id in user_ids:
            if not current_user_id:
                continue
            for row in self._read_jsonl(self._path(current_user_id)):
                latest[str(row.get("job_id", ""))] = row
        rows = [
            row
            for row in latest.values()
            if row.get("status") in {"pending", "stale", "delete_pending"}
        ]
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        return rows[:limit]

    def mark(self, job_id: str, status: str, patch: dict | None = None) -> dict:
        if status not in INDEX_JOB_STATES:
            raise ValueError(f"Unknown index job status: {status}")
        for user_id in self._user_ids():
            rows = self._read_jsonl(self._path(user_id))
            current = next((row for row in reversed(rows) if row.get("job_id") == job_id), None)
            if current is None:
                continue
            updated = {**current, **(patch or {}), "status": status, "updated_at": utc_now()}
            self._append(self._path(user_id), updated)
            return updated
        raise KeyError(f"Index job not found: {job_id}")

    def _path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "events" / "index_jobs.jsonl"

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _user_ids(self) -> list[str]:
        user_root = self.root / "user"
        if not user_root.exists():
            return []
        return sorted(path.name for path in user_root.iterdir() if path.is_dir())
