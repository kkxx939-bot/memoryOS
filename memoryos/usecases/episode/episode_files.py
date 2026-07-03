from __future__ import annotations

import json
from pathlib import Path

from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.security.path_safety import validate_identifier


class EpisodeFileStore:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def episode_dir(self, user_id: str, episode_id: str) -> Path:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        path = self.store.root / "user" / user_id / "episodes" / episode_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, user_id: str, episode_id: str, filename: str) -> Path:
        return self.episode_dir(user_id, episode_id) / filename

    def write_json(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        self.path(user_id, episode_id, filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_json(self, user_id: str, episode_id: str, filename: str) -> dict:
        path = self.path(user_id, episode_id, filename)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def append_jsonl(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        path = self.path(user_id, episode_id, filename)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
