from __future__ import annotations

import json
from pathlib import Path


class PolicyStats:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        predicted_action: str,
        recommended_intervention: str,
        reward: float,
    ) -> dict:
        data = self._load()
        key = self._key(predicted_action, recommended_intervention)
        entry = data.setdefault(
            key,
            {
                "predicted_action": predicted_action,
                "recommended_intervention": recommended_intervention,
                "count": 0,
                "total_reward": 0.0,
                "average_reward": 0.0,
            },
        )
        entry["count"] += 1
        entry["total_reward"] = float(entry["total_reward"]) + reward
        entry["average_reward"] = entry["total_reward"] / entry["count"]
        self._save(data)
        return entry

    def load(self) -> dict:
        return self._load()

    def _key(self, predicted_action: str, recommended_intervention: str) -> str:
        return f"{predicted_action}::{recommended_intervention}"

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
