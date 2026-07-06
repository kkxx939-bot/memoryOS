from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from memoryos.behavior.model.behavior_case import BehaviorCase


@dataclass(frozen=True)
class BehaviorWindowDecision:
    scene_key: str
    similar_refs_3d: list[str] = field(default_factory=list)
    similar_refs_7d: list[str] = field(default_factory=list)
    similar_refs_30d: list[str] = field(default_factory=list)
    create_cluster: bool = False
    create_pattern: bool = False
    archive_stale_single: bool = False
    similarity_key: tuple[str, ...] = ()


class BehaviorWindowEvaluator:
    def evaluate(
        self,
        scene_key: str,
        current_cases: list[BehaviorCase],
        historical_hits: list[dict],
        now: datetime | None = None,
    ) -> BehaviorWindowDecision:
        now = now or datetime.now(timezone.utc)
        current_records = [self._record_from_case(case, current=True) for case in current_cases]
        if not current_records:
            return BehaviorWindowDecision(scene_key=scene_key)
        key = current_records[0]["similarity_key"]
        matching_history = [
            record
            for record in historical_hits
            if record.get("scene_key") == scene_key and tuple(record.get("similarity_key", ())) == key
        ]
        current_refs = [str(record["uri"]) for record in current_records]
        refs_3d = [*self._refs_within(matching_history, now, 3), *current_refs]
        refs_7d = [*self._refs_within(matching_history, now, 7), *current_refs]
        refs_30d = [*self._refs_within(matching_history, now, 30), *current_refs]
        stale_single = len(refs_3d) == 1 and bool(matching_history) and not self._refs_within(matching_history, now, 3)
        return BehaviorWindowDecision(
            scene_key=scene_key,
            similar_refs_3d=list(dict.fromkeys(refs_3d)),
            similar_refs_7d=list(dict.fromkeys(refs_7d)),
            similar_refs_30d=list(dict.fromkeys(refs_30d)),
            create_cluster=len(refs_3d) >= 2,
            create_pattern=len(refs_7d) >= 3 or len(refs_30d) >= 4,
            archive_stale_single=stale_single,
            similarity_key=key,
        )

    def historical_record(self, uri: str, metadata: dict) -> dict:
        observation = dict(metadata.get("observation", {}))
        return {
            "uri": uri,
            "scene_key": str(metadata.get("scene_key", observation.get("scene_key", ""))),
            "similarity_key": self._similarity_key(observation),
            "created_at": str(metadata.get("observed_at") or observation.get("observed_at") or metadata.get("created_at", "")),
        }

    def _record_from_case(self, case: BehaviorCase, current: bool) -> dict:
        return {
            "uri": f"memoryos://user/{case.user_id}/behavior/cases/{case.scene_key}/{case.case_id}",
            "scene_key": case.scene_key,
            "similarity_key": self._similarity_key(case.observation),
            "created_at": str(case.observation.get("observed_at") or case.created_at),
            "current": current,
        }

    def _similarity_key(self, observation: dict) -> tuple[str, ...]:
        tags: list[str] = []
        tags.extend(str(tag) for tag in observation.get("context_tags", []) if tag)
        for key in ("location", "activity"):
            if observation.get(key):
                tags.append(str(observation[key]))
        tags.extend(str(tag) for tag in observation.get("signals", []) if tag)
        environment = observation.get("environment", {})
        if isinstance(environment, dict):
            temperature = environment.get("temperature")
            if isinstance(temperature, int | float):
                if temperature >= 29:
                    tags.append("hot_environment")
                elif temperature <= 18:
                    tags.append("cold_environment")
        return tuple(sorted(set(tags)))

    def _refs_within(self, records: list[dict], now: datetime, days: int) -> list[str]:
        refs = []
        for record in records:
            created = self._parse_time(str(record.get("created_at", "")))
            if created is not None and (now - created).days <= days:
                refs.append(str(record["uri"]))
        return refs

    def _parse_time(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
