from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.core.ids import stable_hash


@dataclass(frozen=True)
class Observation:
    user_id: str
    raw_text: str = ""
    location: str = ""
    activity: str = ""
    signals: list[str] = field(default_factory=list)
    environment: dict = field(default_factory=dict)
    observed_at: str = ""
    explicit_scene_key: str = ""

    @property
    def scene_key(self) -> str:
        if self.explicit_scene_key:
            return self.explicit_scene_key
        tags = [self.location, self.activity, *sorted(self.signals), *self._environment_buckets()]
        return stable_hash([tag for tag in tags if tag], length=16)

    def context_tags(self) -> list[str]:
        return [tag for tag in [self.location, self.activity, *self.signals, *self._environment_buckets()] if tag]

    def _environment_buckets(self) -> list[str]:
        buckets = []
        temperature = self.environment.get("temperature")
        if isinstance(temperature, (int, float)):
            if temperature >= 29:
                buckets.append("hot_environment")
            elif temperature <= 18:
                buckets.append("cold_environment")
        return buckets
