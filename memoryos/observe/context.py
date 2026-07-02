from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ObservationContext:
    raw_text: str = ""
    location: str | None = None
    activity: str | None = None
    started_at: str | None = None
    observed_at: str | None = None
    duration_minutes: int | None = None
    signals: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def computed_duration_minutes(self) -> int | None:
        if self.duration_minutes is not None:
            return max(0, int(self.duration_minutes))
        if not self.started_at or not self.observed_at:
            return None
        start = _parse_datetime(self.started_at)
        observed = _parse_datetime(self.observed_at)
        if not start or not observed:
            return None
        return max(0, int((observed - start).total_seconds() // 60))

    def time_of_day(self) -> str | None:
        if not self.observed_at:
            return None
        observed = _parse_datetime(self.observed_at)
        if not observed:
            return None
        hour = observed.hour
        if 5 <= hour < 11:
            return "morning"
        if 11 <= hour < 14:
            return "noon"
        if 14 <= hour < 18:
            return "afternoon"
        if 18 <= hour < 23:
            return "evening"
        return "night"

    def context_tags(self) -> list[str]:
        tags: list[str] = []
        self._append_tag(tags, self.location)
        self._append_tag(tags, self.activity)
        self._append_tag(tags, self.time_of_day())

        duration = self.computed_duration_minutes()
        if duration is not None:
            tags.append(f"duration_{duration}_minutes")
            if duration >= 120:
                tags.append("duration_120m_plus")
            elif duration >= 60:
                tags.append("duration_60m_plus")
            elif duration >= 30:
                tags.append("duration_30m_plus")
            elif duration >= 10:
                tags.append("duration_10m_plus")

        for signal in self.signals:
            self._append_tag(tags, signal)

        temperature = self._number_from_environment("temperature")
        if temperature is not None:
            tags.append(f"temperature_{round(temperature)}")
            if temperature >= 28:
                tags.append("hot_environment")
            elif temperature <= 16:
                tags.append("cold_environment")

        humidity = self._number_from_environment("humidity")
        if humidity is not None:
            tags.append(f"humidity_{round(humidity)}")
            if humidity >= 70:
                tags.append("humid_environment")

        for key, value in sorted(self.environment.items()):
            if key in {"temperature", "humidity"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                self._append_tag(tags, f"{key}_{value}")

        deduped = []
        seen = set()
        for tag in tags:
            normalized = _normalize_tag(tag)
            if normalized and normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    def to_scene_text(self) -> str:
        parts = []
        if self.raw_text:
            parts.append(self.raw_text)
        if self.location:
            parts.append(f"location={self.location}")
        if self.activity:
            parts.append(f"activity={self.activity}")
        duration = self.computed_duration_minutes()
        if duration is not None:
            parts.append(f"duration_minutes={duration}")
        tod = self.time_of_day()
        if tod:
            parts.append(f"time_of_day={tod}")
        if self.signals:
            parts.append(f"signals={','.join(self.signals)}")
        if self.environment:
            env = ",".join(f"{key}={value}" for key, value in sorted(self.environment.items()))
            parts.append(f"environment={env}")
        return " | ".join(parts)

    def to_retrieval_query(self) -> str:
        return " ".join([self.raw_text, *self.context_tags()]).strip()

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "location": self.location,
            "activity": self.activity,
            "started_at": self.started_at,
            "observed_at": self.observed_at,
            "duration_minutes": self.duration_minutes,
            "computed_duration_minutes": self.computed_duration_minutes(),
            "time_of_day": self.time_of_day(),
            "signals": self.signals,
            "environment": self.environment,
            "context_tags": self.context_tags(),
        }

    def _number_from_environment(self, key: str) -> float | None:
        value = self.environment.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _append_tag(self, tags: list[str], value: str | None) -> None:
        if value:
            tags.append(value)


def _normalize_tag(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
