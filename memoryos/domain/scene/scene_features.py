from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.domain.scene.observation import ObservationContext


@dataclass(frozen=True)
class SceneFeatures:
    raw_text: str = ""
    location: str = ""
    activity: str = ""
    time_bucket: str = ""
    duration_minutes: int | None = None
    thermal_level: str = ""
    signals: list[str] = field(default_factory=list)
    environment: dict = field(default_factory=dict)

    @classmethod
    def from_observation(cls, observation: ObservationContext) -> SceneFeatures:
        payload = observation.to_dict()
        return cls(
            raw_text=str(payload.get("raw_text") or ""),
            location=str(payload.get("location") or ""),
            activity=str(payload.get("activity") or ""),
            time_bucket=str(payload.get("time_of_day") or ""),
            duration_minutes=payload.get("computed_duration_minutes"),
            thermal_level=str(payload.get("thermal_level") or ""),
            signals=[str(signal) for signal in payload.get("signals", [])],
            environment=payload.get("environment", {}) if isinstance(payload.get("environment"), dict) else {},
        )

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "location": self.location,
            "activity": self.activity,
            "time_bucket": self.time_bucket,
            "duration_minutes": self.duration_minutes,
            "thermal_level": self.thermal_level,
            "signals": self.signals,
            "environment": self.environment,
        }
