from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class LifecycleConfig:
    half_life_days: float = 14.0
    hot_threshold: float = 0.45
    cold_threshold: float = 0.12


def parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hotness_score(
    active_count: int,
    updated_at: str | None,
    now: datetime | None = None,
    config: LifecycleConfig = LifecycleConfig(),
) -> float:
    now = now or datetime.now(timezone.utc)
    updated = parse_datetime(updated_at)
    age_days = max((now - updated).total_seconds() / 86400, 0.0)
    usage_signal = 0.2 + 0.8 * (1.0 - math.exp(-max(active_count, 0) / 3.0))
    decay = math.exp(-math.log(2) * age_days / config.half_life_days)
    return round(max(0.0, min(1.0, usage_signal * decay)), 6)


def classify_lifecycle(score: float, config: LifecycleConfig = LifecycleConfig()) -> str:
    if score >= config.hot_threshold:
        return "hot"
    if score <= config.cold_threshold:
        return "cold"
    return "warm"
