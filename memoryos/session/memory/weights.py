from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .lifecycle import parse_datetime


TEMPORAL_SCOPES = {"stable", "rolling_7d", "rolling_30d", "episodic", "seasonal"}

DEFAULT_BASE_WEIGHTS = {
    "profile": 0.95,
    "policy": 0.92,
    "preference": 0.88,
    "habit": 0.82,
    "trigger": 0.82,
    "feedback": 0.72,
    "intervention": 0.68,
    "case": 0.62,
    "event": 0.42,
}

DEFAULT_TEMPORAL_SCOPES = {
    "profile": "stable",
    "policy": "stable",
    "preference": "stable",
    "habit": "rolling_30d",
    "trigger": "rolling_7d",
    "feedback": "rolling_30d",
    "intervention": "rolling_30d",
    "case": "rolling_30d",
    "event": "episodic",
}

HALF_LIFE_DAYS = {
    "stable": 3650.0,
    "rolling_7d": 7.0,
    "rolling_30d": 30.0,
    "episodic": 3.0,
    "seasonal": 90.0,
}


@dataclass(frozen=True)
class MemoryWeight:
    base_weight: float
    temporal_weight: float
    evidence_weight: float
    effective_weight: float


def default_base_weight(memory_type: str) -> float:
    return DEFAULT_BASE_WEIGHTS.get(memory_type, 0.5)


def default_temporal_scope(memory_type: str) -> str:
    return DEFAULT_TEMPORAL_SCOPES.get(memory_type, "rolling_30d")


def score_memory_weight(memory: dict, now: datetime | None = None) -> MemoryWeight:
    now = now or datetime.now(timezone.utc)
    base_weight = float(memory.get("base_weight", default_base_weight(str(memory.get("type", "")))))
    temporal_scope = str(memory.get("temporal_scope", default_temporal_scope(str(memory.get("type", "")))))
    updated_at = str(memory.get("updated_at") or memory.get("created_at"))
    confidence = float(memory.get("confidence", 1.0))
    evidence_count = int(memory.get("evidence_count", 1) or 1)
    positive_count = int(memory.get("positive_count", evidence_count) or 0)
    negative_count = int(memory.get("negative_count", 0) or 0)
    temporal_weight = _temporal_weight(temporal_scope, updated_at, now)
    evidence_weight = _evidence_weight(evidence_count, positive_count, negative_count)
    effective_weight = base_weight * temporal_weight * evidence_weight * confidence
    return MemoryWeight(
        base_weight=round(_clamp(base_weight), 6),
        temporal_weight=round(_clamp(temporal_weight), 6),
        evidence_weight=round(_clamp(evidence_weight), 6),
        effective_weight=round(_clamp(effective_weight), 6),
    )


def _temporal_weight(temporal_scope: str, updated_at: str, now: datetime) -> float:
    if temporal_scope == "stable":
        return 1.0
    updated = parse_datetime(updated_at)
    age_days = max((now - updated).total_seconds() / 86400, 0.0)
    half_life = HALF_LIFE_DAYS.get(temporal_scope, 30.0)
    return math.exp(-math.log(2) * age_days / half_life)


def _evidence_weight(evidence_count: int, positive_count: int, negative_count: int) -> float:
    evidence_count = max(evidence_count, positive_count + negative_count, 1)
    support = 0.35 + min(1.0, math.log1p(evidence_count) / math.log1p(10)) * 0.65
    reliability = (positive_count + 1) / (positive_count + negative_count + 2)
    return support * reliability


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
