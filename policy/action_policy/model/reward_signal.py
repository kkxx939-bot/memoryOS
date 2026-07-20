"""ActionPolicy 的正负反馈信号。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSignal:
    reward: float
    signal_type: str = "implicit_positive"
    evidence_uri: str = ""

    def __post_init__(self) -> None:
        value = float(self.reward)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("reward must be a finite number in (0, 1]")
        if not str(self.signal_type).strip():
            raise ValueError("signal_type is required")
        object.__setattr__(self, "reward", value)

    @classmethod
    def from_payload(cls, payload: dict) -> RewardSignal:
        value = payload.get("reward", payload.get("reward_value", payload.get("reward_delta", 0.0)))
        return cls(
            reward=float(value or 0.0),
            signal_type=str(payload.get("signal_type", payload.get("feedback_type", "implicit_positive"))),
            evidence_uri=str(payload.get("evidence_uri", payload.get("source_uri", ""))),
        )


@dataclass(frozen=True)
class PenaltySignal:
    penalty: float
    signal_type: str = "implicit_negative"
    evidence_uri: str = ""
    explicit_rule: str = ""

    def __post_init__(self) -> None:
        value = float(self.penalty)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("penalty must be a finite number in (0, 1]")
        if not str(self.signal_type).strip():
            raise ValueError("signal_type is required")
        object.__setattr__(self, "penalty", value)

    @classmethod
    def from_payload(cls, payload: dict) -> PenaltySignal:
        value = payload.get("penalty", payload.get("penalty_value", payload.get("penalty_delta", 0.0)))
        if not value and float(payload.get("reward_value", 0.0) or 0.0) < 0:
            value = abs(float(payload["reward_value"]))
        return cls(
            penalty=float(value or 0.0),
            signal_type=str(payload.get("signal_type", payload.get("feedback_type", "implicit_negative"))),
            evidence_uri=str(payload.get("evidence_uri", payload.get("source_uri", ""))),
            explicit_rule=str(payload.get("explicit_rule", "")),
        )
