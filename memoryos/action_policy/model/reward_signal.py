from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSignal:
    reward: float
    signal_type: str = "implicit_positive"
    evidence_uri: str = ""


@dataclass(frozen=True)
class PenaltySignal:
    penalty: float
    signal_type: str = "implicit_negative"
    evidence_uri: str = ""
    explicit_rule: str = ""
