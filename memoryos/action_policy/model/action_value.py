from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionValue:
    q_value: float = 0.5
    reward_score: float = 0.0
    penalty_score: float = 0.0
    trials: int = 0
