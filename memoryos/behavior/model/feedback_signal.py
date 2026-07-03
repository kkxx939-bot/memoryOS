from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackSignal:
    user_id: str
    episode_id: str
    signal_type: str
    reward: float
    predicted_action: str = ""
    actual_action: str = ""
    explicit_rule: str = ""

    def is_explicit_negative_rule(self) -> bool:
        return self.reward < 0 and bool(self.explicit_rule.strip())
