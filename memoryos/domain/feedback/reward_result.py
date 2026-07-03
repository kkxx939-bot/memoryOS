from __future__ import annotations

from dataclasses import dataclass

from memoryos.domain.actions.action_schema import action_need, canonical_action


REWARD_MODEL_VERSION = "reward_v1"


@dataclass(frozen=True)
class RewardBreakdown:
    behavior_reward: float
    intervention_reward: float
    need_match: bool
    action_match: bool
    param_match: bool
    model_version: str = REWARD_MODEL_VERSION

    def to_dict(self) -> dict:
        return {
            "behavior_reward": self.behavior_reward,
            "intervention_reward": self.intervention_reward,
            "need_match": self.need_match,
            "action_match": self.action_match,
            "param_match": self.param_match,
            "model_version": self.model_version,
        }


def compute_rewards(
    predicted_action: str,
    actual_action: str | None,
    user_reward: float,
    intervention_action: str,
    intervention_result: str = "",
    predicted_params: dict | None = None,
    actual_params: dict | None = None,
) -> RewardBreakdown:
    predicted = canonical_action(predicted_action)
    actual = canonical_action(actual_action or "")
    bounded_user_reward = max(-1.0, min(1.0, float(user_reward)))
    action_match = bool(actual and predicted == actual)
    need_match = bool(actual and action_need(predicted) == action_need(actual))
    param_match = _params_match(predicted_params or {}, actual_params or {})

    if action_match and param_match:
        behavior_reward = 1.0
    elif action_match:
        behavior_reward = 0.6
    elif need_match:
        behavior_reward = 0.3
    elif actual:
        behavior_reward = -0.7
    else:
        behavior_reward = 0.0
    behavior_reward = round(max(-1.0, min(1.0, behavior_reward * 0.7 + bounded_user_reward * 0.3)), 6)

    result = str(intervention_result or "").lower()
    if any(token in result for token in ("accepted", "accept", "可以", "同意", "好")):
        intervention_reward = max(0.3, bounded_user_reward)
    elif any(token in result for token in ("rejected", "reject", "不用", "不要", "拒绝")):
        intervention_reward = min(-0.3, bounded_user_reward)
    elif intervention_action in {"do_nothing", ""}:
        intervention_reward = 0.0 if bounded_user_reward >= 0 else bounded_user_reward
    else:
        intervention_reward = bounded_user_reward

    return RewardBreakdown(
        behavior_reward=round(max(-1.0, min(1.0, behavior_reward)), 6),
        intervention_reward=round(max(-1.0, min(1.0, intervention_reward)), 6),
        need_match=need_match,
        action_match=action_match,
        param_match=param_match,
    )


def _params_match(predicted: dict, actual: dict) -> bool:
    if not predicted or not actual:
        return True
    for key, value in predicted.items():
        if key in actual and str(actual[key]) != str(value):
            return False
    return True
