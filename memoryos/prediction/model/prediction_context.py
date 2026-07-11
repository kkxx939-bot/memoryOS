"""预测模块里的预测上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.prediction.model.action_context import ActionContext


@dataclass(frozen=True)
class PredictionContext:
    action_context: ActionContext
    similar_behavior: dict = field(default_factory=dict)
    policy_context: dict = field(default_factory=dict)
