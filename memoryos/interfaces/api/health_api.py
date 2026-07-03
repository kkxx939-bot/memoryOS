from __future__ import annotations

from memoryos.domain.actions.action_schema import ACTION_SCHEMA_VERSION
from memoryos.domain.feedback.reward_result import REWARD_MODEL_VERSION


def health() -> dict:
    return {
        "status": "ok",
        "service": "memoryos",
        "versions": {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "reward_model_version": REWARD_MODEL_VERSION,
        },
    }
