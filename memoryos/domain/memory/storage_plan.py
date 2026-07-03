from __future__ import annotations

from dataclasses import dataclass

from memoryos.application.memory.schema import MEMORY_TYPE_DESCRIPTIONS
from memoryos.application.memory.weights import default_base_weight, default_temporal_scope
from memoryos.domain.memory.memory_item import TYPE_DIR
from memoryos.domain.memory.update_policy import update_policy


@dataclass(frozen=True)
class MemoryStoragePlan:
    memory_type: str
    directory: str
    description: str
    fact_role: str
    update_mode: str
    update_policy: str
    default_temporal_scope: str
    default_base_weight: float
    prediction_role: str

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type,
            "directory": self.directory,
            "description": self.description,
            "fact_role": self.fact_role,
            "update_mode": self.update_mode,
            "update_policy": self.update_policy,
            "default_temporal_scope": self.default_temporal_scope,
            "default_base_weight": self.default_base_weight,
            "prediction_role": self.prediction_role,
        }


FACT_ROLES = {
    "profile": "stable_user_model",
    "preference": "stable_or_medium_term_preference",
    "habit": "behavior_pattern",
    "trigger": "context_to_behavior_signal",
    "intervention": "agent_action_history",
    "feedback": "prediction_or_intervention_feedback",
    "policy": "permission_and_safety_boundary",
    "event": "auditable_episode_fact",
    "case": "reusable_context_action_outcome",
}

UPDATE_MODES = {
    "profile": "replace_or_patch_single_file",
    "preference": "topic_file_patch_or_split",
    "habit": "rolling_pattern_update",
    "trigger": "rolling_pattern_update",
    "intervention": "append_or_aggregate_by_action",
    "feedback": "append_feedback_then_aggregate",
    "policy": "replace_or_patch_strict",
    "event": "append_only",
    "case": "replace_or_version",
}

PREDICTION_ROLES = {
    "profile": "background_prior",
    "preference": "action_default_prior",
    "habit": "candidate_behavior_prior",
    "trigger": "candidate_activation_signal",
    "intervention": "what_agent_tried_before",
    "feedback": "reward_signal_for_ranker",
    "policy": "hard_or_soft_action_constraint",
    "event": "recent_evidence_low_prior",
    "case": "behavior_pattern_case_evidence",
}


def memory_storage_plan(memory_type: str) -> MemoryStoragePlan:
    if memory_type not in TYPE_DIR:
        known = ", ".join(sorted(TYPE_DIR))
        raise ValueError(f"Unknown memory type: {memory_type}. Known types: {known}")
    return MemoryStoragePlan(
        memory_type=memory_type,
        directory=TYPE_DIR[memory_type],
        description=MEMORY_TYPE_DESCRIPTIONS[memory_type],
        fact_role=FACT_ROLES[memory_type],
        update_mode=UPDATE_MODES[memory_type],
        update_policy=update_policy(memory_type).operation_mode,
        default_temporal_scope=default_temporal_scope(memory_type),
        default_base_weight=default_base_weight(memory_type),
        prediction_role=PREDICTION_ROLES[memory_type],
    )


def all_memory_storage_plans() -> list[MemoryStoragePlan]:
    return [memory_storage_plan(memory_type) for memory_type in sorted(TYPE_DIR)]
