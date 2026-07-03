from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryUpdatePolicy:
    memory_type: str
    operation_mode: str
    model_update_allowed: bool
    append_only: bool
    single_file: bool
    evidence_first: bool
    split_threshold_items: int | None
    description: str

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type,
            "operation_mode": self.operation_mode,
            "model_update_allowed": self.model_update_allowed,
            "append_only": self.append_only,
            "single_file": self.single_file,
            "evidence_first": self.evidence_first,
            "split_threshold_items": self.split_threshold_items,
            "description": self.description,
        }


UPDATE_POLICIES = {
    "profile": MemoryUpdatePolicy(
        memory_type="profile",
        operation_mode="patch_or_replace",
        model_update_allowed=True,
        append_only=False,
        single_file=True,
        evidence_first=False,
        split_threshold_items=None,
        description="Keep one compact user profile; rewrite/patch instead of endlessly appending.",
    ),
    "preference": MemoryUpdatePolicy(
        memory_type="preference",
        operation_mode="topic_patch_or_split",
        model_update_allowed=True,
        append_only=False,
        single_file=False,
        evidence_first=False,
        split_threshold_items=8,
        description="Keep preferences topic-scoped; patch existing topic or split when it grows.",
    ),
    "habit": MemoryUpdatePolicy(
        memory_type="habit",
        operation_mode="aggregate_from_evidence",
        model_update_allowed=False,
        append_only=False,
        single_file=False,
        evidence_first=True,
        split_threshold_items=8,
        description="Do not create strong habits from one observation; aggregate repeated events first.",
    ),
    "trigger": MemoryUpdatePolicy(
        memory_type="trigger",
        operation_mode="aggregate_from_evidence",
        model_update_allowed=False,
        append_only=False,
        single_file=False,
        evidence_first=True,
        split_threshold_items=8,
        description="Create triggers from repeated context-action evidence, usually rolling 7d.",
    ),
    "intervention": MemoryUpdatePolicy(
        memory_type="intervention",
        operation_mode="append_then_aggregate",
        model_update_allowed=True,
        append_only=False,
        single_file=False,
        evidence_first=True,
        split_threshold_items=12,
        description="Record actions taken, then aggregate by action/outcome.",
    ),
    "feedback": MemoryUpdatePolicy(
        memory_type="feedback",
        operation_mode="append_then_aggregate",
        model_update_allowed=True,
        append_only=False,
        single_file=False,
        evidence_first=True,
        split_threshold_items=12,
        description="Record user feedback/reward as learning signal; aggregate later.",
    ),
    "policy": MemoryUpdatePolicy(
        memory_type="policy",
        operation_mode="strict_patch",
        model_update_allowed=False,
        append_only=False,
        single_file=False,
        evidence_first=False,
        split_threshold_items=None,
        description="Policies are permission/safety boundaries; require explicit user intent.",
    ),
    "event": MemoryUpdatePolicy(
        memory_type="event",
        operation_mode="append_only",
        model_update_allowed=True,
        append_only=True,
        single_file=False,
        evidence_first=False,
        split_threshold_items=None,
        description="Events are auditable evidence and should not be rewritten.",
    ),
    "case": MemoryUpdatePolicy(
        memory_type="case",
        operation_mode="replace_or_version",
        model_update_allowed=True,
        append_only=False,
        single_file=False,
        evidence_first=True,
        split_threshold_items=None,
        description="Reusable context-action-outcome examples; replace or version when improved.",
    ),
}


def update_policy(memory_type: str) -> MemoryUpdatePolicy:
    if memory_type not in UPDATE_POLICIES:
        known = ", ".join(sorted(UPDATE_POLICIES))
        raise ValueError(f"Unknown memory type: {memory_type}. Known types: {known}")
    return UPDATE_POLICIES[memory_type]


def all_update_policies() -> list[MemoryUpdatePolicy]:
    return [UPDATE_POLICIES[memory_type] for memory_type in sorted(UPDATE_POLICIES)]


def normalize_operation_for_policy(action: str, memory_type: str, explicit_user_intent: bool = False) -> tuple[str, str | None]:
    policy = update_policy(memory_type)
    if action == "ignore":
        return action, None
    if policy.append_only and action == "update":
        return "add", f"{memory_type} is append-only; converted update to add"
    if not policy.model_update_allowed and not explicit_user_intent:
        if memory_type in {"habit", "trigger"}:
            return "add", f"{memory_type} requires evidence-first handling; store as event evidence before aggregation"
        return "ignore", f"{memory_type} requires explicit user intent"
    return action, None
