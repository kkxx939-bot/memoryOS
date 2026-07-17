"""Helpers for explicit canonical memory commands."""

from __future__ import annotations

import math
from typing import Any

from memoryos.memory.canonical import EvidenceRef, ModalForce, bind_field_evidence
from memoryos.memory.schema import MemoryType, MemoryTypeRegistry
from memoryos.operations.model.context_diff import ContextDiff


def _require_committed_diff(diff: ContextDiff, expected_operation_ids: set[str]) -> None:
    committed = {operation.operation_id for operation in diff.operations}
    pending = {operation.operation_id for operation in diff.pending_operations}
    rejected = {operation.operation_id for operation in diff.rejected_operations}
    if (
        not expected_operation_ids
        or expected_operation_ids - committed
        or expected_operation_ids & (pending | rejected)
    ):
        raise RuntimeError("forget operation was not fully committed")


def _explicit_field_evidence(
    identity_fields: Any,
    value_fields: Any,
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, tuple[EvidenceRef, ...]]:
    """Declare evidence for the SDK's own fully materialized remember/forget event."""

    bindings = {
        **{f"identity.{key}": evidence_refs for key in identity_fields},
        **{f"value.{key}": evidence_refs for key in value_fields},
        "semantic.speech_act": evidence_refs,
        "semantic.commitment": evidence_refs,
        "semantic.temporal_scope": evidence_refs,
        "semantic.relation_to_existing": evidence_refs,
        "semantic.utterance_mode": evidence_refs,
        "semantic.attribution": evidence_refs,
        "semantic.durability": evidence_refs,
        "semantic.modal_force": evidence_refs,
        "semantic.atomicity": evidence_refs,
        "transition": evidence_refs,
    }
    return bind_field_evidence(
        identity_fields,
        value_fields,
        evidence_refs,
        bindings=bindings,
        semantic_contract_version="v3",
    )


def _normalize_explicit_memory_type(memory_type: str) -> str:
    aliases = {"user_profile": MemoryType.PROFILE.value, "user_preference": MemoryType.PREFERENCE.value}
    return aliases.get(memory_type, memory_type)


def _explicit_rule_modal_force(raw: str, *, has_condition: bool) -> ModalForce:
    normalized = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "REQUIRED": ModalForce.REQUIRE,
        "FORBIDDEN": ModalForce.FORBID,
        "ALLOWED": ModalForce.ALLOW,
        "PREFERRED": ModalForce.PREFER,
        "DISCOURAGED": ModalForce.DISCOURAGE,
    }
    try:
        force = aliases.get(normalized)
        if force is None:
            force = ModalForce(normalized)
    except ValueError as exc:
        allowed = ", ".join(
            item.value
            for item in (
                ModalForce.REQUIRE,
                ModalForce.FORBID,
                ModalForce.ALLOW,
                ModalForce.PREFER,
                ModalForce.DISCOURAGE,
                ModalForce.CONDITIONAL_REQUIRE,
                ModalForce.CONDITIONAL_FORBID,
            )
        )
        raise ValueError(f"project_rule requires constraint_polarity in {{{allowed}}}") from exc
    if has_condition and force == ModalForce.REQUIRE:
        return ModalForce.CONDITIONAL_REQUIRE
    if has_condition and force == ModalForce.FORBID:
        return ModalForce.CONDITIONAL_FORBID
    if has_condition and force not in {ModalForce.CONDITIONAL_REQUIRE, ModalForce.CONDITIONAL_FORBID}:
        raise ValueError("project_rule condition or exception requires REQUIRE or FORBID polarity")
    if not has_condition and force in {ModalForce.CONDITIONAL_REQUIRE, ModalForce.CONDITIONAL_FORBID}:
        raise ValueError("conditional project_rule requires condition or exception")
    return force


def _explicit_retrieval_views(memory_type: str, *, user_id: str, project_id: str) -> list[str]:
    user_views = {
        MemoryType.PROFILE.value: f"user:{user_id}:profile",
        MemoryType.PREFERENCE.value: f"user:{user_id}:preferences",
    }
    if memory_type in user_views:
        return [user_views[memory_type]]
    project_suffix = {
        MemoryType.PROJECT_RULE.value: "rules",
        MemoryType.PROJECT_DECISION.value: "decisions",
        MemoryType.AGENT_EXPERIENCE.value: "agent_experience",
        MemoryType.ENTITY.value: "knowledge",
        MemoryType.EVENT.value: "knowledge",
    }.get(memory_type, "knowledge")
    return [f"project:{project_id}:{project_suffix}"] if project_id else [f"user:{user_id}:profile"]


def _explicit_identity_fields(
    memory_type: str,
    *,
    title: str,
    user_id: str,
    project_id: str,
    event_id: str,
    explicit_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact Identity V2 slot schema for an explicit command."""

    schema = MemoryTypeRegistry().get(MemoryType(memory_type))
    expected = tuple(schema.slot_identity_fields)
    supplied = dict(explicit_fields or {})
    topic = title.strip()
    if not supplied:
        if not topic:
            raise ValueError(
                f"explicit remember requires identity_fields {expected}; title is only a compatibility identity input"
            )
        generic_topics = {
            "memory",
            "profile",
            "user profile",
            "preference",
            "user preference",
            "project rule",
            "rule",
            "project decision",
            "decision",
            "agent experience",
            "entity",
            "event",
            "记忆",
            "个人资料",
            "偏好",
            "项目规则",
            "规则",
            "项目决策",
            "决策",
        }
        normalized_topic = " ".join(topic.casefold().replace("_", " ").replace("-", " ").split())
        if normalized_topic in generic_topics:
            raise ValueError(
                "explicit remember title is too generic for stable identity; provide type-specific identity_fields"
            )
        compatibility: dict[str, dict[str, Any]] = {
            MemoryType.PROFILE.value: {"attribute_key": topic},
            MemoryType.PREFERENCE.value: {"subject": user_id, "dimension": topic},
            MemoryType.PROJECT_RULE.value: {"rule_topic": topic},
            MemoryType.PROJECT_DECISION.value: {"decision_topic": topic},
            MemoryType.EVENT.value: {"event_key": topic},
        }
        supplied = compatibility.get(memory_type, {})
        if not supplied:
            raise ValueError(
                f"{memory_type} requires explicit identity_fields {expected}; title cannot safely infer them"
            )
    unknown = set(supplied) - set(expected)
    missing = {
        field_name
        for field_name in expected
        if supplied.get(field_name) is None
        or isinstance(supplied.get(field_name), str)
        and not str(supplied[field_name]).strip()
    }
    if unknown or missing:
        details = [
            *(f"missing:{item}" for item in sorted(missing)),
            *(f"unknown:{item}" for item in sorted(unknown)),
        ]
        raise ValueError(f"explicit remember identity_fields mismatch: {','.join(details)}")
    result: dict[str, Any] = {}
    for field_name in expected:
        value = supplied[field_name]
        if isinstance(value, str):
            value = value.strip()
        if isinstance(value, dict | list | tuple | set) or isinstance(value, bool):
            raise ValueError(f"identity field {field_name} must be a stable scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"identity field {field_name} must be finite")
        result[field_name] = value
    return result



__all__ = [
    "_explicit_field_evidence",
    "_explicit_identity_fields",
    "_explicit_retrieval_views",
    "_explicit_rule_modal_force",
    "_normalize_explicit_memory_type",
    "_require_committed_diff",
]
