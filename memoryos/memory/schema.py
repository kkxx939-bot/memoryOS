"""记忆系统里的数据结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MEMORY_SCHEMA_VERSION = "memory_schema_v2"
MEMORY_IDENTITY_SCHEMA_VERSION = "memory_identity_schema_v2"


def _validated_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a finite number between 0 and 1") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return confidence


class MemoryType(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    ENTITY = "entity"
    EVENT = "event"
    PROJECT_RULE = "project_rule"
    PROJECT_DECISION = "project_decision"
    AGENT_EXPERIENCE = "agent_experience"


class AdmissionDecision(str, Enum):
    ACCEPT = "accept"
    PENDING = "pending"
    REJECT = "reject"
    ARCHIVE_ONLY = "archive_only"
    PRIVATE_ONLY = "private_only"
    RESTRICTED = "restricted"


class FieldMergeMode(str, Enum):
    REPLACE = "replace"
    APPEND_UNIQUE = "append_unique"
    PATCH_TEXT = "patch_text"
    IMMUTABLE = "immutable"


COMMON_PROVENANCE_FIELDS = (
    "asserted_by",
    "author",
    "source_role",
    "source_adapter_id",
    "source_session_id",
    "evidence_source",
    "extractor_version",
    "model_id",
    "prompt_version",
)
COMMON_DISPLAY_FIELDS = (
    "title",
    "display_name",
    "display_text",
    "source_text",
    "summary",
    "details",
    "rationale",
    "reason",
    "aliases",
)
PROFILE_DISPLAY_FIELDS = COMMON_DISPLAY_FIELDS
COMMON_APPLICABILITY_FIELDS = ("scope", "project_id", "workspace_id", "applies_to", "visibility")
COMMON_CLAIM_QUALIFIER_FIELDS = (
    "environment",
    "device",
    "activity",
    "valid_time",
    "condition",
    "conditions",
    "exception",
    "exceptions",
    "applicability_qualifier",
)


@dataclass(frozen=True)
class MemoryTypeSchema:
    memory_type: MemoryType
    description: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    operation_mode: str = "add"
    default_retrieval_views: tuple[str, ...] = ()
    allow_user_source: bool = True
    allow_assistant_source: bool = True
    allow_tool_source: bool = False
    share_default: bool = True
    field_merge_rules: dict[str, FieldMergeMode] = field(default_factory=dict)
    identity_schema_version: str = MEMORY_IDENTITY_SCHEMA_VERSION
    slot_identity_fields: tuple[str, ...] = ()
    claim_identity_fields: tuple[str, ...] = ()
    provenance_fields: tuple[str, ...] = ()
    display_fields: tuple[str, ...] = ()
    applicability_fields: tuple[str, ...] = ()

    def claim_identity_keys(self, value_fields: dict[str, Any]) -> tuple[str, ...]:
        """Return all schema-declared semantic claim keys deterministically.

        ``*`` means every value field except fields explicitly classified as
        provenance, display metadata, or applicability.  This prevents new
        semantic qualifiers from being silently ignored by identity.
        """

        if (
            "canonical_value" in self.claim_identity_fields
            and (
                value_fields.get("canonical_value") is None
                or value_fields.get("canonical_value") == ""
            )
        ):
            # Applicability qualifiers refine a semantic value; they cannot be
            # the value. Without a canonical core, two unrelated claims in the
            # same environment/condition would collapse onto one identity.
            return ()
        if self.claim_identity_fields and "*" not in self.claim_identity_fields:
            return tuple(field_name for field_name in self.claim_identity_fields if field_name in value_fields)
        excluded = {*self.provenance_fields, *self.display_fields, *self.applicability_fields}
        return tuple(sorted(field_name for field_name in value_fields if field_name not in excluded))


class MemoryTypeRegistry:
    def __init__(self, schemas: list[MemoryTypeSchema] | None = None) -> None:
        self._schemas = {schema.memory_type: schema for schema in (schemas or self._builtin_schemas())}

    def get(self, memory_type: MemoryType | str) -> MemoryTypeSchema:
        return self._schemas[MemoryType(memory_type)]

    def schemas(self) -> list[MemoryTypeSchema]:
        return list(self._schemas.values())

    def by_value(self) -> dict[str, MemoryTypeSchema]:
        return {memory_type.value: schema for memory_type, schema in self._schemas.items()}

    def _builtin_schemas(self) -> list[MemoryTypeSchema]:
        return [
            MemoryTypeSchema(
                memory_type=MemoryType.PROFILE,
                description="Stable, high-level user profile facts; excludes temporary moods, tasks, and tool output.",
                required_fields=("summary",),
                optional_fields=("scope", "stability"),
                default_retrieval_views=("user:{user_id}:profile",),
                allow_assistant_source=False,
                field_merge_rules={
                    "summary": FieldMergeMode.PATCH_TEXT,
                    "identity": FieldMergeMode.IMMUTABLE,
                    "scope": FieldMergeMode.IMMUTABLE,
                    "stability": FieldMergeMode.REPLACE,
                    "evidence": FieldMergeMode.APPEND_UNIQUE,
                },
                slot_identity_fields=("attribute_key",),
                claim_identity_fields=("canonical_value", *COMMON_CLAIM_QUALIFIER_FIELDS),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=PROFILE_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
            MemoryTypeSchema(
                memory_type=MemoryType.PREFERENCE,
                description="Durable user preferences, work style, communication style, review style, and habits.",
                required_fields=("preference",),
                optional_fields=("scope", "project_id", "applies_to"),
                default_retrieval_views=("user:{user_id}:preferences",),
                allow_assistant_source=False,
                field_merge_rules={
                    "topic": FieldMergeMode.IMMUTABLE,
                    "subject": FieldMergeMode.IMMUTABLE,
                    "preference": FieldMergeMode.PATCH_TEXT,
                    "content": FieldMergeMode.PATCH_TEXT,
                    "evidence": FieldMergeMode.APPEND_UNIQUE,
                },
                slot_identity_fields=("subject", "dimension"),
                claim_identity_fields=(
                    "canonical_value",
                    "preference_value",
                    "value",
                    *COMMON_CLAIM_QUALIFIER_FIELDS,
                ),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=COMMON_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
            MemoryTypeSchema(
                memory_type=MemoryType.ENTITY,
                description="Stable entities such as projects, tools, products, organizations, people, devices, or concepts.",
                required_fields=("name", "entity_type"),
                optional_fields=("project_id", "aliases", "summary"),
                default_retrieval_views=("project:{project_id}:knowledge", "user:{user_id}:profile"),
                field_merge_rules={
                    "name": FieldMergeMode.IMMUTABLE,
                    "entity_type": FieldMergeMode.IMMUTABLE,
                    "type": FieldMergeMode.IMMUTABLE,
                    "aliases": FieldMergeMode.APPEND_UNIQUE,
                    "summary": FieldMergeMode.PATCH_TEXT,
                    "details": FieldMergeMode.PATCH_TEXT,
                },
                slot_identity_fields=("entity_type", "canonical_entity_id"),
                claim_identity_fields=("canonical_value", "name", *COMMON_CLAIM_QUALIFIER_FIELDS),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=COMMON_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
            MemoryTypeSchema(
                memory_type=MemoryType.EVENT,
                description="Atomic real-world or project events, decisions, progress, commitments, and outcomes.",
                required_fields=("event",),
                optional_fields=("project_id", "occurred_at", "outcome"),
                default_retrieval_views=("project:{project_id}:knowledge",),
                field_merge_rules={
                    "event_key": FieldMergeMode.IMMUTABLE,
                    "event": FieldMergeMode.PATCH_TEXT,
                    "date": FieldMergeMode.IMMUTABLE,
                    "occurred_at": FieldMergeMode.IMMUTABLE,
                    "details": FieldMergeMode.PATCH_TEXT,
                },
                slot_identity_fields=("event_key",),
                claim_identity_fields=(
                    "canonical_value",
                    "outcome",
                    "occurred_at",
                    *COMMON_CLAIM_QUALIFIER_FIELDS,
                ),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=COMMON_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
            MemoryTypeSchema(
                memory_type=MemoryType.PROJECT_RULE,
                description="Long-lived project constraints and rules that must be followed in future work.",
                required_fields=("rule", "project_id"),
                optional_fields=("scope", "rationale"),
                default_retrieval_views=("project:{project_id}:rules",),
                field_merge_rules={
                    "rule_key": FieldMergeMode.IMMUTABLE,
                    "rule": FieldMergeMode.PATCH_TEXT,
                    "content": FieldMergeMode.PATCH_TEXT,
                    "constraints": FieldMergeMode.APPEND_UNIQUE,
                    "project_id": FieldMergeMode.IMMUTABLE,
                },
                slot_identity_fields=("rule_topic",),
                claim_identity_fields=("canonical_value", "subject", *COMMON_CLAIM_QUALIFIER_FIELDS),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=COMMON_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
            MemoryTypeSchema(
                memory_type=MemoryType.PROJECT_DECISION,
                description="Project architecture decisions, stage tradeoffs, and explicit adopted/deferred/rejected choices.",
                required_fields=("decision", "project_id"),
                optional_fields=("rationale", "alternatives", "decided_at"),
                default_retrieval_views=("project:{project_id}:decisions",),
                field_merge_rules={
                    "decision_key": FieldMergeMode.IMMUTABLE,
                    "decision": FieldMergeMode.PATCH_TEXT,
                    "content": FieldMergeMode.PATCH_TEXT,
                    "status": FieldMergeMode.REPLACE,
                    "project_id": FieldMergeMode.IMMUTABLE,
                },
                slot_identity_fields=("decision_topic",),
                claim_identity_fields=("canonical_value", *COMMON_CLAIM_QUALIFIER_FIELDS),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=COMMON_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
            MemoryTypeSchema(
                memory_type=MemoryType.AGENT_EXPERIENCE,
                description="Reusable experience distilled after Codex, Claude, Cursor, or OpenClaw execution; not raw logs.",
                required_fields=("situation", "approach", "outcome"),
                optional_fields=("project_id", "adapter_id", "tooling"),
                default_retrieval_views=("project:{project_id}:agent_experience",),
                allow_user_source=False,
                field_merge_rules={
                    "situation_key": FieldMergeMode.IMMUTABLE,
                    "situation": FieldMergeMode.IMMUTABLE,
                    "approach": FieldMergeMode.PATCH_TEXT,
                    "reflect": FieldMergeMode.PATCH_TEXT,
                    "outcome": FieldMergeMode.PATCH_TEXT,
                    "evidence": FieldMergeMode.APPEND_UNIQUE,
                },
                slot_identity_fields=("task_pattern", "environment_signature"),
                claim_identity_fields=("canonical_value", "outcome", *COMMON_CLAIM_QUALIFIER_FIELDS),
                provenance_fields=COMMON_PROVENANCE_FIELDS,
                display_fields=COMMON_DISPLAY_FIELDS,
                applicability_fields=COMMON_APPLICABILITY_FIELDS,
            ),
        ]

    def list(self) -> list[MemoryTypeSchema]:
        return self.schemas()


@dataclass
class MemoryCandidateDraft:
    memory_type: MemoryType
    title: str
    content: str
    fields: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    source_role: str = "unknown"
    source_adapter_id: str = ""
    source_session_id: str = ""
    source_message_ids: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    suggested_retrieval_views: list[str] = field(default_factory=list)
    merge_key: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
        self.confidence = _validated_confidence(self.confidence)


@dataclass
class MemoryAdmissionResult:
    decision: AdmissionDecision
    reason: str
    confidence: float
    memory_type: MemoryType
    retrieval_views: list[str] = field(default_factory=list)
    operation_mode: str = "add"
    merge_key: str = ""
    private: bool = False
    restricted: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.decision, str):
            self.decision = AdmissionDecision(self.decision)
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
        self.confidence = _validated_confidence(self.confidence)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "operation_mode": self.operation_mode,
            "private": self.private,
            "restricted": self.restricted,
        }


@dataclass
class MemoryOperationGroupItem:
    candidate: MemoryCandidateDraft
    admission: MemoryAdmissionResult


@dataclass
class MemoryOperationGroup:
    accepted: list[MemoryOperationGroupItem] = field(default_factory=list)
    pending: list[MemoryOperationGroupItem] = field(default_factory=list)
    rejected: list[MemoryOperationGroupItem] = field(default_factory=list)
    archive_only: list[MemoryOperationGroupItem] = field(default_factory=list)
    private_only: list[MemoryOperationGroupItem] = field(default_factory=list)
    restricted: list[MemoryOperationGroupItem] = field(default_factory=list)

    def add(self, candidate: MemoryCandidateDraft, admission: MemoryAdmissionResult) -> None:
        item = MemoryOperationGroupItem(candidate=candidate, admission=admission)
        if admission.decision == AdmissionDecision.ACCEPT:
            self.accepted.append(item)
        elif admission.decision == AdmissionDecision.PENDING:
            self.pending.append(item)
        elif admission.decision == AdmissionDecision.ARCHIVE_ONLY:
            self.archive_only.append(item)
        elif admission.decision == AdmissionDecision.PRIVATE_ONLY:
            self.private_only.append(item)
        elif admission.decision == AdmissionDecision.RESTRICTED:
            self.restricted.append(item)
        else:
            self.rejected.append(item)

    def summary(self) -> dict[str, int]:
        return {
            "accepted": len(self.accepted),
            "pending": len(self.pending),
            "rejected": len(self.rejected),
            "archive_only": len(self.archive_only),
            "private_only": len(self.private_only),
            "restricted": len(self.restricted),
        }
