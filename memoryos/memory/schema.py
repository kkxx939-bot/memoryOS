from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MEMORY_SCHEMA_VERSION = "memory_schema_v1"


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
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


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
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

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
