"""记忆合并逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.schema import (
    AdmissionDecision,
    FieldMergeMode,
    MemoryAdmissionResult,
    MemoryCandidateDraft,
    MemoryType,
    MemoryTypeRegistry,
)
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


@dataclass(frozen=True)
class ExistingMemory:
    obj: ContextObject
    content: str


@dataclass(frozen=True)
class MemoryMergeDecision:
    action: str
    reason: str
    existing_uri: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    content: str = ""


class MemoryMergePlanner:
    def __init__(
        self,
        source_store: SourceStore | None = None,
        registry: MemoryTypeRegistry | None = None,
    ) -> None:
        self.source_store = source_store
        self.registry = registry or MemoryTypeRegistry()

    def plan(
        self,
        candidate: MemoryCandidateDraft,
        admission: MemoryAdmissionResult,
    ) -> MemoryMergeDecision:
        if admission.decision != AdmissionDecision.ACCEPT:
            return MemoryMergeDecision("ADD", "not_accepted_or_candidate", fields=dict(candidate.fields), content=candidate.content)
        existing = self.find_existing(candidate, admission)
        if existing is None:
            return MemoryMergeDecision("ADD", "no_existing_merge_key", fields=dict(candidate.fields), content=candidate.content)
        if self._is_conflicting_project_rule(candidate, existing):
            return MemoryMergeDecision(
                "SUPERSEDE",
                "conflicting_project_rule_requires_supersede",
                existing_uri=existing.obj.uri,
                fields=dict(candidate.fields),
                content=candidate.content,
            )
        merged_fields = self.merge_fields(candidate.memory_type, dict(existing.obj.metadata.get("fields", {}) or {}), candidate.fields)
        merged_content = self._merge_content(existing.content, candidate.content)
        return MemoryMergeDecision("MERGE", "existing_merge_key_matched", existing.obj.uri, merged_fields, merged_content)

    def apply(self, operation: ContextOperation, decision: MemoryMergeDecision) -> ContextOperation:
        payload = dict(operation.payload)
        payload.update(
            {
                "merge_decision": decision.action,
                "existing_uri": decision.existing_uri,
                "merge_reason": decision.reason,
            }
        )
        object_payload = payload.get("context_object")
        if isinstance(object_payload, dict):
            metadata = dict(object_payload.get("metadata", {}) or {})
            metadata.update(
                {
                    "merge_decision": decision.action,
                    "existing_uri": decision.existing_uri,
                    "merge_reason": decision.reason,
                    "fields": dict(decision.fields),
                }
            )
            object_payload["metadata"] = metadata
            if decision.existing_uri and decision.action in {"UPDATE", "MERGE"}:
                object_payload["uri"] = decision.existing_uri
                operation.target_uri = decision.existing_uri
            payload["context_object"] = object_payload
        if decision.content:
            payload["content"] = decision.content
        if decision.action == "SUPERSEDE":
            operation.action = OperationAction.SUPERSEDE
            operation.target_uri = decision.existing_uri
        elif decision.action in {"UPDATE", "MERGE"}:
            operation.action = OperationAction.UPDATE
            operation.target_uri = decision.existing_uri
        else:
            operation.action = OperationAction.ADD
        operation.payload = payload
        return operation

    def find_existing(
        self,
        candidate: MemoryCandidateDraft,
        admission: MemoryAdmissionResult,
    ) -> ExistingMemory | None:
        merge_key = admission.merge_key or candidate.merge_key
        if not merge_key or self.source_store is None:
            return None
        for obj in self.source_store.list_objects():
            metadata = dict(obj.metadata or {})
            if obj.context_type != ContextType.MEMORY:
                continue
            if obj.lifecycle_state != LifecycleState.ACTIVE:
                continue
            if metadata.get("merge_key") != merge_key:
                continue
            if metadata.get("memory_type") != candidate.memory_type.value:
                continue
            admission_metadata = dict(metadata.get("admission", {}) or {})
            if admission_metadata.get("decision") != AdmissionDecision.ACCEPT.value:
                continue
            try:
                content = self.source_store.read_content(obj.layers.l2_uri or obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = ""
            return ExistingMemory(obj=obj, content=content)
        return None

    def merge_fields(self, memory_type: MemoryType | str, existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        schema = self.registry.get(memory_type)
        merged = dict(existing)
        for field_name, value in incoming.items():
            mode = schema.field_merge_rules.get(field_name, FieldMergeMode.PATCH_TEXT)
            current = merged.get(field_name)
            merged[field_name] = self._merge_value(mode, current, value)
        return merged

    def _merge_value(self, mode: FieldMergeMode, current: Any, incoming: Any) -> Any:
        if mode == FieldMergeMode.IMMUTABLE:
            return incoming if self._empty(current) else current
        if mode == FieldMergeMode.REPLACE:
            return current if self._empty(incoming) else incoming
        if mode == FieldMergeMode.APPEND_UNIQUE:
            return self._append_unique(current, incoming)
        return self._merge_text(current, incoming)

    def _append_unique(self, current: Any, incoming: Any) -> list[Any]:
        values = []
        for item in self._as_list(current) + self._as_list(incoming):
            if item not in values and item not in {None, ""}:
                values.append(item)
        return values

    def _merge_text(self, current: Any, incoming: Any) -> str:
        return self._merge_content(str(current or ""), str(incoming or ""))

    def _merge_content(self, current: str, incoming: str) -> str:
        current = current.strip()
        incoming = incoming.strip()
        if not current:
            return incoming
        if not incoming or incoming == current or incoming in current:
            return current
        if current in incoming:
            return incoming
        return f"{current}\n{incoming}"

    def _as_list(self, value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple | set):
            return list(value)
        return [value]

    def _empty(self, value: Any) -> bool:
        return value is None or value == ""

    def _is_conflicting_project_rule(self, candidate: MemoryCandidateDraft, existing: ExistingMemory) -> bool:
        if candidate.memory_type != MemoryType.PROJECT_RULE:
            return False
        current = existing.content.strip()
        incoming = candidate.content.strip()
        return bool(current and incoming and current != incoming and incoming not in current and current not in incoming)
