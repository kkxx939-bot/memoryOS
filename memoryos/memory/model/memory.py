"""记忆系统里的记忆。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.time import utc_now


class MemoryKind(str, Enum):
    EXPLICIT = "explicit_memory"
    ANCHOR = "anchor_memory"
    CANDIDATE = "memory_candidate"
    CONFIRMED_INFERRED = "confirmed_inferred_memory"
    POLICY = "policy_memory"


@dataclass
class Memory:
    uri: str
    user_id: str
    title: str
    content: str
    kind: MemoryKind
    confidence: float = 1.0
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    supporting_behavior_uris: list[str] = field(default_factory=list)
    constrains_policy_uris: list[str] = field(default_factory=list)
    memory_type: str | None = None
    retrieval_views: list[str] = field(default_factory=list)
    admission: dict[str, Any] | str | None = None
    merge_key: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    memory_schema_version: str = "memory_schema_v1"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = MemoryKind(self.kind)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_context_object(self) -> ContextObject:
        metadata: dict[str, Any] = {
            "memory_kind": self.kind.value,
            "confidence": self.confidence,
            "tags": self.tags,
            "supporting_behavior_uris": self.supporting_behavior_uris,
            "constrains_policy_uris": self.constrains_policy_uris,
        }
        if self.memory_type:
            metadata["memory_type"] = self.memory_type
        if self.retrieval_views:
            metadata["retrieval_views"] = self.retrieval_views
        if self.admission:
            metadata["admission"] = self.admission
        if self.merge_key:
            metadata["merge_key"] = self.merge_key
        if self.fields:
            metadata["fields"] = self.fields
        if self.source:
            metadata["source"] = self.source
            if self.source.get("adapter_id"):
                metadata["source_adapter_id"] = self.source["adapter_id"]
            if self.source.get("session_id"):
                metadata["source_session_id"] = self.source["session_id"]
            if self.source.get("roles"):
                metadata["source_roles"] = self.source["roles"]
        if self.memory_schema_version:
            metadata["schema_version"] = self.memory_schema_version
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=self.title,
            owner_user_id=self.user_id,
            lifecycle_state=LifecycleState.ACTIVE if self.status == "active" else LifecycleState(self.status),
            semantic_hotness=self.confidence,
            behavior_support_hotness=min(1.0, len(self.supporting_behavior_uris) * 0.15),
            metadata=metadata,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


@dataclass
class MemoryAnchor(Memory):
    anchor_key: str = ""

    def __init__(
        self,
        uri: str,
        user_id: str,
        title: str,
        content: str,
        anchor_key: str,
        confidence: float = 0.65,
        supporting_behavior_uris: list[str] | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            user_id=user_id,
            title=title,
            content=content,
            kind=MemoryKind.ANCHOR,
            confidence=confidence,
            tags=["anchor", anchor_key],
            supporting_behavior_uris=supporting_behavior_uris or [],
        )
        self.anchor_key = anchor_key


@dataclass
class MemoryCandidate(Memory):
    confirmation_state: str = "pending"

    def to_context_object(self) -> ContextObject:
        obj = super().to_context_object()
        admission = dict(obj.metadata.get("admission", {}) or {})
        obj.metadata = {
            **obj.metadata,
            "admission": admission,
            "candidate_reason": admission.get("reason", ""),
            "promotion_required": True,
            "confirmation_state": self.confirmation_state,
        }
        return obj
