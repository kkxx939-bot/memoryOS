from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = MemoryKind(self.kind)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_context_object(self) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=self.title,
            owner_user_id=self.user_id,
            lifecycle_state=LifecycleState.ACTIVE if self.status == "active" else LifecycleState(self.status),
            semantic_hotness=self.confidence,
            behavior_support_hotness=min(1.0, len(self.supporting_behavior_uris) * 0.15),
            metadata={
                "memory_kind": self.kind.value,
                "confidence": self.confidence,
                "tags": self.tags,
                "supporting_behavior_uris": self.supporting_behavior_uris,
                "constrains_policy_uris": self.constrains_policy_uris,
            },
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
