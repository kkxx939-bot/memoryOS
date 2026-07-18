"""Neutral support evidence used by Behavior and ActionPolicy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.clock import utc_now


class SupportAnchorKind(str, Enum):
    BEHAVIOR = "behavior"
    ACTION_POLICY = "action_policy"


@dataclass
class SupportAnchor:
    uri: str
    user_id: str
    title: str
    content: str
    anchor_key: str
    kind: SupportAnchorKind = SupportAnchorKind.BEHAVIOR
    confidence: float = 0.65
    supporting_behavior_uris: list[str] = field(default_factory=list)
    constrains_policy_uris: list[str] = field(default_factory=list)
    policy_rule_type: str = ""
    policy_rule_value: str = ""
    related_action: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = SupportAnchorKind(self.kind)
        if not self.anchor_key or not self.uri:
            raise ValueError("support anchor requires an identity and URI")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def context_type(self) -> ContextType:
        if self.kind is SupportAnchorKind.BEHAVIOR:
            return ContextType.BEHAVIOR_SUPPORT
        return ContextType.ACTION_POLICY_SUPPORT

    def to_context_object(self) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=self.context_type,
            title=self.title,
            owner_user_id=self.user_id,
            semantic_hotness=self.confidence,
            behavior_support_hotness=min(1.0, len(self.supporting_behavior_uris) * 0.15),
            metadata={
                "support_anchor_kind": self.kind.value,
                "anchor_key": self.anchor_key,
                "content": self.content,
                "supporting_behavior_uris": list(self.supporting_behavior_uris),
                "constrains_policy_uris": list(self.constrains_policy_uris),
                "policy_rule_type": self.policy_rule_type,
                "policy_rule_value": self.policy_rule_value,
                "related_action": self.related_action,
            },
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


__all__ = ["SupportAnchor", "SupportAnchorKind"]
