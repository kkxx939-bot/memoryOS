"""行为模块里的行为模式。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.behavior.model.opportunity import OpportunityStats
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.clock import utc_now
from memoryos.core.ids import new_id


@dataclass
class BehaviorCluster:
    user_id: str
    scene_key: str
    memory_anchor_uri: str
    case_refs: list[str]
    cluster_id: str = field(default_factory=lambda: new_id("cluster"))
    confidence: float = 0.45
    status: str = "active"
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.memory_anchor_uri:
            raise ValueError("BehaviorCluster requires memory_anchor_uri")


@dataclass
class BehaviorPattern:
    user_id: str
    scene_key: str
    trigger_conditions: dict
    memory_anchor_uri: str
    case_refs: list[str]
    action_distribution: list[dict]
    pattern_id: str = field(default_factory=lambda: new_id("pattern"))
    opportunity: OpportunityStats = field(default_factory=OpportunityStats)
    hotness: float = 0.0
    confidence: float = 0.65
    status: str = "active"
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.memory_anchor_uri:
            raise ValueError("BehaviorPattern requires memory_anchor_uri")
        self.hotness = max(0.0, min(1.0, float(self.hotness)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def uri(self) -> str:
        return f"memoryos://user/{self.user_id}/behavior/patterns/{self.scene_key}/{self.pattern_id}"

    def to_context_object(self) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.BEHAVIOR_PATTERN,
            title=f"BehaviorPattern {self.scene_key}",
            owner_user_id=self.user_id,
            hotness=self.hotness,
            behavior_support_hotness=self.confidence,
            metadata={
                "scene_key": self.scene_key,
                "trigger_conditions": self.trigger_conditions,
                "memory_anchor_uri": self.memory_anchor_uri,
                "case_refs": self.case_refs,
                "action_distribution": self.action_distribution,
                "opportunity": self.opportunity.__dict__,
                "status": self.status,
            },
            updated_at=self.updated_at,
        )
