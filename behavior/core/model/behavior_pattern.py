"""行为模块里的行为模式。"""

from __future__ import annotations

from dataclasses import dataclass, field

from behavior.core.model.opportunity import OpportunityStats
from foundation.clock import utc_now
from foundation.ids import new_id


@dataclass
class BehaviorCluster:
    user_id: str
    scene_key: str
    support_anchor_uri: str
    case_refs: list[str]
    cluster_id: str = field(default_factory=lambda: new_id("cluster"))
    confidence: float = 0.45
    status: str = "active"
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.support_anchor_uri:
            raise ValueError("BehaviorCluster requires support_anchor_uri")


@dataclass
class BehaviorPattern:
    user_id: str
    scene_key: str
    trigger_conditions: dict
    support_anchor_uri: str
    case_refs: list[str]
    action_distribution: list[dict]
    pattern_id: str = field(default_factory=lambda: new_id("pattern"))
    opportunity: OpportunityStats = field(default_factory=OpportunityStats)
    hotness: float = 0.0
    confidence: float = 0.65
    status: str = "active"
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.support_anchor_uri:
            raise ValueError("BehaviorPattern requires support_anchor_uri")
        self.hotness = max(0.0, min(1.0, float(self.hotness)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def uri(self) -> str:
        return f"memoryos://user/{self.user_id}/behavior/patterns/{self.scene_key}/{self.pattern_id}"
