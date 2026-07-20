"""行为案例、聚类和模式共同引用的支撑证据锚点。"""

from __future__ import annotations

from dataclasses import dataclass, field

from foundation.clock import utc_now


@dataclass(frozen=True)
class BehaviorSupportAnchor:
    """描述一个行为主题由哪些行为记录支撑，不包含持久化逻辑。"""

    uri: str
    user_id: str
    title: str
    content: str
    anchor_key: str
    confidence: float = 0.65
    supporting_behavior_uris: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.uri or not self.user_id or not self.anchor_key:
            raise ValueError("behavior support anchor requires URI, user and anchor key")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(
            self,
            "supporting_behavior_uris",
            list(dict.fromkeys(str(uri) for uri in self.supporting_behavior_uris if str(uri))),
        )


__all__ = ["BehaviorSupportAnchor"]
