"""上下文数据库里的上下文关系。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.core.time import utc_now


@dataclass(frozen=True)
class ContextRelation:
    source_uri: str
    relation_type: str
    target_uri: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.source_uri.startswith("memoryos://"):
            object.__setattr__(self, "source_uri", str(ContextURI.parse(self.source_uri)))
        if self.target_uri.startswith("memoryos://"):
            object.__setattr__(self, "target_uri", str(ContextURI.parse(self.target_uri)))
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError("relation weight must be finite")
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict:
        return {
            "source_uri": self.source_uri,
            "type": self.relation_type,
            "target_uri": self.target_uri,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
