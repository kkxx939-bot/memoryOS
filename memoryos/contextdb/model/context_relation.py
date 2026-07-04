from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.core.time import utc_now


@dataclass(frozen=True)
class ContextRelation:
    source_uri: str
    relation_type: str
    target_uri: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "source_uri": self.source_uri,
            "type": self.relation_type,
            "target_uri": self.target_uri,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
