from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContextRelation:
    source_uri: str
    relation_type: str
    target_uri: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_uri": self.source_uri,
            "type": self.relation_type,
            "target_uri": self.target_uri,
            "weight": self.weight,
            "metadata": self.metadata,
        }
