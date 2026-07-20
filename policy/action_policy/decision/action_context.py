"""ActionPolicy 决策时使用的最小上下文快照。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionContext:
    user_id: str
    candidate_actions: list[str]
    packed_context: dict
    source_uris: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "candidate_actions": self.candidate_actions,
            "packed_context": self.packed_context,
            "source_uris": self.source_uris,
        }
