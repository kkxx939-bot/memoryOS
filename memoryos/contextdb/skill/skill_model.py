"""上下文数据库里的技能数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType


@dataclass
class Skill:
    uri: str
    title: str
    tool_name: str
    risk_level: str = "low"
    metadata: dict = field(default_factory=dict)

    def to_context_object(self) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.SKILL,
            title=self.title,
            metadata={"tool_name": self.tool_name, "risk_level": self.risk_level, "executable": True, **self.metadata},
        )
