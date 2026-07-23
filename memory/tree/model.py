"""长期记忆树的语义地址。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from foundation.ids import require_safe_path_segment


class MemoryKind(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    ENTITY = "entity"
    TOOL = "tool"
    EVENT = "event"
    INTENTION = "intention"


def _semantic_name(value: object, field_name: str) -> str:
    name = require_safe_path_segment(value, field_name)
    if name != name.strip() or any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError(f"{field_name} must not contain surrounding whitespace or control characters")
    if name.casefold().endswith(".md"):
        raise ValueError(f"{field_name} must not include the .md suffix")
    return name


@dataclass(frozen=True)
class MemoryAddress:
    """唯一映射到记忆树中一个 Markdown 文件的语义地址。"""

    kind: MemoryKind
    name: str = ""
    category: str = ""
    event_date: date | None = None

    def __post_init__(self) -> None:
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is MemoryKind.PROFILE:
            if self.name or self.category or self.event_date is not None:
                raise ValueError("profile memory does not accept name, category, or event_date")
            return

        object.__setattr__(self, "name", _semantic_name(self.name, f"{kind.value} name"))
        if kind is MemoryKind.ENTITY:
            object.__setattr__(self, "category", _semantic_name(self.category, "entity category"))
        elif self.category:
            raise ValueError(f"{kind.value} memory does not accept category")

        if kind is MemoryKind.EVENT:
            if not isinstance(self.event_date, date):
                raise ValueError("event memory requires event_date")
        elif self.event_date is not None:
            raise ValueError(f"{kind.value} memory does not accept event_date")

    @classmethod
    def profile(cls) -> MemoryAddress:
        return cls(MemoryKind.PROFILE)

    @classmethod
    def preference(cls, topic: str) -> MemoryAddress:
        return cls(MemoryKind.PREFERENCE, name=topic)

    @classmethod
    def entity(cls, category: str, name: str) -> MemoryAddress:
        return cls(MemoryKind.ENTITY, name=name, category=category)

    @classmethod
    def tool(cls, tool_name: str) -> MemoryAddress:
        return cls(MemoryKind.TOOL, name=tool_name)

    @classmethod
    def event(cls, event_date: date, event_name: str) -> MemoryAddress:
        return cls(MemoryKind.EVENT, name=event_name, event_date=event_date)

    @classmethod
    def intention(cls, intent_name: str) -> MemoryAddress:
        return cls(MemoryKind.INTENTION, name=intent_name)


__all__ = ["MemoryAddress", "MemoryKind"]
