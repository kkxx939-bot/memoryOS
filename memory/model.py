"""长期记忆域共享的类型、地址和目录模型。"""

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


class MemoryLevel(int, Enum):
    """记忆树中的语义层级。"""

    ABSTRACT = 0
    OVERVIEW = 1
    DETAIL = 2

    @property
    def sidecar_filename(self) -> str:
        """返回 L0/L1 唯一侧车文件名；L2 没有侧车文件。"""

        if self is MemoryLevel.ABSTRACT:
            return ".abstract.md"
        if self is MemoryLevel.OVERVIEW:
            return ".overview.md"
        raise ValueError("L2 uses a MemoryAddress instead of a semantic sidecar")

    @classmethod
    def from_sidecar_filename(cls, filename: object) -> MemoryLevel | None:
        """把唯一侧车文件名反解为 L0/L1。"""

        if filename == ".abstract.md":
            return cls.ABSTRACT
        if filename == ".overview.md":
            return cls.OVERVIEW
        return None


def _semantic_name(value: object, field_name: str) -> str:
    name = require_safe_path_segment(value, field_name)
    if name != name.strip() or any(not character.isprintable() for character in name):
        raise ValueError(f"{field_name} must not contain surrounding whitespace or control characters")
    if name.casefold().endswith(".md"):
        raise ValueError(f"{field_name} must not include the .md suffix")
    reserved_stems = {
        MemoryLevel.ABSTRACT.sidecar_filename[:-3],
        MemoryLevel.OVERVIEW.sidecar_filename[:-3],
    }
    if name.casefold() in reserved_stems:
        raise ValueError(f"{field_name} conflicts with a reserved semantic layer")
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


@dataclass(frozen=True)
class MemoryDirectory:
    """严格限定于已确认记忆树的目录地址。"""

    parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.parts, str) or not isinstance(self.parts, tuple):
            raise TypeError("memory directory parts must be a tuple of strings")
        parts = tuple(self.parts)
        if not parts:
            return
        root = parts[0]
        if root not in {"preferences", "entities", "tools", "events", "intentions"}:
            raise ValueError("memory directory is outside the confirmed tree")
        if root in {"preferences", "tools", "intentions"}:
            if len(parts) != 1:
                raise ValueError(f"{root} memory directory cannot contain subdirectories")
            return
        if root == "entities":
            if len(parts) > 2:
                raise ValueError("entity memory directory is deeper than category level")
            if len(parts) == 2:
                _semantic_name(parts[1], "entity category")
            return
        self._validate_event_parts(parts)

    @staticmethod
    def _validate_event_parts(parts: tuple[str, ...]) -> None:
        if len(parts) > 4:
            raise ValueError("event memory directory is deeper than day level")
        values = parts[1:]
        widths = (4, 2, 2)
        labels = ("year", "month", "day")
        for value, width, label in zip(values, widths, labels, strict=False):
            if not isinstance(value, str) or len(value) != width or not value.isdigit():
                raise ValueError(f"event {label} directory has an invalid format")
        if values:
            year = int(values[0])
            if not 1 <= year <= 9999:
                raise ValueError("event year directory is outside the calendar range")
        if len(values) >= 2:
            month = int(values[1])
            if not 1 <= month <= 12:
                raise ValueError("event month directory is outside the calendar range")
        if len(values) == 3:
            try:
                date(int(values[0]), int(values[1]), int(values[2]))
            except ValueError as exc:
                raise ValueError("event day directory is not a valid calendar date") from exc

    @classmethod
    def root(cls) -> MemoryDirectory:
        return cls()

    @classmethod
    def preferences(cls) -> MemoryDirectory:
        return cls(("preferences",))

    @classmethod
    def entities(cls, category: str | None = None) -> MemoryDirectory:
        if category is None:
            return cls(("entities",))
        return cls(("entities", _semantic_name(category, "entity category")))

    @classmethod
    def tools(cls) -> MemoryDirectory:
        return cls(("tools",))

    @classmethod
    def events(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> MemoryDirectory:
        if year is None:
            if month is not None or day is not None:
                raise ValueError("event month or day requires a year")
            return cls(("events",))
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("event directory year must be an integer")
        parts = ["events", f"{year:04d}"]
        if month is None:
            if day is not None:
                raise ValueError("event day requires a month")
            return cls(tuple(parts))
        if isinstance(month, bool) or not isinstance(month, int):
            raise TypeError("event directory month must be an integer")
        parts.append(f"{month:02d}")
        if day is None:
            return cls(tuple(parts))
        if isinstance(day, bool) or not isinstance(day, int):
            raise TypeError("event directory day must be an integer")
        parts.append(f"{day:02d}")
        return cls(tuple(parts))

    @classmethod
    def intentions(cls) -> MemoryDirectory:
        return cls(("intentions",))

    @classmethod
    def for_address(cls, address: MemoryAddress) -> MemoryDirectory:
        if not isinstance(address, MemoryAddress):
            raise TypeError("address must be a MemoryAddress")
        if address.kind is MemoryKind.PROFILE:
            return cls.root()
        if address.kind is MemoryKind.PREFERENCE:
            return cls.preferences()
        if address.kind is MemoryKind.ENTITY:
            return cls.entities(address.category)
        if address.kind is MemoryKind.TOOL:
            return cls.tools()
        if address.kind is MemoryKind.EVENT:
            assert address.event_date is not None
            return cls.events(
                address.event_date.year,
                address.event_date.month,
                address.event_date.day,
            )
        return cls.intentions()

    def parent(self) -> MemoryDirectory | None:
        """返回上级目录；根目录没有上级。"""

        if not self.parts:
            return None
        return MemoryDirectory(self.parts[:-1])

    def lineage(self) -> tuple[MemoryDirectory, ...]:
        """按当前目录到根目录的顺序返回完整祖先链。"""

        directories: list[MemoryDirectory] = []
        current: MemoryDirectory | None = self
        while current is not None:
            directories.append(current)
            current = current.parent()
        return tuple(directories)


__all__ = ["MemoryAddress", "MemoryDirectory", "MemoryKind", "MemoryLevel"]
