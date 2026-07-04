from __future__ import annotations

from memoryos.contextdb.skill.skill_model import Skill
from memoryos.contextdb.store.source_store import IndexStore, SourceStore


class SkillRegistry:
    def __init__(self, source_store: SourceStore | None = None, index_store: IndexStore | None = None) -> None:
        self.skills: dict[str, Skill] = {}
        self.source_store = source_store
        self.index_store = index_store

    def register(self, skill: Skill, content: str | None = None) -> None:
        self.skills[skill.uri] = skill
        obj = skill.to_context_object()
        if self.source_store is not None:
            self.source_store.write_object(obj, content=content or skill.title)
        if self.index_store is not None:
            self.index_store.upsert_index(obj, content=content or skill.title)

    def get(self, uri: str) -> Skill | None:
        return self.skills.get(uri)

    def find_by_tool(self, tool_name: str) -> list[Skill]:
        return [skill for skill in self.skills.values() if skill.tool_name == tool_name]
