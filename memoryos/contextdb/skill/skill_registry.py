"""上下文数据库里的技能注册表。"""

from __future__ import annotations

from memoryos.contextdb.skill.skill_model import Skill
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.source_store import SourceStore


class SkillRegistry:
    def __init__(
        self,
        source_store: SourceStore | None = None,
        index_store: IndexStore | None = None,
        *,
        migration_gate=None,  # noqa: ANN001
    ) -> None:
        self.skills: dict[str, Skill] = {}
        self.source_store = source_store
        self.index_store = index_store
        self.migration_gate = migration_gate or getattr(source_store, "migration_gate", None)

    def register(self, skill: Skill, content: str | None = None) -> None:
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        try:
            self._register_unfenced(skill, content=content)
        finally:
            if callable(release):
                release(fence)

    def _register_unfenced(self, skill: Skill, content: str | None = None) -> None:
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
