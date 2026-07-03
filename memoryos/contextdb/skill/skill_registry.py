from __future__ import annotations

from memoryos.contextdb.skill.skill_model import Skill


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self.skills[skill.uri] = skill

    def get(self, uri: str) -> Skill | None:
        return self.skills.get(uri)

    def find_by_tool(self, tool_name: str) -> list[Skill]:
        return [skill for skill in self.skills.values() if skill.tool_name == tool_name]
