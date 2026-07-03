from __future__ import annotations

from memoryos.contextdb.skill.skill_registry import SkillRegistry


class SkillContextBuilder:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def build(self, required_skill_uris: list[str]) -> list[dict]:
        contexts = []
        for uri in required_skill_uris:
            skill = self.registry.get(uri)
            if skill:
                contexts.append(skill.to_context_object().to_dict())
        return contexts
