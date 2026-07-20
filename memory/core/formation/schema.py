"""模型只能选择的有限记忆候选类型，不能在运行时扩张写入权限。"""

from __future__ import annotations

import builtins
from dataclasses import dataclass

from memory.core.model import MemoryCandidateKind

MEMORY_SCHEMA_VERSION = "markdown_memory_candidate_v1"


@dataclass(frozen=True)
class MemoryCandidateSchema:
    candidate_kind: MemoryCandidateKind
    description: str
    requires_occurred_at: bool = False
    allow_assistant_source: bool = False


class MemoryCandidateRegistry:
    def __init__(self, schemas: builtins.list[MemoryCandidateSchema] | None = None) -> None:
        # ``[]`` 表示调用方明确关闭全部候选类型，不能误退回默认集合。
        rows = self._builtins() if schemas is None else schemas
        if len({item.candidate_kind for item in rows}) != len(rows):
            raise ValueError("memory candidate schemas must use unique kinds")
        self._schemas = {item.candidate_kind: item for item in rows}

    def get(self, kind: MemoryCandidateKind | str) -> MemoryCandidateSchema:
        return self._schemas[MemoryCandidateKind(kind)]

    def list(self) -> builtins.list[MemoryCandidateSchema]:
        return list(self._schemas.values())

    @staticmethod
    def _builtins() -> builtins.list[MemoryCandidateSchema]:
        return [
            MemoryCandidateSchema(
                MemoryCandidateKind.PROFILE_FACT,
                "稳定的用户身份、背景或自我描述。",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.PREFERENCE,
                "长期偏好、沟通习惯或稳定约束。",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.ENTITY_NOTE,
                "关于人物、组织、产品、系统或其他实体的知识。",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.TOPIC_NOTE,
                "按主题组织、跨事件复用的知识，而不是项目目录信息。",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.EPISODE,
                "具有明确发生时间的讨论、事件、决策或结果。",
                requires_occurred_at=True,
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.OPEN_LOOP,
                "尚未解决的问题、待确认事项或后续跟进。",
            ),
            MemoryCandidateSchema(
                MemoryCandidateKind.EXPERIENCE,
                "从实际结果中提炼出的可复用经验。",
                requires_occurred_at=True,
                allow_assistant_source=True,
            ),
        ]


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryCandidateKind",
    "MemoryCandidateRegistry",
    "MemoryCandidateSchema",
]
