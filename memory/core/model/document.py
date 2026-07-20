"""用户可编辑 Markdown 记忆文档模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class MemoryDocumentKind(str, Enum):
    ROOT_INDEX = "root_index"
    PROFILE = "profile"
    PREFERENCES = "preferences"
    KNOWLEDGE_INDEX = "knowledge_index"
    ENTITY = "entity"
    TOPIC = "topic"
    EPISODE = "episode"
    OPEN_LOOPS = "open_loops"
    EXPERIENCE = "experience"


@dataclass(frozen=True)
class MemoryDocument:
    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    document_kind: MemoryDocumentKind
    raw_sha256: str
    size: int
    raw_bytes: bytes = field(repr=False)
    body: str = field(repr=False)
    front_matter: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "front_matter", MappingProxyType(dict(self.front_matter)))

    @property
    def uri(self) -> str:
        return f"memoryos://user/{self.owner_user_id}/memory/documents/{self.document_id}"


__all__ = ["MemoryDocument", "MemoryDocumentKind"]
