from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType


@dataclass
class Resource:
    uri: str
    title: str
    resource_type: str
    owner_user_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_context_object(self) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.RESOURCE,
            title=self.title,
            owner_user_id=self.owner_user_id,
            metadata={"resource_type": self.resource_type, **self.metadata},
        )
