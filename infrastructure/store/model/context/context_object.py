"""上下文事实源与检索投影共享的核心对象。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.context_uri import ContextURI
from infrastructure.store.model.context.lifecycle import LifecycleState


@dataclass
class ContextObject:
    uri: str
    context_type: ContextType
    title: str
    owner_user_id: str | None = None
    tenant_id: str | None = "default"
    layers: ContextLayers = field(default_factory=ContextLayers)
    metadata: dict = field(default_factory=dict)
    relations: list[ContextRelation] = field(default_factory=list)
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    hotness: float = 0.0
    semantic_hotness: float = 0.0
    behavior_support_hotness: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    schema_version: str = "context_object_v1"

    def __post_init__(self) -> None:
        self.uri = str(ContextURI.parse(self.uri))
        if isinstance(self.context_type, str):
            self.context_type = ContextType(self.context_type)
        if isinstance(self.lifecycle_state, str):
            self.lifecycle_state = LifecycleState(self.lifecycle_state)
        for field_name in ("hotness", "semantic_hotness", "behavior_support_hotness"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            value = max(0.0, min(1.0, value))
            setattr(self, field_name, value)

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "context_type": self.context_type.value,
            "title": self.title,
            "owner_user_id": self.owner_user_id,
            "tenant_id": self.tenant_id,
            "layers": self.layers.to_dict(),
            "metadata": self.metadata,
            "relations": [relation.to_dict() for relation in self.relations],
            "lifecycle_state": self.lifecycle_state.value,
            "hotness": self.hotness,
            "semantic_hotness": self.semantic_hotness,
            "behavior_support_hotness": self.behavior_support_hotness,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ContextObject:
        layers = payload.get("layers") or {}
        relations = [
            ContextRelation(
                source_uri=str(item.get("source_uri", payload.get("uri", ""))),
                relation_type=str(item.get("type", item.get("relation_type", ""))),
                target_uri=str(item.get("target_uri", "")),
                weight=float(item.get("weight", 1.0)),
                metadata=dict(item.get("metadata", {})),
                # 关系时间戳属于审计数据；旧记录缺失时间时保持稳定空值，不能在读取时重新生成。
                created_at=str(item.get("created_at") or ""),
            )
            for item in payload.get("relations", [])
            if isinstance(item, dict)
        ]
        return cls(
            uri=str(payload["uri"]),
            context_type=ContextType(str(payload["context_type"])),
            title=str(payload.get("title", "")),
            owner_user_id=payload.get("owner_user_id"),
            tenant_id=payload.get("tenant_id", "default"),
            layers=ContextLayers(
                l0_uri=layers.get("l0_uri"),
                l1_uri=layers.get("l1_uri"),
                l2_uri=layers.get("l2_uri"),
            ),
            metadata=dict(payload.get("metadata", {})),
            relations=relations,
            lifecycle_state=LifecycleState(str(payload.get("lifecycle_state", LifecycleState.ACTIVE.value))),
            hotness=float(payload.get("hotness", 0.0)),
            semantic_hotness=float(payload.get("semantic_hotness", 0.0)),
            behavior_support_hotness=float(payload.get("behavior_support_hotness", 0.0)),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            schema_version=str(payload.get("schema_version", "context_object_v1")),
        )
