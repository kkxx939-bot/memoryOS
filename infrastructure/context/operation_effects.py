"""把 Context 分层和关系语义适配给通用操作事务层。"""

from __future__ import annotations

from infrastructure.context.layers.refresher import LayerRefresher
from infrastructure.context.relations.ordinary import (
    ordinary_relation_serving_eligibility,
    ordinary_relation_specs_for_object,
)
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_object import ContextObject
from transaction.commit.domain_protocols import RelationEligibility


class InfrastructureContextOperationEffects:
    """Context 负责生成语义副作用，Operations 只负责耐久执行。"""

    def prepare_object(self, obj: ContextObject, content: str) -> ContextObject:
        """在事务落盘前确定分层 URI，使 effect digest 在写入前后保持稳定。"""

        if not content:
            return obj
        obj.layers = ContextLayers(
            l0_uri=f"{obj.uri}/.abstract.md",
            l1_uri=f"{obj.uri}/.overview.md",
            l2_uri=f"{obj.uri}/content.md",
        )
        return obj

    def refresh_layers(
        self,
        source_store: SourceStore,
        obj: ContextObject,
        content: str,
        *,
        bullets: list[str] | None = None,
    ) -> ContextObject:
        self.prepare_object(obj, content)
        LayerRefresher(source_store).refresh(obj, content, bullets=bullets)
        return obj

    def relation_specs_for_object(self, obj: ContextObject) -> list[dict]:
        return ordinary_relation_specs_for_object(obj)

    def relation_eligibility(
        self,
        spec: dict,
        *,
        authority_uri: str,
        tenant_id: str,
        source_store: SourceStore,
        index_store: IndexStore,
        authority_object: ContextObject | None,
    ) -> RelationEligibility:
        result = ordinary_relation_serving_eligibility(
            spec,
            authority_uri=authority_uri,
            tenant_id=tenant_id,
            source_store=source_store,
            index_store=index_store,
            authority_object=authority_object,
            allow_virtual_targets=True,
        )
        return RelationEligibility(result.allowed, result.reason)


__all__ = ["InfrastructureContextOperationEffects"]
