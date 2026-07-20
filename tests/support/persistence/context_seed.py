"""仅供测试准备普通 Context 事实源和 Serving 索引。"""

from __future__ import annotations

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject


def seed_context_object(
    source_store: SourceStore,
    index_store: IndexStore,
    obj: ContextObject,
    *,
    content: str | bytes = "",
) -> None:
    """同步写入测试事实源和索引；生产写入不得使用这个辅助函数。"""

    source_store.write_object(obj, content=content)
    index_content = obj.title if isinstance(content, bytes) else content or obj.metadata.get("summary", obj.title)
    index_store.upsert_index(
        obj,
        content=index_content,
        tenant_id=str(obj.tenant_id or getattr(source_store, "tenant_id", "default") or "default"),
    )


__all__ = ["seed_context_object"]
