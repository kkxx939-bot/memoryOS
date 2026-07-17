"""上下文数据库里的资源导入器。"""

from __future__ import annotations

from memoryos.contextdb.resource.resource_model import Resource
from memoryos.contextdb.resource.resource_parser import ResourceParser
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.source_store import SourceStore


class ResourceImporter:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore | None = None,
        *,
        migration_gate=None,  # noqa: ANN001
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.parser = ResourceParser()
        self.migration_gate = migration_gate or getattr(source_store, "migration_gate", None)

    def import_text(self, uri: str, title: str, resource_type: str, content: str, owner_user_id: str | None = None) -> Resource:
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        try:
            return self._import_text_unfenced(uri, title, resource_type, content, owner_user_id)
        finally:
            if callable(release):
                release(fence)

    def _import_text_unfenced(
        self,
        uri: str,
        title: str,
        resource_type: str,
        content: str,
        owner_user_id: str | None,
    ) -> Resource:
        parsed = self.parser.parse(content)
        resource = Resource(uri=uri, title=title, resource_type=resource_type, owner_user_id=owner_user_id, metadata=parsed)
        obj = resource.to_context_object()
        self.source_store.write_object(obj, content=content)
        if self.index_store is not None:
            self.index_store.upsert_index(obj, content=content)
        return resource
