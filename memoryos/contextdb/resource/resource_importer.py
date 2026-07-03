from __future__ import annotations

from memoryos.contextdb.resource.resource_model import Resource
from memoryos.contextdb.resource.resource_parser import ResourceParser
from memoryos.contextdb.store.source_store import SourceStore


class ResourceImporter:
    def __init__(self, source_store: SourceStore) -> None:
        self.source_store = source_store
        self.parser = ResourceParser()

    def import_text(self, uri: str, title: str, resource_type: str, content: str, owner_user_id: str | None = None) -> Resource:
        parsed = self.parser.parse(content)
        resource = Resource(uri=uri, title=title, resource_type=resource_type, owner_user_id=owner_user_id, metadata=parsed)
        self.source_store.write_object(resource.to_context_object(), content=content)
        return resource
