"""按 Schema 范围搜索并预取相关旧记忆完整快照。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Protocol

from memory.editor.reader import MemorySnapshotReader
from memory.editor.retrieval.model import (
    MemoryRelatedContext,
    MemoryRetrievalConfig,
    MemoryRetrievalError,
    MemorySearchHit,
)
from memory.editor.retrieval.query import ConversationSegmentQueryBuilder
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind
from memory.schema import MemoryOperationMode, MemorySchemaRegistry
from memory.uri import MemoryURI, MemoryURINodeType
from pre.conversation import ConversationMessageRole, ConversationSegment


class MemorySemanticSearch(Protocol):
    """由具体索引实现的记忆领域语义搜索契约。"""

    async def search(
        self,
        query: str,
        *,
        roots: tuple[MemoryURI, ...],
        limit: int,
    ) -> Sequence[MemorySearchHit]: ...


class MemoryRelatedRetriever:
    """复用 OpenViking 的固定直读、目录搜索和 Top-N 预取编排。"""

    def __init__(
        self,
        *,
        schema_registry: MemorySchemaRegistry,
        snapshot_reader: MemorySnapshotReader,
        semantic_search: MemorySemanticSearch,
        config: MemoryRetrievalConfig | None = None,
    ) -> None:
        if not isinstance(schema_registry, MemorySchemaRegistry):
            raise TypeError("schema_registry must be a MemorySchemaRegistry")
        if not isinstance(snapshot_reader, MemorySnapshotReader):
            raise TypeError("snapshot_reader must be a MemorySnapshotReader")
        if not callable(getattr(semantic_search, "search", None)):
            raise TypeError("semantic_search must implement async search")
        if config is not None and not isinstance(config, MemoryRetrievalConfig):
            raise TypeError("config must be MemoryRetrievalConfig")
        self.schema_registry = schema_registry
        self.snapshot_reader = snapshot_reader
        self.semantic_search = semantic_search
        self.config = config or MemoryRetrievalConfig()
        self.query_builder = ConversationSegmentQueryBuilder(self.config)

        fixed_uris, _roots = self._schema_read_plan()
        maximum_snapshots = len(fixed_uris) + self.config.max_tool_uris + self.config.search_limit
        if maximum_snapshots > self.snapshot_reader.config.max_items:
            raise ValueError("retrieval result can exceed the snapshot reader item limit")

    async def retrieve(self, segment: ConversationSegment) -> MemoryRelatedContext:
        """确定相关 URI，并在返回前完整读取每一个选中 L2 文档。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        query = self.query_builder.build(segment)
        fixed_uris, search_roots = self._schema_read_plan()
        tool_uris = self._tool_uris(segment)
        search_hits = await self._search(query, search_roots)
        selected = tuple(
            sorted(
                {
                    *(str(uri) for uri in fixed_uris),
                    *(str(uri) for uri in tool_uris),
                    *(str(hit.uri) for hit in search_hits),
                }
            )
        )
        snapshots = self.snapshot_reader.read_many(selected)
        return MemoryRelatedContext(
            conversation_id=segment.conversation_id,
            segment_id=segment.segment_id,
            source_segment_digest=segment.digest,
            query=query,
            search_roots=search_roots,
            search_hits=search_hits,
            snapshots=snapshots,
        )

    def _schema_read_plan(self) -> tuple[tuple[MemoryURI, ...], tuple[MemoryURI, ...]]:
        fixed: set[MemoryURI] = set()
        roots: set[MemoryURI] = set()
        for schema in self.schema_registry.all():
            if schema.operation_mode is MemoryOperationMode.ADD_ONLY:
                continue
            if schema.kind is MemoryKind.PROFILE:
                fixed.add(MemoryURI.from_address(MemoryAddress.profile()))
                continue
            if schema.kind is MemoryKind.TOOL:
                continue
            path_root = PurePosixPath(schema.path_template).parts[0]
            roots.add(MemoryURI.from_directory(MemoryDirectory((path_root,))))
        return (
            tuple(sorted(fixed, key=str)),
            tuple(sorted(roots, key=str)),
        )

    def _tool_uris(self, segment: ConversationSegment) -> tuple[MemoryURI, ...]:
        names: set[str] = set()
        for message in segment.messages:
            if message.role is not ConversationMessageRole.TOOL_CALL:
                continue
            assert message.tool_name is not None
            names.add(message.tool_name)
            if len(names) > self.config.max_tool_uris:
                raise MemoryRetrievalError("conversation segment contains too many distinct tool names")
        try:
            return tuple(
                sorted(
                    (MemoryURI.from_address(MemoryAddress.tool(name)) for name in names),
                    key=str,
                )
            )
        except (TypeError, ValueError) as exc:
            raise MemoryRetrievalError(
                "conversation tool name cannot map exactly to the confirmed memory tree"
            ) from exc

    async def _search(
        self,
        query: str,
        roots: tuple[MemoryURI, ...],
    ) -> tuple[MemorySearchHit, ...]:
        if not roots:
            return ()
        try:
            raw_hits = await self.semantic_search.search(
                query,
                roots=roots,
                limit=self.config.search_limit,
            )
        except Exception as exc:
            raise MemoryRetrievalError("memory semantic search failed") from exc
        if isinstance(raw_hits, str) or not isinstance(raw_hits, Sequence):
            raise MemoryRetrievalError("memory semantic search must return a sequence of hits")
        if len(raw_hits) > self.config.search_limit:
            raise MemoryRetrievalError("memory semantic search exceeded its requested result limit")

        by_uri: dict[MemoryURI, MemorySearchHit] = {}
        for hit in raw_hits:
            if not isinstance(hit, MemorySearchHit):
                raise MemoryRetrievalError("memory semantic search returned an invalid hit")
            if hit.uri.node_type is not MemoryURINodeType.DOCUMENT:
                raise MemoryRetrievalError("memory semantic search returned a non-L2 URI")
            if not any(hit.uri.matches_prefix(root) for root in roots):
                raise MemoryRetrievalError("memory semantic search returned an out-of-scope URI")
            current = by_uri.get(hit.uri)
            if current is None or hit.score > current.score:
                by_uri[hit.uri] = hit
        return tuple(sorted(by_uri.values(), key=lambda hit: (-hit.score, str(hit.uri))))


__all__ = ["MemoryRelatedRetriever", "MemorySemanticSearch"]
