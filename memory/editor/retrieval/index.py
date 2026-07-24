"""只面向当前记忆树的有界向量索引契约和内存实现。"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from LLMClient import Embedder, EmbeddingVector
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.tree import MemoryTree
from memory.uri import MemoryURI, MemoryURINodeType


class MemoryVectorIndexError(RuntimeError):
    """记忆索引无法在完整性和资源边界内完成搜索。"""


@dataclass(frozen=True)
class MemoryVectorIndexConfig:
    """限制一次即时索引搜索和进程内向量缓存。"""

    max_records: int = 10_000
    max_direct_entries: int = 1_000
    max_directories: int = 2_000
    max_search_hits: int = 10_000
    max_record_chars: int = 16_000
    max_cache_entries: int = 10_000

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_records", self.max_records, 100_000),
            ("max_direct_entries", self.max_direct_entries, 10_000),
            ("max_directories", self.max_directories, 100_000),
            ("max_search_hits", self.max_search_hits, 10_000),
            ("max_record_chars", self.max_record_chars, 1_000_000),
            ("max_cache_entries", self.max_cache_entries, 100_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between one and {maximum}")


@dataclass(frozen=True)
class MemoryVectorMatch:
    """索引返回的一个目录语义节点或 L2 文档候选。"""

    uri: MemoryURI
    level: MemoryLevel
    directory: MemoryDirectory
    content: str
    score: float

    def __post_init__(self) -> None:
        uri = MemoryURI.parse(self.uri)
        level = MemoryLevel(self.level)
        if level is MemoryLevel.DETAIL:
            if uri.node_type is not MemoryURINodeType.DOCUMENT:
                raise ValueError("L2 vector match must identify a memory document")
            if uri.containing_directory != self.directory:
                raise ValueError("L2 vector match directory does not match its URI")
        elif uri.node_type is not MemoryURINodeType.LAYER:
            raise ValueError("L0/L1 vector match must identify a semantic layer")
        else:
            directory, uri_level = uri.to_layer()
            if directory != self.directory or uri_level is not level:
                raise ValueError("semantic vector match does not match its URI")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("memory vector match content must be non-empty")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("memory vector match score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("memory vector match score must be a finite cosine value")
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "score", score)


class MemoryVectorIndex(Protocol):
    """使用已经生成的 query vector 搜索受限记忆节点。"""

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        roots: tuple[MemoryURI, ...],
        levels: tuple[MemoryLevel, ...],
        limit: int,
    ) -> Sequence[MemoryVectorMatch]: ...

    async def search_children(
        self,
        query_vector: EmbeddingVector,
        *,
        parent: MemoryURI,
        limit: int,
    ) -> Sequence[MemoryVectorMatch]: ...


@dataclass(frozen=True)
class _VectorRecord:
    uri: MemoryURI
    level: MemoryLevel
    directory: MemoryDirectory
    content: str
    digest: str


class MemoryTreeVectorIndex:
    """即时读取 L2/L0/L1，并按内容摘要复用有界进程内向量。"""

    def __init__(
        self,
        tree: MemoryTree,
        embedder: Embedder,
        *,
        config: MemoryVectorIndexConfig | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be a MemoryTree")
        if not callable(getattr(embedder, "embed_documents", None)):
            raise TypeError("embedder must implement embed_documents")
        self.tree = tree
        self.embedder = embedder
        self.config = config or MemoryVectorIndexConfig()
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._cache_lock = threading.RLock()

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        roots: tuple[MemoryURI, ...],
        levels: tuple[MemoryLevel, ...],
        limit: int,
    ) -> tuple[MemoryVectorMatch, ...]:
        """在全部受限根目录下搜索指定层级。"""

        vector = self._query_vector(query_vector)
        normalized_roots = self._roots(roots)
        normalized_levels = self._levels(levels)
        maximum = self._limit(limit)
        records: dict[MemoryURI, _VectorRecord] = {}
        visited_directories: set[MemoryDirectory] = set()
        pending = [root.to_directory() for root in reversed(normalized_roots)]
        while pending:
            directory = pending.pop()
            if directory in visited_directories:
                continue
            visited_directories.add(directory)
            if len(visited_directories) > self.config.max_directories:
                raise MemoryVectorIndexError("memory vector search exceeded its directory bound")
            if not self.tree.directory_exists(directory):
                continue
            if MemoryLevel.DETAIL in normalized_levels:
                for record in self._document_records(directory):
                    records[record.uri] = record
                    self._require_record_bound(records)
            layer = self._directory_record(directory, normalized_levels)
            if layer is not None:
                records[layer.uri] = layer
                self._require_record_bound(records)
            children = self.tree.child_directories(
                directory,
                limit=self.config.max_direct_entries,
            )
            pending.extend(reversed(children))
        return await self._score(vector, tuple(records.values()), maximum)

    async def search_children(
        self,
        query_vector: EmbeddingVector,
        *,
        parent: MemoryURI,
        limit: int,
    ) -> tuple[MemoryVectorMatch, ...]:
        """搜索一个目录的直接 L2 和直接子目录语义层。"""

        vector = self._query_vector(query_vector)
        parsed = MemoryURI.parse(parent)
        if parsed.node_type is not MemoryURINodeType.DIRECTORY:
            raise ValueError("memory vector child search parent must be a directory URI")
        maximum = self._limit(limit)
        directory = parsed.to_directory()
        if not self.tree.directory_exists(directory):
            return ()
        records = list(self._document_records(directory))
        for child in self.tree.child_directories(
            directory,
            limit=self.config.max_direct_entries,
        ):
            record = self._directory_record(
                child,
                (MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW),
            )
            if record is not None:
                records.append(record)
        if len(records) > self.config.max_records:
            raise MemoryVectorIndexError("memory child search exceeded its record bound")
        return await self._score(vector, tuple(records), maximum)

    async def _score(
        self,
        query_vector: EmbeddingVector,
        records: tuple[_VectorRecord, ...],
        limit: int,
    ) -> tuple[MemoryVectorMatch, ...]:
        if not records:
            return ()
        vectors = await self._record_vectors(records)
        matches: list[MemoryVectorMatch] = []
        for record, vector in zip(records, vectors, strict=True):
            if vector.dimension != query_vector.dimension:
                raise MemoryVectorIndexError("query and memory embedding dimensions do not match")
            score = sum(
                query_value * record_value
                for query_value, record_value in zip(
                    query_vector.values,
                    vector.values,
                    strict=True,
                )
            )
            matches.append(
                MemoryVectorMatch(
                    uri=record.uri,
                    level=record.level,
                    directory=record.directory,
                    content=record.content,
                    score=max(-1.0, min(1.0, score)),
                )
            )
        matches.sort(key=lambda item: (-item.score, str(item.uri)))
        return tuple(matches[:limit])

    async def _record_vectors(
        self,
        records: tuple[_VectorRecord, ...],
    ) -> tuple[EmbeddingVector, ...]:
        resolved: list[EmbeddingVector | None] = [None] * len(records)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        with self._cache_lock:
            for index, record in enumerate(records):
                cached = self._cache.get(record.digest)
                if cached is None:
                    missing_indexes.append(index)
                    missing_texts.append(record.content)
                    continue
                self._cache.move_to_end(record.digest)
                resolved[index] = cached
        if missing_texts:
            generated = await self.embedder.embed_documents(tuple(missing_texts))
            if not isinstance(generated, tuple) or len(generated) != len(missing_texts):
                raise MemoryVectorIndexError("embedder returned an unexpected document vector count")
            if any(not isinstance(vector, EmbeddingVector) for vector in generated):
                raise MemoryVectorIndexError("embedder returned an invalid document vector")
            with self._cache_lock:
                for index, vector in zip(missing_indexes, generated, strict=True):
                    digest = records[index].digest
                    self._cache[digest] = vector
                    self._cache.move_to_end(digest)
                    resolved[index] = vector
                while len(self._cache) > self.config.max_cache_entries:
                    self._cache.popitem(last=False)
        if any(vector is None for vector in resolved):
            raise AssertionError("memory vector resolution ended with a missing value")
        return tuple(vector for vector in resolved if vector is not None)

    def _document_records(self, directory: MemoryDirectory) -> tuple[_VectorRecord, ...]:
        records: list[_VectorRecord] = []
        for address in self.tree.direct_addresses(
            directory,
            limit=self.config.max_direct_entries,
        ):
            document = self.tree.read(address)
            if self._excluded(document.address, document.fields):
                continue
            uri = MemoryURI.from_address(address)
            content = self._bounded(
                f"[{address.kind.value}] {uri.decoded_path}\n{document.markdown_body.strip()}"
            )
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            records.append(
                _VectorRecord(
                    uri=uri,
                    level=MemoryLevel.DETAIL,
                    directory=directory,
                    content=content,
                    digest=digest,
                )
            )
        return tuple(records)

    def _directory_record(
        self,
        directory: MemoryDirectory,
        levels: tuple[MemoryLevel, ...],
    ) -> _VectorRecord | None:
        for level in (MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW):
            if level not in levels or not self.tree.layer_exists(directory, level):
                continue
            content = self.tree.read_layer_bounded(
                directory,
                level,
                max_bytes=self.config.max_record_chars * 16 + 4,
            ).strip()
            if not content:
                raise MemoryVectorIndexError("memory semantic layer is empty")
            uri = MemoryURI.from_layer(directory, level)
            rendered = self._bounded(f"[directory] {uri.decoded_path}\n{content}")
            return _VectorRecord(
                uri=uri,
                level=level,
                directory=directory,
                content=rendered,
                digest=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            )
        return None

    def _bounded(self, text: str) -> str:
        maximum = self.config.max_record_chars
        if len(text) <= maximum:
            return text
        if maximum <= 3:
            return text[:maximum]
        return text[: maximum - 3].rstrip() + "..."

    @staticmethod
    def _excluded(address: MemoryAddress, fields: object) -> bool:
        if address.kind is not MemoryKind.INTENTION or not isinstance(fields, Mapping):
            return False
        return fields.get("status") == "completed"

    def _roots(self, roots: tuple[MemoryURI, ...]) -> tuple[MemoryURI, ...]:
        if not isinstance(roots, tuple) or not roots:
            raise ValueError("memory vector search requires at least one root")
        normalized = tuple(MemoryURI.parse(root) for root in roots)
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in normalized):
            raise ValueError("memory vector roots must identify directories")
        if len(normalized) != len(set(normalized)):
            raise ValueError("memory vector roots must be unique")
        return normalized

    @staticmethod
    def _levels(levels: tuple[MemoryLevel, ...]) -> tuple[MemoryLevel, ...]:
        if not isinstance(levels, tuple) or not levels:
            raise ValueError("memory vector search requires at least one level")
        normalized = tuple(MemoryLevel(level) for level in levels)
        if len(normalized) != len(set(normalized)):
            raise ValueError("memory vector levels must be unique")
        return normalized

    def _limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config.max_search_hits:
            raise ValueError("memory vector search limit is outside its configured bound")
        return limit

    @staticmethod
    def _query_vector(value: EmbeddingVector) -> EmbeddingVector:
        if not isinstance(value, EmbeddingVector):
            raise TypeError("query_vector must be an EmbeddingVector")
        return value

    def _require_record_bound(self, records: dict[MemoryURI, _VectorRecord]) -> None:
        if len(records) > self.config.max_records:
            raise MemoryVectorIndexError("memory vector search exceeded its record bound")


__all__ = [
    "MemoryTreeVectorIndex",
    "MemoryVectorIndex",
    "MemoryVectorIndexConfig",
    "MemoryVectorIndexError",
    "MemoryVectorMatch",
]
