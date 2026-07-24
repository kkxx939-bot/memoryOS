"""使用严格 ``memory://`` L2 URI 读取完整旧记忆快照。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from foundation.integrity.canonical_json import canonical_json
from infrastructure.editor.snapshot import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotReader,
    VersionedSnapshot,
)
from memory.document import MemoryDocument
from memory.tree import MemoryTree
from memory.uri import MemoryURI

MemorySnapshot: TypeAlias = VersionedSnapshot[MemoryDocument]
MemorySnapshotBatch: TypeAlias = SnapshotBatch[MemoryDocument]


class MemorySnapshotReader:
    """把记忆领域的 URI 和文档模型适配到公共版本快照机制。"""

    def __init__(
        self,
        tree: MemoryTree,
        *,
        config: SnapshotReadConfig | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be a MemoryTree")
        self.tree = tree
        self._reader = SnapshotReader[MemoryDocument](
            load=self._load,
            revision_of=lambda document: document.metadata.revision,
            serialize=self._serialize,
            config=config,
        )

    @property
    def config(self) -> SnapshotReadConfig:
        """返回当前批量读取使用的显式资源边界。"""

        return self._reader.config

    def read(self, uri: MemoryURI | str) -> MemorySnapshot:
        """读取一个 L2 URI；目录和 L0/L1 URI 会被严格拒绝。"""

        parsed = self._document_uri(uri)
        return self._reader.read(str(parsed))

    def read_many(self, uris: Iterable[MemoryURI | str]) -> MemorySnapshotBatch:
        """先验证全部 L2 URI，再执行去重且有界的批量读取。"""

        if isinstance(uris, str) or not isinstance(uris, Iterable):
            raise TypeError("uris must be an iterable of MemoryURI or string values")
        identities = (str(self._document_uri(uri)) for uri in uris)
        return self._reader.read_many(identities)

    def _load(self, identity: str) -> MemoryDocument:
        uri = self._document_uri(identity)
        return self.tree.read(uri.to_address())

    @staticmethod
    def _document_uri(value: MemoryURI | str) -> MemoryURI:
        parsed = MemoryURI.parse(value)
        parsed.to_address()
        return parsed

    @staticmethod
    def _serialize(document: MemoryDocument) -> bytes:
        """序列化完整规范内容，生成与读取版本绑定的稳定摘要输入。"""

        payload = {
            "memory_type": document.kind.value,
            "address": str(MemoryURI.from_address(document.address)),
            "revision": document.metadata.revision,
            "created_at": document.metadata.created_at,
            "updated_at": document.metadata.updated_at,
            "fields": document.fields,
            "markdown_body": document.markdown_body,
        }
        return canonical_json(payload).encode("utf-8")


__all__ = ["MemorySnapshot", "MemorySnapshotBatch", "MemorySnapshotReader"]
