"""与领域地址、Schema 和存储树无关的版本快照读取流程。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

from infrastructure.editor.snapshot.model import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotState,
    VersionedSnapshot,
)

T = TypeVar("T")


class SnapshotReadLimitError(ValueError):
    """单项或整批旧版本读取超过显式配置边界。"""


class SnapshotReader(Generic[T]):
    """使用领域提供的函数读取对象并形成确定性版本快照。"""

    def __init__(
        self,
        *,
        load: Callable[[str], T],
        revision_of: Callable[[T], int],
        serialize: Callable[[T], bytes],
        config: SnapshotReadConfig | None = None,
    ) -> None:
        for name, function in {
            "load": load,
            "revision_of": revision_of,
            "serialize": serialize,
        }.items():
            if not callable(function):
                raise TypeError(f"{name} must be callable")
        if config is not None and not isinstance(config, SnapshotReadConfig):
            raise TypeError("config must be SnapshotReadConfig")
        self._load = load
        self._revision_of = revision_of
        self._serialize = serialize
        self.config = config or SnapshotReadConfig()

    def read(self, identity: str) -> VersionedSnapshot[T]:
        """完整读取一个对象；只把目标不存在转换为缺失快照。"""

        normalized = self._identity(identity)
        try:
            value = self._load(normalized)
        except FileNotFoundError:
            return VersionedSnapshot.missing(normalized)

        revision = self._revision_of(value)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
            raise ValueError("snapshot revision extractor must return a positive integer")
        payload = self._serialize(value)
        if not isinstance(payload, bytes):
            raise TypeError("snapshot serializer must return bytes")
        size_bytes = len(payload)
        if size_bytes > self.config.max_item_bytes:
            raise SnapshotReadLimitError("snapshot item exceeds its configured byte limit")
        return VersionedSnapshot(
            identity=normalized,
            state=SnapshotState.FOUND,
            value=value,
            revision=revision,
            source_digest=hashlib.sha256(payload).hexdigest(),
            size_bytes=size_bytes,
        )

    def read_many(self, identities: Iterable[str]) -> SnapshotBatch[T]:
        """去重并按身份稳定排序后执行有界批量读取。"""

        if isinstance(identities, str) or not isinstance(identities, Iterable):
            raise TypeError("identities must be an iterable of strings")
        unique_identities: set[str] = set()
        input_count = 0
        for identity in identities:
            input_count += 1
            if input_count > self.config.max_items:
                raise SnapshotReadLimitError("snapshot batch exceeds its configured item limit")
            unique_identities.add(self._identity(identity))
        normalized = tuple(sorted(unique_identities))

        snapshots: list[VersionedSnapshot[T]] = []
        total_bytes = 0
        for identity in normalized:
            snapshot = self.read(identity)
            total_bytes += snapshot.size_bytes
            if total_bytes > self.config.max_total_bytes:
                raise SnapshotReadLimitError("snapshot batch exceeds its configured total byte limit")
            snapshots.append(snapshot)
        return SnapshotBatch(snapshots=tuple(snapshots), total_bytes=total_bytes)

    @staticmethod
    def _identity(value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("snapshot identity must be a non-empty string without surrounding whitespace")
        return value


__all__ = ["SnapshotReadLimitError", "SnapshotReader"]
