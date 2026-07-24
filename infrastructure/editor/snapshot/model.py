"""Editor 在修改领域文档前使用的版本快照模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")
_SHA256_LENGTH = 64


class SnapshotState(str, Enum):
    """目标在读取时是否存在。"""

    FOUND = "found"
    MISSING = "missing"


@dataclass(frozen=True)
class SnapshotReadConfig:
    """一次旧版本读取允许使用的显式资源边界。"""

    max_items: int = 64
    max_item_bytes: int = 256_000
    max_total_bytes: int = 4_000_000

    def __post_init__(self) -> None:
        for name, value in {
            "max_items": self.max_items,
            "max_item_bytes": self.max_item_bytes,
            "max_total_bytes": self.max_total_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_total_bytes < self.max_item_bytes:
            raise ValueError("max_total_bytes must not be smaller than max_item_bytes")


@dataclass(frozen=True)
class VersionedSnapshot(Generic[T]):
    """一个领域对象在某次完整读取后的不可变版本证据。"""

    identity: str
    state: SnapshotState
    value: T | None
    revision: int | None
    source_digest: str | None
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity or self.identity != self.identity.strip():
            raise ValueError("snapshot identity must be a non-empty string without surrounding whitespace")
        state = SnapshotState(self.state)
        object.__setattr__(self, "state", state)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("snapshot size_bytes must be a non-negative integer")

        if state is SnapshotState.MISSING:
            if self.value is not None or self.revision is not None or self.source_digest is not None:
                raise ValueError("missing snapshot cannot contain value, revision, or digest")
            if self.size_bytes != 0:
                raise ValueError("missing snapshot size_bytes must be zero")
            return

        if self.value is None:
            raise ValueError("found snapshot must contain a value")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ValueError("found snapshot revision must be a positive integer")
        if not _is_sha256(self.source_digest):
            raise ValueError("found snapshot source_digest must be a SHA-256 hex digest")

    @property
    def exists(self) -> bool:
        """返回目标在快照时是否存在。"""

        return self.state is SnapshotState.FOUND

    @classmethod
    def missing(cls, identity: str) -> VersionedSnapshot[T]:
        """构造一个明确表示目标不存在的快照。"""

        return cls(
            identity=identity,
            state=SnapshotState.MISSING,
            value=None,
            revision=None,
            source_digest=None,
            size_bytes=0,
        )


@dataclass(frozen=True)
class SnapshotBatch(Generic[T]):
    """按身份稳定排序且没有重复项的一组版本快照。"""

    snapshots: tuple[VersionedSnapshot[T], ...]
    total_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshots, tuple) or any(
            not isinstance(snapshot, VersionedSnapshot) for snapshot in self.snapshots
        ):
            raise TypeError("snapshots must be a tuple of VersionedSnapshot values")
        identities = tuple(snapshot.identity for snapshot in self.snapshots)
        if identities != tuple(sorted(identities)):
            raise ValueError("snapshot batch must be sorted by identity")
        if len(identities) != len(set(identities)):
            raise ValueError("snapshot batch cannot contain duplicate identities")
        expected_total = sum(snapshot.size_bytes for snapshot in self.snapshots)
        if (
            isinstance(self.total_bytes, bool)
            or not isinstance(self.total_bytes, int)
            or self.total_bytes != expected_total
        ):
            raise ValueError("snapshot batch total_bytes does not match its snapshots")

    def get(self, identity: str) -> VersionedSnapshot[T] | None:
        """按规范身份返回快照；不存在时返回 ``None``。"""

        for snapshot in self.snapshots:
            if snapshot.identity == identity:
                return snapshot
        return None


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


__all__ = [
    "SnapshotBatch",
    "SnapshotReadConfig",
    "SnapshotState",
    "VersionedSnapshot",
]
