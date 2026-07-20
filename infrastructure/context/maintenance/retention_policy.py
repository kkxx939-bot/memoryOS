"""上下文 Serving 层保留策略及其运行结果。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from infrastructure.store.model.catalog import CatalogRecord


class RetentionCatalogStore(Protocol):
    """保留策略需要的最小 Catalog 存储能力。"""

    def scan_catalog_batch(
        self,
        *,
        tenant_id: str,
        after_record_key: str = "",
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
    ) -> list[CatalogRecord]: ...

    def get_catalog(self, record_key: str, *, tenant_id: str) -> CatalogRecord | None: ...

    def upsert_catalog(self, record: CatalogRecord, *, tenant_id: str) -> None: ...

    def enqueue_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str = "",
        source_revision: int = 0,
        tombstone_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def mark_tombstone_applied(self, tombstone_id: str, *, tenant_id: str) -> dict[str, Any] | None: ...

    def mark_tombstone_failed(
        self,
        tombstone_id: str,
        error: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None: ...

    def gc_orphan_paths(self, *, tenant_id: str, limit: int = 256) -> int: ...

    def gc_applied_tombstones(
        self,
        *,
        tenant_id: str,
        updated_before: str,
        limit: int = 256,
    ) -> int: ...


class VectorDeleteStore(Protocol):
    """保留策略清理向量时需要的最小能力。"""

    def delete_vector(self, uri: str) -> None: ...

    def get_vector_metadata(self, uri: str) -> dict[str, Any] | None: ...


class TombstoneProcessor(Protocol):
    """处理已经耐久记录的投影删除任务。"""

    def process_pending(self, *, tenant_id: str, limit: int = 100) -> Any: ...


@dataclass(frozen=True)
class RetentionPolicy:
    """只调整可重建 Serving 数据，不删除不可变证据。"""

    hot_for: timedelta = timedelta(days=7)
    warm_for: timedelta = timedelta(days=30)
    cold_for: timedelta = timedelta(days=180)
    tombstone_journal_for: timedelta = timedelta(days=30)
    vectorize_warm: bool = False
    batch_size: int = 256
    max_compaction_sources: int = 256

    def __post_init__(self) -> None:
        if not timedelta(0) <= self.hot_for <= self.warm_for <= self.cold_for:
            raise ValueError("retention tier durations must be monotonically increasing")
        if self.tombstone_journal_for <= timedelta(0):
            raise ValueError("tombstone_journal_for must be positive")
        if not 1 <= self.batch_size <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        if not 1 <= self.max_compaction_sources <= 1_000:
            raise ValueError("max_compaction_sources must be between 1 and 1000")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> RetentionPolicy:
        """从显式配置构造经过校验的保留策略。"""

        values = dict(config or {})
        allowed = {
            "hot_days",
            "warm_days",
            "cold_days",
            "tombstone_journal_days",
            "vectorize_warm",
            "batch_size",
            "max_compaction_sources",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unsupported retention policy fields: {', '.join(unknown)}")

        def duration(name: str, default: timedelta) -> timedelta:
            raw = values.get(name)
            if raw is None:
                return default
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise ValueError(f"{name} must be a finite number of days")
            days = float(raw)
            if days != days or days in {float("inf"), float("-inf")}:
                raise ValueError(f"{name} must be a finite number of days")
            return timedelta(days=days)

        def integer(name: str, default: int) -> int:
            raw = values.get(name, default)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"{name} must be an integer")
            return raw

        vectorize_warm = values.get("vectorize_warm", False)
        if not isinstance(vectorize_warm, bool):
            raise ValueError("vectorize_warm must be a boolean")

        defaults = cls()
        return cls(
            hot_for=duration("hot_days", defaults.hot_for),
            warm_for=duration("warm_days", defaults.warm_for),
            cold_for=duration("cold_days", defaults.cold_for),
            tombstone_journal_for=duration(
                "tombstone_journal_days",
                defaults.tombstone_journal_for,
            ),
            vectorize_warm=vectorize_warm,
            batch_size=integer("batch_size", defaults.batch_size),
            max_compaction_sources=integer("max_compaction_sources", defaults.max_compaction_sources),
        )


@dataclass(frozen=True)
class RetentionRunResult:
    scanned: int = 0
    tier_changes: int = 0
    stale_projections: int = 0
    vectors_deleted: int = 0
    orphan_paths_deleted: int = 0
    tombstones_deleted: int = 0
    tombstones_enqueued: int = 0
    tombstones_applied: int = 0
    tombstones_failed: int = 0


__all__ = [
    "RetentionCatalogStore",
    "RetentionPolicy",
    "RetentionRunResult",
    "TombstoneProcessor",
    "VectorDeleteStore",
]
