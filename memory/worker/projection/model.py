"""Markdown Memory 投影 Worker 使用的小型协议和运行结果。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, SupportsIndex, SupportsInt, cast

from infrastructure.store.model.catalog import CatalogRecord

_IntConvertible = str | bytes | bytearray | SupportsInt | SupportsIndex


class CatalogLister(Protocol):
    """Catalog 实现可选提供的有界列表接口。"""

    def __call__(
        self,
        *,
        tenant_id: str,
        filters: Mapping[str, object],
        limit: int,
    ) -> Sequence[CatalogRecord]: ...


class CatalogBatchScanner(Protocol):
    """Catalog 实现可选提供的游标批量扫描接口。"""

    def __call__(
        self,
        *,
        tenant_id: str,
        after_record_key: str,
        filters: Mapping[str, object],
        limit: int,
    ) -> Sequence[CatalogRecord]: ...


def coerce_persisted_int(value: object) -> int:
    """将一个已持久化标量转换为整数，不改变其原始语义。"""

    return int(cast(_IntConvertible, value))


@dataclass(frozen=True)
class MemoryProjectionRun:
    """一次投影队列消费的稳定结果。"""

    processed: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


__all__ = [
    "CatalogBatchScanner",
    "CatalogLister",
    "MemoryProjectionRun",
    "coerce_persisted_int",
]
