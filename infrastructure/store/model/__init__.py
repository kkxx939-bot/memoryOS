"""存储层共享数据模型的延迟导出入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_ATTRS = {
    "CatalogProjectionStatus": ("infrastructure.store.model.catalog", "CatalogProjectionStatus"),
    "CatalogRecord": ("infrastructure.store.model.catalog", "CatalogRecord"),
    "CatalogRecordKind": ("infrastructure.store.model.catalog", "CatalogRecordKind"),
    "ServingTier": ("infrastructure.store.model.catalog", "ServingTier"),
}


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_PUBLIC_ATTRS)
