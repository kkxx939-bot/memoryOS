"""文件系统存储实现的延迟导出入口。

延迟加载使会话证据归档可以独立使用，不会因为包级导入而初始化完整的
Markdown 记忆文档领域。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infrastructure.store.filesystem.memory_document_store import (
        FileSystemMemoryDocumentStore as FileSystemMemoryDocumentStore,
    )
    from infrastructure.store.filesystem.session_archive import (
        SessionArchiveStore as SessionArchiveStore,
    )
    from infrastructure.store.filesystem.source_store import (
        BundleIntegrityError as BundleIntegrityError,
    )
    from infrastructure.store.filesystem.source_store import (
        FileSystemSourceStore as FileSystemSourceStore,
    )

_PUBLIC_ATTRS = {
    "BundleIntegrityError": (
        "infrastructure.store.filesystem.source_store",
        "BundleIntegrityError",
    ),
    "FileSystemMemoryDocumentStore": (
        "infrastructure.store.filesystem.memory_document_store",
        "FileSystemMemoryDocumentStore",
    ),
    "FileSystemSourceStore": (
        "infrastructure.store.filesystem.source_store",
        "FileSystemSourceStore",
    ),
    "SessionArchiveStore": (
        "infrastructure.store.filesystem.session_archive",
        "SessionArchiveStore",
    ),
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
