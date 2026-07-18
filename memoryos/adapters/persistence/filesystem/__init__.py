"""Lazy filesystem adapter exports.

Importing the immutable Session archive adapter must not initialize the
Markdown memory domain.  Package-level exports therefore resolve only when a
caller explicitly asks for the corresponding Source or document adapter.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoryos.adapters.persistence.filesystem.memory_document_store import (
        FileSystemMemoryDocumentStore as FileSystemMemoryDocumentStore,
    )
    from memoryos.adapters.persistence.filesystem.source_store import (
        BundleIntegrityError as BundleIntegrityError,
    )
    from memoryos.adapters.persistence.filesystem.source_store import (
        FileSystemSourceStore as FileSystemSourceStore,
    )

_PUBLIC_ATTRS = {
    "BundleIntegrityError": (
        "memoryos.adapters.persistence.filesystem.source_store",
        "BundleIntegrityError",
    ),
    "FileSystemMemoryDocumentStore": (
        "memoryos.adapters.persistence.filesystem.memory_document_store",
        "FileSystemMemoryDocumentStore",
    ),
    "FileSystemSourceStore": (
        "memoryos.adapters.persistence.filesystem.source_store",
        "FileSystemSourceStore",
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
