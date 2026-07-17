"""Stable, lazily resolved session exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_ATTRS = {
    "SessionArchiveStore": (
        "memoryos.adapters.persistence.filesystem.session_archive",
        "SessionArchiveStore",
    ),
    "SessionCommitService": ("memoryos.application.session.commit_service", "SessionCommitService"),
    "SessionArchive": ("memoryos.contextdb.session.session_model", "SessionArchive"),
    "SessionCommitResult": ("memoryos.contextdb.session.session_model", "SessionCommitResult"),
}


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = list(_PUBLIC_ATTRS)
