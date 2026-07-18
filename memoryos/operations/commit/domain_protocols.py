"""Narrow domain protocols used by the generic operation plane."""

from __future__ import annotations

from typing import Protocol


class ActionPolicy(Protocol):
    """Structural boundary for ActionPolicy-owned mutations."""

    @property
    def uri(self) -> str: ...


__all__ = ["ActionPolicy"]
