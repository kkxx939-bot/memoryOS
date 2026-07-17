"""Narrow domain values consumed by the generic commit facade."""

from __future__ import annotations

from typing import Any, Protocol


class AliasRegistry(Protocol):
    def resolve(self, namespace: str, value: Any) -> str: ...


class PendingMemoryProposal(Protocol):
    uri: str


class ActionPolicy(Protocol):
    uri: str


__all__ = ["ActionPolicy", "AliasRegistry", "PendingMemoryProposal"]
