"""普通上下文事实源的持久化协议。"""

from __future__ import annotations

from typing import Protocol

from infrastructure.store.model.context.context_object import ContextObject


class SourceStore(Protocol):
    def read_object(self, uri: str) -> ContextObject: ...

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None: ...

    def list_objects(self) -> list[ContextObject]: ...

    def read_content(self, uri: str) -> str: ...

    def write_content(self, uri: str, content: str | bytes) -> None: ...

    def soft_delete(self, uri: str, reason: str) -> None: ...

    def delete_object(self, uri: str) -> None: ...


__all__ = ["SourceStore"]
