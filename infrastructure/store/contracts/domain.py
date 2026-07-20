"""存储实现识别领域专属对象时使用的最小协议。"""

from __future__ import annotations

from typing import Protocol

from infrastructure.store.model.context.context_object import ContextObject


class ContextDomainClassifier(Protocol):
    """判断 URI 或上下文对象是否由专属领域负责。"""

    def owns_uri(self, uri: str) -> bool: ...

    def owns_object(self, obj: ContextObject) -> bool: ...


class NoContextDomainClassifier:
    """未配置专属领域时使用的空分类器。"""

    def owns_uri(self, uri: str) -> bool:
        del uri
        return False

    def owns_object(self, obj: ContextObject) -> bool:
        del obj
        return False


__all__ = ["ContextDomainClassifier", "NoContextDomainClassifier"]
