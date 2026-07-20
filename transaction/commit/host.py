"""事务组件共享的组合宿主类型。"""

from __future__ import annotations

from typing import Any, Protocol


class OperationTransactionHost(Protocol):
    """允许拆分后的事务组件通过同一提交器协作。

    具体状态由 ``OperationCommitter`` 组合。这里的动态属性只服务静态类型检查；
    运行时缺少组件或状态时仍抛出 ``AttributeError``，不会静默降级。
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


__all__ = ["OperationTransactionHost"]
