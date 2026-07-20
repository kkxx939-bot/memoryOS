"""事务测试使用的显式依赖组合。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.operation import build_operation_control_stores
from transaction.commit.operation_committer import OperationCommitter


def build_test_operation_committer(
    source_store: SourceStore,
    index_store: IndexStore,
    root: str | Path,
    *args: Any,
    **kwargs: Any,
) -> OperationCommitter:
    """使用文件控制存储创建真实事务提交器，不替换领域或存储行为。"""

    tenant_id = str(kwargs.get("tenant_id") or getattr(source_store, "tenant_id", "default"))
    artifact_root = Path(root) if tenant_id == "default" else Path(root) / "tenants" / tenant_id
    return OperationCommitter(
        source_store,
        index_store,
        str(root),
        build_operation_control_stores(artifact_root),
        *args,
        **kwargs,
    )


__all__ = ["build_test_operation_committer"]
