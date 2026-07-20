"""从受信任 RuntimeLayout 枚举有限的记忆 Owner。"""

from __future__ import annotations

from infrastructure.store.memory.layout import RuntimeLayout
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


def discover_owner_ids(layout: RuntimeLayout, *, limit: int) -> tuple[str, ...]:
    """在 tenant 根内枚举 Owner，拒绝符号链接和无界遍历。"""

    if limit <= 0:
        raise ValueError("memory owner enumeration limit must be positive")
    candidates: set[str] = set()
    source_root = layout.root / "tenants" / layout.tenant_id / "users"
    control_root = layout.tenant_root / "system" / "memory-documents"
    for parent in (source_root, control_root):
        if not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError("memory owner root is unsafe")
        for child in parent.iterdir():
            if child.name == "sealed-proposals":
                continue
            if child.is_symlink() or not child.is_dir():
                raise RuntimeError("memory owner entry is unsafe")
            candidates.add(MemoryDocumentPathPolicy.trusted_segment(child.name, "owner_user_id"))
            if len(candidates) > limit:
                raise RuntimeError("document owner enumeration exceeded its bound")
    return tuple(sorted(candidates))


def bounded_owner_ids(
    layout: RuntimeLayout,
    tenant_id: str,
    limit: int,
) -> tuple[str, ...]:
    """只允许在当前运行时绑定的 tenant 内枚举 Owner。"""

    if tenant_id != layout.tenant_id:
        raise PermissionError("document owner enumeration crossed the runtime tenant")
    return discover_owner_ids(layout, limit=limit)


__all__ = ["bounded_owner_ids", "discover_owner_ids"]
