"""单用户本地运行时身份。

这里不实现登录、租户授权或能力令牌，只保存当前本地用户、插件适配器和工作区。
持久化层继续使用一个固定命名空间，避免把本地存储格式和外部认证概念混在一起。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LOCAL_STORAGE_NAMESPACE = "default"
PRINCIPAL_ONLY_WORKSPACE = "__memoryos_principal_only__"


@dataclass(frozen=True)
class LocalUserContext:
    """一个进程内唯一用户的运行上下文。"""

    user_id: str = "local-user"
    adapter_id: str = "codex"
    workspace_id: str = ""

    def __post_init__(self) -> None:
        for name in ("user_id", "adapter_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"local context requires {name}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "workspace_id", str(self.workspace_id or "").strip())

    @property
    def tenant_id(self) -> str:
        """返回内部固定存储命名空间，不表示多租户身份。"""

        return LOCAL_STORAGE_NAMESPACE

    @property
    def actor_id(self) -> str:
        return self.adapter_id

    @property
    def actor_kind(self) -> str:
        return "local"

    def assert_identity(self, *, user_id: Any = None, tenant_id: Any = None) -> None:
        """阻止同一调用意外混用其他本地用户或存储命名空间。"""

        if user_id is not None and str(user_id) != self.user_id:
            raise ValueError("request user_id does not match the configured local user")
        if tenant_id is not None and str(tenant_id) != LOCAL_STORAGE_NAMESPACE:
            raise ValueError("tenant selection is unavailable in local single-user mode")

    def bind_read_workspace(self, workspace_id: Any = None) -> str:
        requested = str(workspace_id or "").strip()
        return requested or self.workspace_id or PRINCIPAL_ONLY_WORKSPACE

    def bind_write_workspace(self, workspace_id: Any = None) -> str:
        resolved = self.bind_read_workspace(workspace_id)
        if resolved == PRINCIPAL_ONLY_WORKSPACE:
            raise ValueError("a workspace is required for this local write")
        return resolved

    def retrieval_scope_keys(self, *, workspace_id: str | None = None) -> frozenset[str]:
        keys = {f"memoryos:principal:{self.user_id}"}
        selected = str(workspace_id or self.workspace_id).strip()
        if selected and selected != PRINCIPAL_ONLY_WORKSPACE:
            keys.add(f"memoryos:workspace:{selected}")
        return frozenset(keys)


__all__ = ["LOCAL_STORAGE_NAMESPACE", "LocalUserContext", "PRINCIPAL_ONLY_WORKSPACE"]
