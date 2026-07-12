"""适配器里的配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentHookConfig:
    root: str
    user_id: str
    adapter_id: str
    agent_name: str
    token_budget: int
    queue_path: str
    flush_mode: str = "stop"
    tenant_id: str = "default"
    allowed_workspace_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _validate_tenant_id(self.tenant_id)
        _validate_user_id(self.user_id)

    @classmethod
    def from_env(cls, adapter_id: str = "codex") -> AgentHookConfig:
        root = os.environ.get("MEMORYOS_ROOT", "./memory-root")
        tenant_id = os.environ.get("MEMORYOS_TENANT_ID", "default")
        user_id = os.environ.get("MEMORYOS_USER_ID", "default")
        _validate_tenant_id(tenant_id)
        _validate_user_id(user_id)
        configured_queue_path = os.environ.get("MEMORYOS_HOOK_QUEUE_PATH")
        queue_path = configured_queue_path or str(_default_queue_path(Path(root), tenant_id, user_id))
        return cls(
            root=root,
            user_id=user_id,
            adapter_id=os.environ.get("MEMORYOS_ADAPTER_ID", adapter_id),
            agent_name=os.environ.get("MEMORYOS_AGENT_NAME", adapter_id),
            token_budget=_env_int("MEMORYOS_TOKEN_BUDGET", 1200),
            queue_path=queue_path,
            flush_mode=_flush_mode(),
            tenant_id=tenant_id,
            allowed_workspace_ids=frozenset(
                item.strip() for item in os.environ.get("MEMORYOS_WORKSPACE_IDS", "").split(",") if item.strip()
            ),
        )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _flush_mode() -> str:
    value = os.environ.get("MEMORYOS_HOOK_FLUSH_MODE", "stop").lower()
    return value if value in {"never", "stop", "immediate"} else "stop"


def _validate_tenant_id(tenant_id: str) -> None:
    if (
        not isinstance(tenant_id, str)
        or not tenant_id.strip()
        or tenant_id in {".", ".."}
        or "/" in tenant_id
        or "\\" in tenant_id
    ):
        raise ValueError("tenant_id must be one safe non-empty path segment")


def _validate_user_id(user_id: str) -> None:
    if (
        not isinstance(user_id, str)
        or not user_id.strip()
        or user_id in {".", ".."}
        or "/" in user_id
        or "\\" in user_id
    ):
        raise ValueError("user_id must be one safe non-empty path segment")


def _default_queue_path(root: Path, tenant_id: str, user_id: str) -> Path:
    if tenant_id == "default" and user_id == "default":
        return root / "queues" / "agent_hooks.jsonl"
    if tenant_id == "default":
        return root / "users" / user_id / "queues" / "agent_hooks.jsonl"
    return root / "tenants" / tenant_id / "users" / user_id / "queues" / "agent_hooks.jsonl"
