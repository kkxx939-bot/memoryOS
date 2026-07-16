"""接口层里的配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from memoryos.api.trusted_context import (
    DEFAULT_AGENT_CAPABILITIES,
    TrustedRequestContext,
    capabilities_from_csv,
    scope_keys_from_csv,
    workspace_ids_from_csv,
)
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode

DEFAULT_AGENT_ADAPTERS = ("codex", "claude_code", "cursor", "windsurf", "cline", "continue", "generic_agent")


@dataclass(frozen=True)
class MCPServerConfig:
    root: str
    user_id: str
    tenant_id: str = "default"
    adapter_id: str = "codex"
    agent_name: str = "codex"
    actor_kind: str = "agent"
    actor_id: str = ""
    capabilities: frozenset[str] = DEFAULT_AGENT_CAPABILITIES
    token_budget: int = 2000
    enable_action_tools: bool = False
    hook_queue_path: str = ""
    log_level: str = "WARNING"
    allowed_adapter_ids: tuple[str, ...] = DEFAULT_AGENT_ADAPTERS
    allowed_workspace_ids: frozenset[str] = frozenset()
    authorized_scope_keys: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls) -> MCPServerConfig:
        root = os.environ.get("MEMORYOS_ROOT", "./memory-root")
        queue_path = os.environ.get("MEMORYOS_HOOK_QUEUE_PATH") or str(Path(root) / "queues" / "agent_hooks.jsonl")
        adapter_id = os.environ.get("MEMORYOS_ADAPTER_ID", "codex")
        allowed = _allowed_adapter_ids(adapter_id)
        return cls(
            root=root,
            user_id=os.environ.get("MEMORYOS_USER_ID", "default"),
            tenant_id=os.environ.get("MEMORYOS_TENANT_ID", "default"),
            adapter_id=adapter_id,
            agent_name=os.environ.get("MEMORYOS_AGENT_NAME", adapter_id),
            actor_kind=os.environ.get("MEMORYOS_ACTOR_KIND", "agent"),
            actor_id=os.environ.get("MEMORYOS_ACTOR_ID", adapter_id),
            capabilities=capabilities_from_csv(os.environ.get("MEMORYOS_MCP_CAPABILITIES")),
            allowed_workspace_ids=workspace_ids_from_csv(os.environ.get("MEMORYOS_WORKSPACE_IDS")),
            authorized_scope_keys=scope_keys_from_csv(os.environ.get("MEMORYOS_AUTHORIZED_SCOPE_KEYS")),
            token_budget=_env_int("MEMORYOS_TOKEN_BUDGET", 2000),
            enable_action_tools=os.environ.get("MEMORYOS_ENABLE_ACTION_TOOLS", "").lower() in {"1", "true", "yes"},
            hook_queue_path=queue_path,
            log_level=os.environ.get("MEMORYOS_LOG_LEVEL", "WARNING"),
            allowed_adapter_ids=allowed,
        )

    def trusted_context(self) -> TrustedRequestContext:
        return TrustedRequestContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            actor_kind=self.actor_kind,
            actor_id=self.actor_id or self.adapter_id,
            capabilities=self.capabilities,
            allowed_workspace_ids=self.allowed_workspace_ids,
            authorized_scope_keys=self.authorized_scope_keys,
        )

    def default_agent_metadata(self) -> ConnectMetadata:
        return ConnectMetadata(
            connect_type=ConnectType.AGENT,
            adapter_id=self.adapter_id,
            run_mode=PipelineMode.CONTEXT_REDUCTION,
            world_domain="digital",
            source_kind="coding_agent",
        )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _allowed_adapter_ids(adapter_id: str) -> tuple[str, ...]:
    configured = [
        item.strip() for item in os.environ.get("MEMORYOS_ALLOWED_ADAPTER_IDS", "").split(",") if item.strip()
    ]
    allowed = [*DEFAULT_AGENT_ADAPTERS, *configured, adapter_id]
    return tuple(dict.fromkeys(allowed))
