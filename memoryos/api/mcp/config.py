from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode

DEFAULT_AGENT_ADAPTERS = ("codex", "claude_code", "cursor", "generic_agent")


@dataclass(frozen=True)
class MCPServerConfig:
    root: str
    user_id: str
    adapter_id: str = "codex"
    agent_name: str = "codex"
    token_budget: int = 2000
    enable_action_tools: bool = False
    hook_queue_path: str = ""
    log_level: str = "WARNING"
    allowed_adapter_ids: tuple[str, ...] = DEFAULT_AGENT_ADAPTERS

    @classmethod
    def from_env(cls) -> MCPServerConfig:
        root = os.environ.get("MEMORYOS_ROOT", "./memory-root")
        queue_path = os.environ.get("MEMORYOS_HOOK_QUEUE_PATH") or str(
            Path(root) / "queues" / "agent_hooks.jsonl"
        )
        return cls(
            root=root,
            user_id=os.environ.get("MEMORYOS_USER_ID", "default"),
            adapter_id=os.environ.get("MEMORYOS_ADAPTER_ID", "codex"),
            agent_name=os.environ.get("MEMORYOS_AGENT_NAME", os.environ.get("MEMORYOS_ADAPTER_ID", "codex")),
            token_budget=_env_int("MEMORYOS_TOKEN_BUDGET", 2000),
            enable_action_tools=os.environ.get("MEMORYOS_ENABLE_ACTION_TOOLS", "").lower() in {"1", "true", "yes"},
            hook_queue_path=queue_path,
            log_level=os.environ.get("MEMORYOS_LOG_LEVEL", "WARNING"),
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
