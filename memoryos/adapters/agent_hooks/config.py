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

    @classmethod
    def from_env(cls, adapter_id: str = "codex") -> AgentHookConfig:
        root = os.environ.get("MEMORYOS_ROOT", "./memory-root")
        queue_path = os.environ.get("MEMORYOS_HOOK_QUEUE_PATH") or str(Path(root) / "queues" / "agent_hooks.jsonl")
        return cls(
            root=root,
            user_id=os.environ.get("MEMORYOS_USER_ID", "default"),
            adapter_id=os.environ.get("MEMORYOS_ADAPTER_ID", adapter_id),
            agent_name=os.environ.get("MEMORYOS_AGENT_NAME", adapter_id),
            token_budget=_env_int("MEMORYOS_TOKEN_BUDGET", 1200),
            queue_path=queue_path,
        )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
