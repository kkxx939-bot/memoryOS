"""构建本地 Agent Hook 用户、适配器和重试队列配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config import MemoryOSConfig, RuntimeMode
from LLMClient.config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class AgentHookConfig(MemoryOSConfig):
    """Agent Hook 的本地用户、队列和转录读取配置。"""

    user_id: str
    adapter_id: str
    agent_name: str
    queue_path: str
    model: ModelConfig = field(default_factory=ModelConfig)
    flush_mode: str = "stop"
    transcript_roots: tuple[str, ...] = ()
    max_transcript_bytes: int = 20_000_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        _validate_user_id(self.user_id)

    @classmethod
    def from_env(
        cls,
        adapter_id: str = "codex",
        *,
        default_mode: RuntimeMode | str = RuntimeMode.LOCAL,
    ) -> AgentHookConfig:
        common = MemoryOSConfig.from_env(default_mode=default_mode)
        user_id = os.environ.get("MEMORYOS_USER_ID", "local-user")
        _validate_user_id(user_id)
        configured_queue_path = os.environ.get("MEMORYOS_HOOK_QUEUE_PATH")
        queue_path = configured_queue_path or str(_default_queue_path(common.root_path, user_id))
        return cls(
            root=common.root,
            mode=common.mode,
            log_level=common.log_level,
            model=ModelConfig.from_env(),
            user_id=user_id,
            adapter_id=os.environ.get("MEMORYOS_ADAPTER_ID", adapter_id),
            agent_name=os.environ.get("MEMORYOS_AGENT_NAME", adapter_id),
            queue_path=queue_path,
            flush_mode=_flush_mode(),
            transcript_roots=tuple(
                item.strip()
                for item in os.environ.get("MEMORYOS_TRANSCRIPT_ROOTS", "").split(os.pathsep)
                if item.strip()
            ),
            max_transcript_bytes=_env_int("MEMORYOS_MAX_TRANSCRIPT_BYTES", 20_000_000),
        )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _flush_mode() -> str:
    value = os.environ.get("MEMORYOS_HOOK_FLUSH_MODE", "stop").lower()
    return value if value in {"never", "stop", "immediate"} else "stop"


def _validate_user_id(user_id: str) -> None:
    if (
        not isinstance(user_id, str)
        or not user_id.strip()
        or user_id in {".", ".."}
        or "/" in user_id
        or "\\" in user_id
    ):
        raise ValueError("user_id must be one safe non-empty path segment")


def _default_queue_path(root: Path, user_id: str) -> Path:
    return root / "users" / user_id / "queues" / "agent_hooks.jsonl"
