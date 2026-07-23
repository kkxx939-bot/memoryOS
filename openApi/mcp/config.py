"""MCP 服务端的本地单用户环境配置。

配置在进程启动时确定本地用户、Agent 适配器和默认工作区；动作工具是否开放仍由
独立开关控制，但这里不实现登录、多租户或访问授权。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from config import MemoryOSConfig, RuntimeMode
from foundation.identity import LocalUserContext
from LLMClient.config import ModelConfig
from pre.connect import ConnectMetadata, ConnectType, PipelineMode

DEFAULT_AGENT_ADAPTERS = ("codex", "claude_code", "cursor", "windsurf", "cline", "continue", "generic_agent")


@dataclass(frozen=True, kw_only=True)
class MCPServerConfig(MemoryOSConfig):
    """本地单用户 MCP 进程配置。"""

    user_id: str
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter_id: str = "codex"
    agent_name: str = "codex"
    workspace_id: str = ""
    enable_action_tools: bool = False
    hook_queue_path: str = ""
    allowed_adapter_ids: tuple[str, ...] = DEFAULT_AGENT_ADAPTERS

    @classmethod
    def from_env(
        cls,
        *,
        default_mode: RuntimeMode | str = RuntimeMode.LOCAL,
    ) -> MCPServerConfig:
        """从进程环境构造一次服务端配置。"""

        common = MemoryOSConfig.from_env(default_mode=default_mode)
        queue_path = os.environ.get("MEMORYOS_HOOK_QUEUE_PATH") or str(
            common.root_path / "queues" / "agent_hooks.jsonl"
        )
        adapter_id = os.environ.get("MEMORYOS_ADAPTER_ID", "codex")
        allowed = _allowed_adapter_ids(adapter_id)
        return cls(
            root=common.root,
            mode=common.mode,
            log_level=common.log_level,
            model=ModelConfig.from_env(),
            user_id=os.environ.get("MEMORYOS_USER_ID", "local-user"),
            adapter_id=adapter_id,
            agent_name=os.environ.get("MEMORYOS_AGENT_NAME", adapter_id),
            workspace_id=os.environ.get("MEMORYOS_WORKSPACE_ID", ""),
            enable_action_tools=os.environ.get("MEMORYOS_ENABLE_ACTION_TOOLS", "").lower() in {"1", "true", "yes"},
            hook_queue_path=queue_path,
            allowed_adapter_ids=allowed,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")

    def local_context(self) -> LocalUserContext:
        """构造当前 MCP 进程唯一的本地用户上下文。"""

        return LocalUserContext(
            user_id=self.user_id,
            adapter_id=self.adapter_id,
            workspace_id=self.workspace_id,
        )

    def default_agent_metadata(self) -> ConnectMetadata:
        return ConnectMetadata(
            connect_type=ConnectType.AGENT,
            adapter_id=self.adapter_id,
            run_mode=PipelineMode.CONTEXT_REDUCTION,
            world_domain="digital",
            source_kind="coding_agent",
        )


def _allowed_adapter_ids(adapter_id: str) -> tuple[str, ...]:
    configured = [
        item.strip() for item in os.environ.get("MEMORYOS_ALLOWED_ADAPTER_IDS", "").split(",") if item.strip()
    ]
    allowed = [*DEFAULT_AGENT_ADAPTERS, *configured, adapter_id]
    return tuple(dict.fromkeys(allowed))
