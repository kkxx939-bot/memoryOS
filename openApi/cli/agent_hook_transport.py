"""Agent Hook 到 MemoryOS 对外能力的传输组合根。

未配置远程地址时在进程内调用 MCP 工具路由；配置 ``MEMORYOS_BASE_URL`` 时改用
HTTP SDK。Hook 适配器因此不需要知道本地与远程部署差异。
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from LLMClient.config import ModelConfig
from openApi.mcp.config import MCPServerConfig
from openApi.mcp.server import MemoryOSMCPServer
from openApi.sdk.http_client import HTTPMemoryOSClient


class AgentHookConfig(Protocol):
    """Agent Hook 交付传输所需的最小配置视图。"""

    @property
    def root(self) -> str: ...

    @property
    def user_id(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def agent_name(self) -> str: ...

    @property
    def queue_path(self) -> str: ...


class AgentHookTransportClient:
    """向 Agent Hook 提供与 MCP ``call_tool`` 一致的交付接口。"""

    def __init__(self, config: AgentHookConfig) -> None:
        model_config = getattr(config, "model", ModelConfig())
        mcp_config = MCPServerConfig(
            root=config.root,
            model=model_config,
            user_id=config.user_id,
            adapter_id=config.adapter_id,
            agent_name=config.agent_name,
            hook_queue_path=config.queue_path,
        )
        remote = os.environ.get("MEMORYOS_BASE_URL")
        # 本地模式复用 MCP 工具路由，以保证 Hook 与 MCP 客户端拥有相同的校验语义。
        if remote:
            self.server = None
        else:
            # 只有本地 Hook 交付才加载完整 SDK 运行时。
            from openApi.sdk.client import MemoryOSClient

            self.server = MemoryOSMCPServer(
                MemoryOSClient(
                    config.root,
                    user_id=config.user_id,
                    adapter_id=config.adapter_id,
                    model_config=model_config,
                ),
                config=mcp_config,
            )
        self.remote = (
            HTTPMemoryOSClient(remote)
            if remote
            else None
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """把一个 Hook 工具调用转交给远程 HTTP 或进程内 MCP 服务。"""

        if self.remote:
            if name == "memoryos_search_context":
                results = self.remote.search_context(
                    arguments.get("query", ""), **{k: v for k, v in arguments.items() if k != "query"}
                )
                return {"results": results, "trace_id": self.remote.last_recall_trace_id}
            if name == "memoryos_assemble_context":
                return self.remote.assemble_context(
                    arguments.get("query", ""), **{k: v for k, v in arguments.items() if k != "query"}
                )
            if name == "memoryos_commit_session":
                return self.remote.commit_agent_session(**arguments)
            if name == "memoryos_health":
                return self.remote.health()
            if name == "memoryos_read":
                return self.remote.read(str(arguments.get("uri") or ""), layer=str(arguments.get("layer") or "L2"))
            if name == "memoryos_recall_trace":
                return self.remote.recall_trace(str(arguments.get("trace_id") or ""))
            return {
                "error": {
                    "code": "UNSUPPORTED_REMOTE_TOOL",
                    "message": f"Remote tool is not supported: {name}",
                    "retryable": False,
                }
            }
        if self.server is None:
            return {
                "error": {"code": "CLIENT_UNAVAILABLE", "message": "MemoryOS transport unavailable", "retryable": True}
            }
        return self.server.call_tool(name, arguments)


AgentHookMCPClient = AgentHookTransportClient

__all__ = ["AgentHookMCPClient", "AgentHookTransportClient"]
