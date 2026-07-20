"""Agent 平台 Hook 接入层。

这里负责把 Codex、Claude Code、Cursor 等平台事件规范化为 MemoryOS 会话事件，
并通过由交付入口注入的 transport 完成上下文召回和会话提交。
"""

from agent_hook.events import AgentHookEvent
from agent_hook.queue import PendingQueue
from sanitization import sanitize_payload

__all__ = ["AgentHookEvent", "PendingQueue", "sanitize_payload"]
