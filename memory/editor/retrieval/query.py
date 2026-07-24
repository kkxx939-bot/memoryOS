"""从完整 ConversationSegment 构造有界的旧记忆语义查询。"""

from __future__ import annotations

from foundation.integrity import canonical_json
from memory.editor.retrieval.model import MemoryRetrievalConfig
from pre.conversation import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationSegment,
)


class ConversationSegmentQueryBuilder:
    """保留消息角色和工具身份，并优先使用用户 prompt 作为召回信号。"""

    def __init__(self, config: MemoryRetrievalConfig | None = None) -> None:
        if config is not None and not isinstance(config, MemoryRetrievalConfig):
            raise TypeError("config must be MemoryRetrievalConfig")
        self.config = config or MemoryRetrievalConfig()

    def build(self, segment: ConversationSegment) -> str:
        """从原始片段生成搜索文本；Conversation Summary 不参与此过程。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        primary: list[str] = []
        supporting: list[str] = []
        for message in segment.messages:
            rendered = self._render_message(message)
            if message.role is ConversationMessageRole.PROMPT:
                primary.append(rendered)
            else:
                supporting.append(rendered)
        query = "\n\n".join((*primary, *supporting))
        return self._truncate(query, self.config.max_query_chars)

    def _render_message(self, message: ConversationMessage) -> str:
        content = message.content if isinstance(message.content, str) else canonical_json(message.content)
        limit = self._message_limit(message.role)
        body = self._truncate(content, limit)
        header = f"[{message.sequence}][{message.role.value}]"
        if message.tool_name is not None:
            header += f"[tool={message.tool_name}]"
        if message.tool_call_id is not None:
            header += f"[call={message.tool_call_id}]"
        if message.tool_status is not None:
            header += f"[status={message.tool_status.value}]"
        if message.content_mode is not None:
            header += f"[content={message.content_mode.value}]"
        return f"{header}: {body}"

    def _message_limit(self, role: ConversationMessageRole) -> int:
        if role is ConversationMessageRole.PROMPT:
            return self.config.max_prompt_chars
        if role is ConversationMessageRole.COMPLETION:
            return self.config.max_completion_chars
        return self.config.max_tool_message_chars

    @staticmethod
    def _truncate(value: object, limit: int) -> str:
        normalized = " ".join(str(value or "").split())
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return normalized[:limit]
        return normalized[: limit - 3].rstrip() + "..."


__all__ = ["ConversationSegmentQueryBuilder"]
