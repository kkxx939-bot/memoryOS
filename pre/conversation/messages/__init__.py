"""Conversation 原始消息与归档片段 Schema。"""

from pre.conversation.messages.model import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationMessageSchemaError,
    ConversationSegment,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)

__all__ = [
    "ConversationBatch",
    "ConversationMessage",
    "ConversationMessageRole",
    "ConversationMessageSchemaError",
    "ConversationSegment",
    "ConversationToolResultContentMode",
    "ConversationToolResultStatus",
]
