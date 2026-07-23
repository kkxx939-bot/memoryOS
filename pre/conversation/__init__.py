"""Conversation 原始消息与可重建历史过程摘要的数据契约。"""

from pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationMessageSchemaError,
    ConversationSegment,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from pre.conversation.summaries import (
    ConversationSegmentSummary,
    ConversationSummarySchemaError,
)

__all__ = [
    "ConversationBatch",
    "ConversationMessage",
    "ConversationMessageRole",
    "ConversationMessageSchemaError",
    "ConversationSegment",
    "ConversationSegmentSummary",
    "ConversationSummarySchemaError",
    "ConversationToolResultContentMode",
    "ConversationToolResultStatus",
]
