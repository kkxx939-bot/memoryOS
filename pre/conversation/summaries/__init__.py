"""与原始 ConversationBatch 绑定的语义摘要。"""

from pre.conversation.summaries.model import ConversationSummary, ConversationSummaryError
from pre.conversation.summaries.store import ConversationSummaryStore

__all__ = ["ConversationSummary", "ConversationSummaryError", "ConversationSummaryStore"]
