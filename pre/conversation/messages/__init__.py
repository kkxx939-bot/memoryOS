"""完整、严格角色化的原始会话事实。"""

from pre.conversation.messages.model import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageError,
    ConversationMessageRole,
)
from pre.conversation.messages.projector import (
    ConversationProjectionError,
    SessionArchiveConversationProjector,
)
from pre.conversation.messages.store import ConversationMessageStore

__all__ = [
    "ConversationBatch",
    "ConversationMessage",
    "ConversationMessageError",
    "ConversationMessageRole",
    "ConversationMessageStore",
    "ConversationProjectionError",
    "SessionArchiveConversationProjector",
]
