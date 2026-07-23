"""ConversationBatch 的原子 JSONL 存储。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pre.conversation._files import ConversationFiles
from pre.conversation.messages.model import ConversationBatch, ConversationMessageError


class ConversationMessageStore:
    def __init__(self, root: str | Path) -> None:
        self._files = ConversationFiles(root)

    @property
    def root(self) -> Path:
        return self._files.root

    def path_for(self, conversation_id: str, started_on: date) -> Path:
        return self._files.path_for("messages", conversation_id, started_on, ".jsonl")

    def write(self, batch: ConversationBatch) -> Path:
        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be a ConversationBatch")
        path = self.path_for(batch.conversation_id, batch.started_at.date())
        return self._files.write(path, batch.to_jsonl().encode("utf-8"))

    def read(self, conversation_id: str, started_on: date) -> ConversationBatch:
        path = self.path_for(conversation_id, started_on)
        try:
            source = self._files.read(path).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversationMessageError("conversation messages are not valid UTF-8") from exc
        batch = ConversationBatch.from_jsonl(source)
        if batch.conversation_id != conversation_id or batch.started_at.date() != started_on:
            raise ConversationMessageError("conversation message path does not match its content")
        return batch


__all__ = ["ConversationMessageStore"]
