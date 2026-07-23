"""ConversationSummary 的独立原子 JSON 存储。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from foundation.integrity import canonical_json
from pre.conversation._files import ConversationFiles
from pre.conversation.summaries.model import ConversationSummary, ConversationSummaryError


class ConversationSummaryStore:
    def __init__(self, root: str | Path) -> None:
        self._files = ConversationFiles(root)

    @property
    def root(self) -> Path:
        return self._files.root

    def path_for(self, conversation_id: str, started_on: date) -> Path:
        return self._files.path_for("summaries", conversation_id, started_on, ".json")

    def write(self, summary: ConversationSummary) -> Path:
        if not isinstance(summary, ConversationSummary):
            raise TypeError("summary must be a ConversationSummary")
        path = self.path_for(summary.conversation_id, summary.started_at.date())
        payload = (canonical_json(summary.to_dict()) + "\n").encode("utf-8")
        return self._files.write(path, payload)

    def read(self, conversation_id: str, started_on: date) -> ConversationSummary:
        path = self.path_for(conversation_id, started_on)
        try:
            raw = json.loads(self._files.read(path).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ConversationSummaryError("conversation summary is not valid UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise ConversationSummaryError("conversation summary is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ConversationSummaryError("conversation summary must be an object")
        summary = ConversationSummary.from_dict(raw)
        if summary.conversation_id != conversation_id or summary.started_at.date() != started_on:
            raise ConversationSummaryError("conversation summary path does not match its content")
        return summary


__all__ = ["ConversationSummaryStore"]
