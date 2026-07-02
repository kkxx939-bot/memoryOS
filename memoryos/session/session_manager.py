from __future__ import annotations

import json
from pathlib import Path

from .memory.models import utc_now
from .memory.extractor import MemoryOperation, RuleBasedExtractor
from ..storage.memory_store import MemoryStore
from .memory.update_service import MemoryUpdateContext, MemoryUpdateService


class SessionManager:
    def __init__(self, store: MemoryStore, extractor: RuleBasedExtractor | None = None) -> None:
        self.store = store
        self.extractor = extractor or RuleBasedExtractor()
        self.memory_updates = MemoryUpdateService(store)

    def add_message(self, user_id: str, session_id: str, role: str, text: str) -> Path:
        session_dir = self._session_dir(user_id, session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        message = {
            "role": role,
            "text": text,
            "created_at": utc_now(),
        }
        messages_path = session_dir / "messages.jsonl"
        with messages_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(message, ensure_ascii=False) + "\n")
        return messages_path

    def commit(self, user_id: str, session_id: str) -> dict:
        session_dir = self._session_dir(user_id, session_id)
        messages_path = session_dir / "messages.jsonl"
        messages = self._read_messages(messages_path)
        extracted = self.extractor.extract(messages)
        diff = self.memory_updates.apply(
            extracted,
            MemoryUpdateContext(
                user_id=user_id,
                source=f"session:{session_id}",
                diff_id=session_id,
            ),
        )
        diff["session_id"] = session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "memory_diff.json").write_text(
            json.dumps(diff, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_session_layers(session_dir, messages, extracted)
        return diff

    def _session_dir(self, user_id: str, session_id: str) -> Path:
        return self.store.root / "user" / user_id / "sessions" / session_id

    def _read_messages(self, messages_path: Path) -> list[dict[str, str]]:
        if not messages_path.exists():
            return []
        messages = []
        for line in messages_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            messages.append(json.loads(line))
        return messages

    def _operation_record(self, memory: MemoryOperation) -> dict:
        return {
            "action": memory.action,
            "target": memory.target,
            "memory_type": memory.memory_type,
            "title": memory.title,
            "text": memory.text,
            "tags": memory.tags,
            "confidence": memory.confidence,
            "rationale": memory.rationale,
        }

    def _write_session_layers(self, session_dir: Path, messages: list[dict[str, str]], extracted: list[MemoryOperation]) -> None:
        abstract = f"Session with {len(messages)} messages and {len(extracted)} extracted memories."
        overview = [
            "# Session Overview",
            "",
            f"- Messages: {len(messages)}",
            f"- Extracted memories: {len(extracted)}",
        ]
        for memory in extracted:
            overview.append(f"- {memory.action}/{memory.memory_type}: {memory.text}")
        (session_dir / ".abstract.md").write_text(abstract + "\n", encoding="utf-8")
        (session_dir / ".overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
