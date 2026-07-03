from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from memoryos.application.memory.extractor import MemoryOperation, RuleBasedExtractor
from memoryos.application.memory.update_service import MemoryUpdateContext, MemoryUpdateService
from memoryos.domain.memory.memory_item import utc_now
from memoryos.infrastructure.repositories.memory_repository import MemoryStore
from memoryos.infrastructure.safety.path_safety import validate_identifier


class MemoryExtractor(Protocol):
    def extract(self, messages: list[dict[str, str]]) -> list[MemoryOperation]:
        ...


class SessionManager:
    def __init__(self, store: MemoryStore, extractor: MemoryExtractor | None = None) -> None:
        self.store = store
        self.extractor = extractor or RuleBasedExtractor()
        self.memory_updates = MemoryUpdateService(store)

    def add_message(self, user_id: str, session_id: str, role: str, text: str) -> Path:
        validate_identifier(user_id, "user_id")
        validate_identifier(session_id, "session_id")
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
        validate_identifier(user_id, "user_id")
        validate_identifier(session_id, "session_id")
        session_dir = self._session_dir(user_id, session_id)
        messages_path = session_dir / "messages.jsonl"
        messages = self._read_messages(messages_path)
        state = self._read_commit_state(session_dir)
        messages_hash = self._messages_hash(messages)
        if state.get("messages_hash") == messages_hash and (session_dir / "memory_diff.json").exists():
            diff = json.loads((session_dir / "memory_diff.json").read_text(encoding="utf-8"))
            diff["idempotent"] = True
            return diff
        committed_count = int(state.get("committed_count", 0) or 0)
        if committed_count > len(messages):
            committed_count = 0
        new_messages = messages[committed_count:]
        extracted = self.extractor.extract(new_messages)
        diff = self.memory_updates.apply(
            extracted,
            MemoryUpdateContext(
                user_id=user_id,
                source=f"session:{session_id}",
                diff_id=session_id,
            ),
        )
        diff["session_id"] = session_id
        diff["committed_message_count"] = len(new_messages)
        diff["total_message_count"] = len(messages)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "memory_diff.json").write_text(
            json.dumps(diff, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_commit_state(
            session_dir,
            {
                "session_id": session_id,
                "messages_hash": messages_hash,
                "committed_count": len(messages),
                "committed_at": utc_now(),
            },
        )
        self._write_session_layers(session_dir, messages, extracted)
        return diff

    def _session_dir(self, user_id: str, session_id: str) -> Path:
        return self.store.root / "user" / user_id / "sessions" / session_id

    def _read_commit_state(self, session_dir: Path) -> dict:
        path = session_dir / "commit_state.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_commit_state(self, session_dir: Path, payload: dict) -> None:
        (session_dir / "commit_state.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _messages_hash(self, messages: list[dict[str, str]]) -> str:
        stable = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

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
