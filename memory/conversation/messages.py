"""Conversation live 消息追加与不可变 history 封存。"""

from __future__ import annotations

import json
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from foundation.integrity import canonical_json
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    read_regular_bytes,
)
from memory.conversation.layout import ConversationAddress, ConversationLayout
from pre.conversation.messages.model import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationMessageSchemaError,
    ConversationSegment,
)


class ConversationJournalError(ValueError):
    """Conversation live 或 history 数据不完整、不合法或无法安全处理。"""


class ConversationWriteConflictError(ConversationJournalError):
    """已有会话事实与本次追加或封存请求冲突。"""


class ConversationAppendStatus(str, Enum):
    CREATED = "created"
    EXTENDED = "extended"
    UNCHANGED = "unchanged"


class ConversationSealStatus(str, Enum):
    CREATED = "created"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ConversationAppendResult:
    status: ConversationAppendStatus
    appended_count: int
    live: ConversationBatch | None


@dataclass(frozen=True)
class ConversationSealResult:
    status: ConversationSealStatus
    segment: ConversationSegment
    live: ConversationBatch | None


@dataclass(frozen=True)
class _HistoryReference:
    segment_id: str
    start_sequence: int
    end_sequence: int
    path: Path


class ConversationMessageJournal:
    """按 Conversation 粒度串行化 live/history 文件操作。"""

    _MAX_FILE_BYTES = 64 * 1024 * 1024
    _MAX_HISTORY_FILES = 10_000

    def __init__(
        self,
        root: str | Path,
        path_lock: PathLock,
        *,
        lock_ttl_seconds: int = 30,
    ) -> None:
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be a PathLock")
        ttl = int(lock_ttl_seconds)
        if ttl <= 0:
            raise ValueError("lock_ttl_seconds must be positive")
        self.layout = ConversationLayout(root)
        self.path_lock = path_lock
        self.lock_ttl_seconds = ttl

    def append(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
    ) -> ConversationAppendResult:
        """幂等追加一个连续消息批次到 live.jsonl。"""

        self._require_batch(address, batch)
        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.lock_ttl_seconds,
        ) as guard:
            with guard.fenced():
                live_messages = list(self._read_live_messages(address))
                latest = self._read_latest_segment(address)
                live_messages = self._recover_archived_prefix(address, latest, live_messages)
                self._require_tail_continuity(latest, live_messages)
                known = self._known_tail(latest, live_messages)
                expected_next = self._expected_next_sequence(latest, live_messages)

                known_by_sequence = {message.sequence: message for message in known}
                known_ids = {message.message_id: message.sequence for message in known}
                known_tool_calls = {
                    message.tool_call_id: message.sequence
                    for message in known
                    if message.role is ConversationMessageRole.TOOL_CALL
                }
                unseen: list[ConversationMessage] = []
                for message in batch.messages:
                    if message.sequence < expected_next:
                        existing = known_by_sequence.get(message.sequence)
                        if existing is None:
                            raise ConversationWriteConflictError(
                                "append replay predates the retained conversation tail"
                            )
                        if existing.to_dict() != message.to_dict():
                            raise ConversationWriteConflictError(
                                "append replay conflicts with an existing message sequence"
                            )
                        continue
                    required_sequence = expected_next + len(unseen)
                    if message.sequence != required_sequence:
                        raise ConversationWriteConflictError(
                            "append would create a gap in the global message sequence"
                        )
                    existing_sequence = known_ids.get(message.message_id)
                    if existing_sequence is not None and existing_sequence != message.sequence:
                        raise ConversationWriteConflictError(
                            "message_id is already bound to another sequence"
                        )
                    if message.role is ConversationMessageRole.TOOL_CALL:
                        existing_call_sequence = known_tool_calls.get(message.tool_call_id)
                        if (
                            existing_call_sequence is not None
                            and existing_call_sequence != message.sequence
                        ):
                            raise ConversationWriteConflictError(
                                "tool_call_id is already bound to another sequence"
                            )
                    unseen.append(message)

                if not unseen:
                    return ConversationAppendResult(
                        status=ConversationAppendStatus.UNCHANGED,
                        appended_count=0,
                        live=self._batch_or_none(address, live_messages),
                    )

                updated_live = tuple([*live_messages, *unseen])
                encoded = self._encode_messages(updated_live)
                self._require_write_bound(encoded)
                atomic_replace_bytes(
                    self.layout.live_path(address),
                    encoded,
                    artifact_root=self.layout.root,
                )
                status = (
                    ConversationAppendStatus.CREATED
                    if not known
                    else ConversationAppendStatus.EXTENDED
                )
                return ConversationAppendResult(
                    status=status,
                    appended_count=len(unseen),
                    live=ConversationBatch(address.conversation_id, updated_live),
                )

    def read_live(self, address: ConversationAddress) -> ConversationBatch | None:
        """读取一致的 live 快照，并完成可证明安全的中断封存恢复。"""

        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.lock_ttl_seconds,
        ) as guard:
            with guard.fenced():
                live_messages = list(self._read_live_messages(address))
                latest = self._read_latest_segment(address)
                live_messages = self._recover_archived_prefix(address, latest, live_messages)
                self._require_tail_continuity(latest, live_messages)
                return self._batch_or_none(address, live_messages)

    def seal(
        self,
        address: ConversationAddress,
        *,
        through_sequence: int,
    ) -> ConversationSealResult:
        """先耐久创建 history，再从 live 原子移除相同前缀。"""

        if isinstance(through_sequence, bool) or not isinstance(through_sequence, int):
            raise TypeError("through_sequence must be an integer")
        if through_sequence < 0:
            raise ValueError("through_sequence must be non-negative")
        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.lock_ttl_seconds,
        ) as guard:
            with guard.fenced():
                live_messages = list(self._read_live_messages(address))
                latest = self._read_latest_segment(address)
                live_messages = self._recover_archived_prefix(address, latest, live_messages)
                self._require_tail_continuity(latest, live_messages)

                if not live_messages or through_sequence < live_messages[0].sequence:
                    if latest is not None and latest.end_sequence == through_sequence:
                        return ConversationSealResult(
                            status=ConversationSealStatus.UNCHANGED,
                            segment=latest,
                            live=self._batch_or_none(address, live_messages),
                        )
                    raise ConversationWriteConflictError(
                        "seal boundary does not select an unarchived live prefix"
                    )
                if through_sequence > live_messages[-1].sequence:
                    raise ConversationWriteConflictError("seal boundary exceeds the live message range")

                split_index = through_sequence - live_messages[0].sequence + 1
                archived_messages = tuple(live_messages[:split_index])
                retained_messages = tuple(live_messages[split_index:])
                if not archived_messages or archived_messages[-1].sequence != through_sequence:
                    raise ConversationWriteConflictError("seal boundary is not present in live messages")
                if latest is not None and archived_messages[0].sequence != latest.end_sequence + 1:
                    raise ConversationWriteConflictError(
                        "sealed segment would not continue the latest history sequence"
                    )

                segment_id = self.layout.segment_id(
                    archived_messages[0].sequence,
                    archived_messages[-1].sequence,
                )
                segment = ConversationSegment(
                    conversation_id=address.conversation_id,
                    segment_id=segment_id,
                    messages=archived_messages,
                )
                encoded_history = self._encode_messages(segment.messages)
                self._require_write_bound(encoded_history)
                try:
                    created = atomic_create_bytes(
                        self.layout.history_path(address, segment_id),
                        encoded_history,
                        artifact_root=self.layout.root,
                    )
                except ImmutableArtifactConflictError as exc:
                    raise ConversationWriteConflictError(
                        "history segment identity conflicts with different bytes"
                    ) from exc

                encoded_live = self._encode_messages(retained_messages)
                self._require_write_bound(encoded_live)
                atomic_replace_bytes(
                    self.layout.live_path(address),
                    encoded_live,
                    artifact_root=self.layout.root,
                )
                return ConversationSealResult(
                    status=(
                        ConversationSealStatus.CREATED
                        if created
                        else ConversationSealStatus.UNCHANGED
                    ),
                    segment=segment,
                    live=self._batch_or_none(address, retained_messages),
                )

    def read_segment(
        self,
        address: ConversationAddress,
        segment_id: str,
    ) -> ConversationSegment:
        """读取一个已经封存且不可变的 history 片段。"""

        start_sequence, end_sequence = self.layout.segment_range(segment_id)
        path = self.layout.history_path(address, segment_id)
        messages = self._read_messages(path, missing_ok=False)
        segment = ConversationSegment(address.conversation_id, segment_id, messages)
        if (
            segment.start_sequence != start_sequence
            or segment.end_sequence != end_sequence
        ):
            raise ConversationJournalError("history filename range does not match its messages")
        return segment

    def _read_live_messages(self, address: ConversationAddress) -> tuple[ConversationMessage, ...]:
        return self._read_messages(self.layout.live_path(address), missing_ok=True)

    def _read_messages(
        self,
        path: Path,
        *,
        missing_ok: bool,
    ) -> tuple[ConversationMessage, ...]:
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.layout.root,
                max_bytes=self._MAX_FILE_BYTES,
            )
        except FileNotFoundError:
            if missing_ok:
                return ()
            raise
        try:
            source = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversationJournalError("conversation JSONL is not valid UTF-8") from exc
        if not source:
            return ()
        messages: list[ConversationMessage] = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            if not line:
                raise ConversationJournalError(
                    f"conversation JSONL contains an empty line at {line_number}"
                )
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversationJournalError(
                    f"conversation JSONL is invalid at line {line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise ConversationJournalError(
                    f"conversation JSONL line {line_number} must be an object"
                )
            try:
                messages.append(ConversationMessage.from_dict(raw))
            except ConversationMessageSchemaError as exc:
                raise ConversationJournalError(
                    f"conversation JSONL message is invalid at line {line_number}"
                ) from exc
        try:
            return ConversationBatch("read-validation", tuple(messages)).messages
        except ConversationMessageSchemaError as exc:
            raise ConversationJournalError("conversation JSONL messages are not contiguous") from exc

    def _read_latest_segment(self, address: ConversationAddress) -> ConversationSegment | None:
        references = self._history_references(address)
        if not references:
            return None
        latest = references[-1]
        return self.read_segment(address, latest.segment_id)

    def _history_references(self, address: ConversationAddress) -> tuple[_HistoryReference, ...]:
        directory = self.layout.history_directory(address)
        if directory.is_symlink():
            raise ConversationJournalError("history directory cannot be a symbolic link")
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise ConversationJournalError("history path is not a directory")
        references: list[_HistoryReference] = []
        for child in directory.iterdir():
            if child.is_symlink():
                raise ConversationJournalError("history cannot contain symbolic links")
            if child.name.startswith("."):
                if child.is_file() and child.name.endswith(".tmp"):
                    continue
                raise ConversationJournalError("history contains an unsupported hidden entry")
            try:
                metadata = child.stat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode) or child.suffix != ".jsonl":
                raise ConversationJournalError("history may contain only segment JSONL files")
            segment_id = child.stem
            start_sequence, end_sequence = self.layout.segment_range(segment_id)
            references.append(
                _HistoryReference(segment_id, start_sequence, end_sequence, child)
            )
            if len(references) > self._MAX_HISTORY_FILES:
                raise ConversationJournalError("history file count exceeds its enumeration bound")
        references.sort(key=lambda item: (item.start_sequence, item.end_sequence))
        for previous, current in zip(references, references[1:], strict=False):
            if current.start_sequence <= previous.end_sequence:
                raise ConversationJournalError("history segment ranges overlap")
        return tuple(references)

    def _recover_archived_prefix(
        self,
        address: ConversationAddress,
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        if latest is None or not live_messages:
            return live_messages
        if live_messages[0].sequence > latest.end_sequence:
            return live_messages
        if live_messages[0].sequence < latest.start_sequence:
            raise ConversationJournalError("live messages precede the latest history segment")

        archived_by_sequence = {message.sequence: message for message in latest.messages}
        overlap_count = 0
        for message in live_messages:
            if message.sequence > latest.end_sequence:
                break
            archived = archived_by_sequence.get(message.sequence)
            if archived is None or archived.to_dict() != message.to_dict():
                raise ConversationWriteConflictError(
                    "live/history overlap contains different message bytes"
                )
            overlap_count += 1
        if overlap_count == 0:
            return live_messages
        retained = live_messages[overlap_count:]
        encoded = self._encode_messages(retained)
        atomic_replace_bytes(
            self.layout.live_path(address),
            encoded,
            artifact_root=self.layout.root,
        )
        return retained

    @staticmethod
    def _known_tail(
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
    ) -> tuple[ConversationMessage, ...]:
        archived = latest.messages if latest is not None else ()
        return tuple([*archived, *live_messages])

    @staticmethod
    def _require_tail_continuity(
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
    ) -> None:
        if not live_messages:
            return
        expected_start = latest.end_sequence + 1 if latest is not None else 0
        if live_messages[0].sequence != expected_start:
            raise ConversationJournalError("live messages do not continue the retained history tail")

    @staticmethod
    def _expected_next_sequence(
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
    ) -> int:
        if live_messages:
            return live_messages[-1].sequence + 1
        if latest is not None:
            return latest.end_sequence + 1
        return 0

    @staticmethod
    def _encode_messages(messages: Sequence[ConversationMessage]) -> bytes:
        return "".join(canonical_json(message.to_dict()) + "\n" for message in messages).encode(
            "utf-8"
        )

    def _require_write_bound(self, encoded: bytes) -> None:
        if len(encoded) > self._MAX_FILE_BYTES:
            raise ConversationJournalError("conversation JSONL exceeds its hard safety bound")

    @staticmethod
    def _require_batch(address: ConversationAddress, batch: ConversationBatch) -> None:
        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be a ConversationBatch")
        if address.conversation_id != batch.conversation_id:
            raise ConversationWriteConflictError(
                "conversation address does not match the appended batch"
            )

    @staticmethod
    def _batch_or_none(
        address: ConversationAddress,
        messages: tuple[ConversationMessage, ...] | list[ConversationMessage],
    ) -> ConversationBatch | None:
        resolved = tuple(messages)
        return ConversationBatch(address.conversation_id, resolved) if resolved else None


__all__ = [
    "ConversationAppendResult",
    "ConversationAppendStatus",
    "ConversationJournalError",
    "ConversationMessageJournal",
    "ConversationSealResult",
    "ConversationSealStatus",
    "ConversationWriteConflictError",
]
