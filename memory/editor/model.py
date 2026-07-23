"""绑定 Conversation Segment 的结构化长期记忆候选变更。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonicalize
from memory.schema import MemorySchemaError, MemorySchemaRegistry
from memory.tree import MemoryAddress, MemoryKind
from pre.conversation.messages import ConversationSegment
from pre.conversation.messages.model import (
    ConversationMessageSchemaError,
    require_sha256,
)


class MemoryEditSchemaError(ValueError):
    """Memory Edit 来源或候选内容不满足当前数据契约。"""


def _source_identifier(value: object, label: str) -> str:
    try:
        identifier = require_safe_path_segment(value, label)
    except ValueError as exc:
        raise MemoryEditSchemaError(str(exc)) from exc
    if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
        raise MemoryEditSchemaError(f"{label} contains unsafe characters")
    return identifier


def _source_digest(value: object) -> str:
    try:
        return require_sha256(value, "source_message_digest")
    except ConversationMessageSchemaError as exc:
        raise MemoryEditSchemaError(str(exc)) from exc


@lru_cache(maxsize=1)
def _default_registry() -> MemorySchemaRegistry:
    return MemorySchemaRegistry.load_default()


@dataclass(frozen=True)
class MemoryEditSource:
    """由可信运行时绑定的不可变 Conversation Segment 来源。"""

    conversation_id: str
    segment_id: str
    source_message_digest: str

    SCHEMA_VERSION = "memory_edit_source_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            _source_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(self, "segment_id", _source_identifier(self.segment_id, "segment_id"))
        object.__setattr__(self, "source_message_digest", _source_digest(self.source_message_digest))

    @classmethod
    def from_segment(cls, segment: ConversationSegment) -> MemoryEditSource:
        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        return cls(
            conversation_id=segment.conversation_id,
            segment_id=segment.segment_id,
            source_message_digest=segment.digest,
        )

    def require_matches(self, segment: ConversationSegment) -> None:
        """确认候选批次只绑定声明的不可变原始消息片段。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        expected = (segment.conversation_id, segment.segment_id, segment.digest)
        actual = (self.conversation_id, self.segment_id, self.source_message_digest)
        if actual != expected:
            raise MemoryEditSchemaError("memory edit source does not match its conversation segment")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": self.SCHEMA_VERSION,
                "conversation_id": self.conversation_id,
                "segment_id": self.segment_id,
                "source_message_digest": self.source_message_digest,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryEditSource:
        if not isinstance(payload, Mapping):
            raise MemoryEditSchemaError("memory edit source must be an object")
        allowed = {
            "schema_version",
            "conversation_id",
            "segment_id",
            "source_message_digest",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise MemoryEditSchemaError(
                f"memory edit source contains unknown fields: {sorted(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise MemoryEditSchemaError("unsupported memory edit source schema")
        required = allowed - {"schema_version"}
        missing = required - set(payload)
        if missing:
            raise MemoryEditSchemaError(f"memory edit source is missing fields: {sorted(missing)}")
        return cls(
            conversation_id=payload["conversation_id"],
            segment_id=payload["segment_id"],
            source_message_digest=payload["source_message_digest"],
        )


_BATCH_FIELDS: tuple[tuple[str, MemoryKind], ...] = (
    ("profile", MemoryKind.PROFILE),
    ("preferences", MemoryKind.PREFERENCE),
    ("entities", MemoryKind.ENTITY),
    ("tools", MemoryKind.TOOL),
    ("events", MemoryKind.EVENT),
    ("intentions", MemoryKind.INTENTION),
)


@dataclass(frozen=True)
class MemoryEditBatch:
    """一个 Segment 解析得到的六类候选记忆；不表达文件写入或删除。"""

    source: MemoryEditSource
    profile: tuple[Mapping[str, Any], ...] = ()
    preferences: tuple[Mapping[str, Any], ...] = ()
    entities: tuple[Mapping[str, Any], ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    intentions: tuple[Mapping[str, Any], ...] = ()

    SCHEMA_VERSION = "memory_edit_batch_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.source, MemoryEditSource):
            raise TypeError("source must be a MemoryEditSource")

        registry = _default_registry()
        seen_addresses: set[MemoryAddress] = set()
        for field_name, kind in _BATCH_FIELDS:
            items = self._validate_items(
                field_name,
                kind,
                getattr(self, field_name),
                registry,
                seen_addresses,
            )
            object.__setattr__(self, field_name, items)

    @staticmethod
    def _validate_items(
        field_name: str,
        kind: MemoryKind,
        raw_items: object,
        registry: MemorySchemaRegistry,
        seen_addresses: set[MemoryAddress],
    ) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(raw_items, tuple | list):
            raise MemoryEditSchemaError(f"memory edit field {field_name} must be a list")

        items: list[Mapping[str, Any]] = []
        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, Mapping):
                raise MemoryEditSchemaError(
                    f"memory edit field {field_name}[{index}] must be an object"
                )
            try:
                normalized = registry.validate(kind, raw_item)
                address = registry.address_for(kind, normalized)
            except MemorySchemaError as exc:
                raise MemoryEditSchemaError(
                    f"memory edit field {field_name}[{index}] is invalid: {exc}"
                ) from exc
            if address in seen_addresses:
                raise MemoryEditSchemaError(
                    f"memory edit batch contains duplicate target: {kind.value}:{address.name}"
                )
            seen_addresses.add(address)
            items.append(MappingProxyType(dict(normalized)))
        return tuple(items)

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, field_name) for field_name, _kind in _BATCH_FIELDS)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def require_matches_source(self, segment: ConversationSegment) -> None:
        self.source.require_matches(segment)

    def items_for(self, kind: MemoryKind | str) -> tuple[Mapping[str, Any], ...]:
        selected = MemoryKind(kind)
        for field_name, field_kind in _BATCH_FIELDS:
            if field_kind is selected:
                return getattr(self, field_name)
        raise MemoryEditSchemaError(f"unsupported memory kind: {selected.value}")

    def entries(self) -> tuple[tuple[MemoryKind, Mapping[str, Any]], ...]:
        return tuple(
            (kind, item)
            for field_name, kind in _BATCH_FIELDS
            for item in getattr(self, field_name)
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "source": self.source.to_dict(),
        }
        for field_name, _kind in _BATCH_FIELDS:
            payload[field_name] = list(getattr(self, field_name))
        return canonicalize(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryEditBatch:
        if not isinstance(payload, Mapping):
            raise MemoryEditSchemaError("memory edit batch must be an object")
        field_names = {field_name for field_name, _kind in _BATCH_FIELDS}
        allowed = {"schema_version", "source", *field_names}
        unknown = set(payload) - allowed
        if unknown:
            raise MemoryEditSchemaError(
                f"memory edit batch contains unknown fields: {sorted(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise MemoryEditSchemaError("unsupported memory edit batch schema")
        required = {"source", *field_names}
        missing = required - set(payload)
        if missing:
            raise MemoryEditSchemaError(f"memory edit batch is missing fields: {sorted(missing)}")
        source_payload = payload["source"]
        if not isinstance(source_payload, Mapping):
            raise MemoryEditSchemaError("memory edit batch source must be an object")
        batch_fields = {
            field_name: cls._items_from_payload(payload, field_name) for field_name in field_names
        }
        return cls(
            source=MemoryEditSource.from_dict(source_payload),
            profile=batch_fields["profile"],
            preferences=batch_fields["preferences"],
            entities=batch_fields["entities"],
            tools=batch_fields["tools"],
            events=batch_fields["events"],
            intentions=batch_fields["intentions"],
        )

    @staticmethod
    def _items_from_payload(
        payload: Mapping[str, Any],
        field_name: str,
    ) -> tuple[Mapping[str, Any], ...]:
        raw_items = payload[field_name]
        if not isinstance(raw_items, tuple | list):
            raise MemoryEditSchemaError(f"memory edit field {field_name} must be a list")
        if any(not isinstance(item, Mapping) for item in raw_items):
            raise MemoryEditSchemaError(f"memory edit field {field_name} contains a non-object")
        return tuple(raw_items)


__all__ = ["MemoryEditBatch", "MemoryEditSchemaError", "MemoryEditSource"]
