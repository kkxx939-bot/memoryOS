"""L2 记忆文档的严格规范编解码器。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, NoReturn, Protocol

from memory.document.model import MemoryDocument, MemoryDocumentMetadata
from memory.model import MemoryAddress, MemoryKind

_MARKER = "\n<!-- M2BOS_MEMORY_FIELDS\n"
_FOOTER = "\n-->\n"
_METADATA_KEYS = {"memory_type", "revision", "created_at", "updated_at", "fields"}


class MemoryDocumentIntegrityError(ValueError):
    """L2 文档缺少结构字段，或正文、路径与字段不一致。"""


class _MemorySchemaRegistry(Protocol):
    def validate(self, kind: MemoryKind | str, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def address_for(
        self,
        kind: MemoryKind | str,
        payload: Mapping[str, Any],
    ) -> MemoryAddress: ...

    def render_markdown(
        self,
        kind: MemoryKind | str,
        payload: Mapping[str, Any],
    ) -> str: ...


class MemoryDocumentCodec:
    """使用同一份 Schema 构造、序列化并验证 L2 文档。"""

    def __init__(self, registry: _MemorySchemaRegistry) -> None:
        required = ("validate", "address_for", "render_markdown")
        if any(not callable(getattr(registry, name, None)) for name in required):
            raise TypeError("registry must implement the memory Schema registry contract")
        self.registry = registry

    def build(
        self,
        kind: MemoryKind | str,
        payload: Mapping[str, Any],
        *,
        metadata: MemoryDocumentMetadata,
    ) -> MemoryDocument:
        """从结构字段生成地址和可读正文，不接受调用者提供的路径或正文。"""

        if not isinstance(metadata, MemoryDocumentMetadata):
            raise TypeError("metadata must be MemoryDocumentMetadata")
        normalized_kind = MemoryKind(kind)
        normalized = self.registry.validate(normalized_kind, payload)
        address = self.registry.address_for(normalized_kind, normalized)
        markdown_body = self.registry.render_markdown(normalized_kind, normalized)
        if _MARKER in markdown_body:
            raise MemoryDocumentIntegrityError(
                "memory Markdown body contains the reserved metadata marker"
            )
        return MemoryDocument(
            kind=normalized_kind,
            address=address,
            metadata=metadata,
            fields=normalized,
            markdown_body=markdown_body,
        )

    def encode(self, document: MemoryDocument) -> str:
        """重新校验文档后输出唯一规范的 Markdown 物理格式。"""

        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be a MemoryDocument")
        canonical = self.build(
            document.kind,
            document.fields,
            metadata=document.metadata,
        )
        if canonical.address != document.address:
            raise MemoryDocumentIntegrityError("memory document address is not canonical")
        if canonical.markdown_body != document.markdown_body:
            raise MemoryDocumentIntegrityError("memory document body is not canonical")
        fields = {
            name: self._json_value(value) for name, value in canonical.fields.items()
        }
        metadata = {
            "memory_type": canonical.kind.value,
            "revision": canonical.metadata.revision,
            "created_at": self._timestamp(canonical.metadata.created_at),
            "updated_at": self._timestamp(canonical.metadata.updated_at),
            "fields": fields,
        }
        metadata_json = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        # HTML 注释不能包含双连字符；只转义 JSON 字符串中的该序列。
        metadata_json = metadata_json.replace("--", "\\u002d\\u002d")
        return f"{canonical.markdown_body}{_MARKER}{metadata_json}{_FOOTER}"

    def decode(self, raw: str, *, expected_address: MemoryAddress) -> MemoryDocument:
        """解析并验证唯一末尾注释、Schema、真实地址和正文规范性。"""

        if not isinstance(raw, str):
            raise TypeError("raw memory document must be a string")
        if not isinstance(expected_address, MemoryAddress):
            raise TypeError("expected_address must be a MemoryAddress")
        if raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
            raise MemoryDocumentIntegrityError(
                "memory document must contain one terminal M2BOS_MEMORY_FIELDS comment"
            )
        markdown_body, _separator, metadata_with_footer = raw.partition(_MARKER)
        metadata_source = metadata_with_footer[: -len(_FOOTER)]
        try:
            metadata = json.loads(
                metadata_source,
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_json_constant,
            )
        except (json.JSONDecodeError, MemoryDocumentIntegrityError) as exc:
            raise MemoryDocumentIntegrityError(
                "memory document metadata is not strict JSON"
            ) from exc
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise MemoryDocumentIntegrityError("memory document metadata has an invalid shape")
        raw_kind = metadata["memory_type"]
        raw_revision = metadata["revision"]
        raw_created_at = metadata["created_at"]
        raw_updated_at = metadata["updated_at"]
        raw_fields = metadata["fields"]
        if not isinstance(raw_kind, str):
            raise MemoryDocumentIntegrityError("memory document type must be a string")
        if not isinstance(raw_fields, dict) or any(
            not isinstance(name, str) for name in raw_fields
        ):
            raise MemoryDocumentIntegrityError("memory document fields must be an object")
        try:
            system_metadata = MemoryDocumentMetadata(
                revision=raw_revision,
                created_at=self._parse_timestamp(raw_created_at, "created_at"),
                updated_at=self._parse_timestamp(raw_updated_at, "updated_at"),
            )
            document = self.build(
                MemoryKind(raw_kind),
                raw_fields,
                metadata=system_metadata,
            )
        except (TypeError, ValueError) as exc:
            raise MemoryDocumentIntegrityError(
                "memory document fields do not satisfy their Schema"
            ) from exc
        if document.address != expected_address:
            raise MemoryDocumentIntegrityError(
                "memory document fields do not match the physical tree address"
            )
        if document.markdown_body != markdown_body:
            raise MemoryDocumentIntegrityError(
                "memory document body does not match its structured fields"
            )
        if self.encode(document) != raw:
            raise MemoryDocumentIntegrityError("memory document is not canonically serialized")
        return document

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MemoryDocumentIntegrityError(
                    "memory document metadata contains a duplicate JSON key"
                )
            result[key] = value
        return result

    @staticmethod
    def _reject_json_constant(value: str) -> NoReturn:
        raise MemoryDocumentIntegrityError(
            f"memory document metadata contains an invalid JSON constant: {value}"
        )

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: object, field_name: str) -> datetime:
        if not isinstance(value, str):
            raise MemoryDocumentIntegrityError(
                f"memory document {field_name} must be a timestamp string"
            )
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MemoryDocumentIntegrityError(
                f"memory document {field_name} is not a valid ISO timestamp"
            ) from exc

    @classmethod
    def _json_value(cls, value: Any) -> str | int | float | bool:
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, bool | str | int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise MemoryDocumentIntegrityError(
            "memory document contains a field that cannot be serialized"
        )


__all__ = ["MemoryDocumentCodec", "MemoryDocumentIntegrityError"]
