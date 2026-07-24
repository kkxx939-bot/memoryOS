"""声明式长期记忆内容 Schema 模型。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import PurePosixPath
from string import Formatter
from typing import Any

from memory.model import MemoryAddress, MemoryKind

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_PATH_TEMPLATES = {
    MemoryKind.PROFILE: "profile.md",
    MemoryKind.PREFERENCE: "preferences/{topic}.md",
    MemoryKind.ENTITY: "entities/{category}/{name}.md",
    MemoryKind.TOOL: "tools/{tool_name}.md",
    MemoryKind.EVENT: "events/{event_date:%Y}/{event_date:%m}/{event_date:%d}/{event_name}.md",
    MemoryKind.INTENTION: "intentions/{intent_name}.md",
}


class MemorySchemaError(ValueError):
    """Schema 声明或内容字段不满足当前记忆树约束。"""


class MemoryFieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"


class MemoryFieldRole(str, Enum):
    ADDRESS = "address"
    CONTENT = "content"


class MemoryMergeStrategy(str, Enum):
    IMMUTABLE = "immutable"
    PATCH = "patch"
    REPLACE = "replace"


class MemoryOperationMode(str, Enum):
    UPSERT = "upsert"
    ADD_ONLY = "add_only"


def _template_fields(template: str, label: str) -> frozenset[str]:
    fields: set[str] = set()
    try:
        parsed = Formatter().parse(template)
        for _literal, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not _FIELD_NAME.fullmatch(field_name):
                raise MemorySchemaError(f"{label} contains an invalid field placeholder")
            if conversion or (format_spec and field_name != "event_date"):
                raise MemorySchemaError(f"{label} contains an unsupported field formatter")
            if format_spec and format_spec not in {"%Y", "%m", "%d"}:
                raise MemorySchemaError(f"{label} contains an unsupported date formatter")
            fields.add(field_name)
    except ValueError as exc:
        raise MemorySchemaError(f"{label} is not a valid format template") from exc
    return frozenset(fields)


@dataclass(frozen=True)
class MemoryFieldSchema:
    name: str
    field_type: MemoryFieldType
    role: MemoryFieldRole
    required: bool
    merge_strategy: MemoryMergeStrategy
    description: str
    allowed_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _FIELD_NAME.fullmatch(self.name):
            raise MemorySchemaError("memory schema field name must use lowercase snake_case")
        object.__setattr__(self, "field_type", MemoryFieldType(self.field_type))
        object.__setattr__(self, "role", MemoryFieldRole(self.role))
        object.__setattr__(self, "merge_strategy", MemoryMergeStrategy(self.merge_strategy))
        if isinstance(self.allowed_values, str):
            raise MemorySchemaError("memory schema allowed_values must be a tuple of strings")
        allowed_values = tuple(self.allowed_values)
        object.__setattr__(self, "allowed_values", allowed_values)
        if not isinstance(self.required, bool):
            raise MemorySchemaError("memory schema field required must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise MemorySchemaError("memory schema field description must be non-empty")
        if any(not isinstance(value, str) or not value.strip() for value in allowed_values):
            raise MemorySchemaError("memory schema allowed_values must contain non-empty strings")
        if len(allowed_values) != len(set(allowed_values)):
            raise MemorySchemaError("memory schema allowed_values must be unique")
        if allowed_values and self.field_type is not MemoryFieldType.STRING:
            raise MemorySchemaError("memory schema allowed_values currently supports string fields only")
        if self.role is MemoryFieldRole.ADDRESS and (
            not self.required or self.merge_strategy is not MemoryMergeStrategy.IMMUTABLE
        ):
            raise MemorySchemaError("memory address fields must be required and immutable")


@dataclass(frozen=True)
class MemoryTypeSchema:
    kind: MemoryKind
    description: str
    path_template: str
    markdown_template: str
    operation_mode: MemoryOperationMode
    fields: tuple[MemoryFieldSchema, ...]
    min_non_empty_content_fields: int = 0
    omit_empty_sections: bool = False

    def __post_init__(self) -> None:
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "operation_mode", MemoryOperationMode(self.operation_mode))
        object.__setattr__(self, "fields", tuple(self.fields))
        if (
            isinstance(self.min_non_empty_content_fields, bool)
            or not isinstance(self.min_non_empty_content_fields, int)
            or self.min_non_empty_content_fields < 0
        ):
            raise MemorySchemaError(
                "memory min_non_empty_content_fields must be a non-negative integer"
            )
        if not isinstance(self.omit_empty_sections, bool):
            raise MemorySchemaError("memory omit_empty_sections must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise MemorySchemaError("memory type description must be non-empty")
        if self.path_template != _CANONICAL_PATH_TEMPLATES[kind]:
            raise MemorySchemaError(f"{kind.value} schema path does not match the confirmed memory tree")
        path = PurePosixPath(self.path_template)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise MemorySchemaError("memory schema path template is unsafe")
        if not isinstance(self.markdown_template, str) or not self.markdown_template:
            raise MemorySchemaError("memory markdown template must be non-empty")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise MemorySchemaError("memory schema field names must be unique")
        if not self.fields or not any(field.role is MemoryFieldRole.CONTENT for field in self.fields):
            raise MemorySchemaError("memory schema must declare at least one content field")

        declared = frozenset(names)
        address_fields = frozenset(
            field.name for field in self.fields if field.role is MemoryFieldRole.ADDRESS
        )
        path_fields = _template_fields(self.path_template, "memory path template")
        markdown_fields = _template_fields(self.markdown_template, "memory markdown template")
        content_fields = frozenset(
            field.name for field in self.fields if field.role is MemoryFieldRole.CONTENT
        )
        if self.min_non_empty_content_fields > len(content_fields):
            raise MemorySchemaError(
                "memory min_non_empty_content_fields exceeds the declared content fields"
            )
        if path_fields != address_fields:
            raise MemorySchemaError("memory path placeholders must exactly match address fields")
        if not markdown_fields <= declared:
            raise MemorySchemaError("memory markdown template references undeclared fields")
        if not content_fields <= markdown_fields:
            raise MemorySchemaError("every content field must appear in the Markdown template")

    @property
    def field_map(self) -> dict[str, MemoryFieldSchema]:
        return {field.name: field for field in self.fields}

    @property
    def address_fields(self) -> tuple[MemoryFieldSchema, ...]:
        return tuple(field for field in self.fields if field.role is MemoryFieldRole.ADDRESS)

    @property
    def content_fields(self) -> tuple[MemoryFieldSchema, ...]:
        return tuple(field for field in self.fields if field.role is MemoryFieldRole.CONTENT)

    def validate_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """严格校验单条记忆字段；不容错转换未知结构或忽略额外字段。"""

        if not isinstance(payload, Mapping):
            raise MemorySchemaError("memory payload must be an object")
        unknown = set(payload) - set(self.field_map)
        if unknown:
            raise MemorySchemaError(f"memory payload contains unknown fields: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for field in self.fields:
            if field.name not in payload or payload[field.name] is None:
                if field.required:
                    raise MemorySchemaError(f"memory payload is missing required field: {field.name}")
                continue
            value = self._validate_value(field, payload[field.name])
            if field.allowed_values and value not in field.allowed_values:
                raise MemorySchemaError(
                    f"memory field {field.name} must be one of {list(field.allowed_values)}"
                )
            normalized[field.name] = value
        present_content_fields = sum(
            1
            for field in self.content_fields
            if field.name in normalized and self._is_non_empty(normalized[field.name])
        )
        if present_content_fields < self.min_non_empty_content_fields:
            raise MemorySchemaError(
                "memory payload does not contain enough non-empty content fields"
            )
        self._address_from_normalized(normalized)
        return normalized

    def address_for(self, payload: Mapping[str, Any]) -> MemoryAddress:
        """从通过 Schema 的地址字段构造记忆树地址。"""

        return self._address_from_normalized(self.validate_payload(payload))

    def render_markdown(self, payload: Mapping[str, Any]) -> str:
        """校验字段后按 Schema 生成 Markdown，不写入记忆树。"""

        normalized = self.validate_payload(payload)
        rendered_fields = {
            field.name: self._render_value(normalized.get(field.name)) for field in self.fields
        }
        template = (
            self._without_empty_sections(normalized)
            if self.omit_empty_sections
            else self.markdown_template
        )
        try:
            rendered = template.format_map(rendered_fields)
            if self.omit_empty_sections:
                return rendered.rstrip() + "\n"
            return rendered
        except (KeyError, ValueError) as exc:  # pragma: no cover - 构造时已验证模板。
            raise MemorySchemaError("memory markdown rendering failed") from exc

    def _without_empty_sections(self, payload: Mapping[str, Any]) -> str:
        """删除全部引用字段为空的二级 Markdown 模板区块。"""

        lines = self.markdown_template.splitlines(keepends=True)
        starts = [index for index, line in enumerate(lines) if line.startswith("## ")]
        if not starts:
            return self.markdown_template
        boundaries = [*starts, len(lines)]
        rendered = lines[: starts[0]]
        for index, start in enumerate(starts):
            block = lines[start : boundaries[index + 1]]
            block_source = "".join(block)
            fields = _template_fields(block_source, "memory Markdown section")
            if fields and not any(self._is_non_empty(payload.get(name)) for name in fields):
                continue
            rendered.extend(block)
        return "".join(rendered)

    def _address_from_normalized(self, payload: Mapping[str, Any]) -> MemoryAddress:
        try:
            if self.kind is MemoryKind.PROFILE:
                return MemoryAddress.profile()
            if self.kind is MemoryKind.PREFERENCE:
                return MemoryAddress.preference(str(payload["topic"]))
            if self.kind is MemoryKind.ENTITY:
                return MemoryAddress.entity(str(payload["category"]), str(payload["name"]))
            if self.kind is MemoryKind.TOOL:
                return MemoryAddress.tool(str(payload["tool_name"]))
            if self.kind is MemoryKind.EVENT:
                event_date = payload["event_date"]
                if not isinstance(event_date, date):  # pragma: no cover - 字段校验保证。
                    raise MemorySchemaError("event_date is not normalized")
                return MemoryAddress.event(event_date, str(payload["event_name"]))
            return MemoryAddress.intention(str(payload["intent_name"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise MemorySchemaError("memory payload contains an invalid tree address") from exc

    @staticmethod
    def _validate_value(field: MemoryFieldSchema, value: Any) -> Any:
        if field.field_type is MemoryFieldType.STRING:
            if not isinstance(value, str):
                raise MemorySchemaError(f"memory field {field.name} must be a string")
            if field.required and not value.strip():
                raise MemorySchemaError(f"memory field {field.name} must be non-empty")
            return value
        if field.field_type is MemoryFieldType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise MemorySchemaError(f"memory field {field.name} must be an integer")
            return value
        if field.field_type is MemoryFieldType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise MemorySchemaError(f"memory field {field.name} must be a number")
            return value
        if field.field_type is MemoryFieldType.BOOLEAN:
            if not isinstance(value, bool):
                raise MemorySchemaError(f"memory field {field.name} must be a boolean")
            return value
        if isinstance(value, datetime):
            raise MemorySchemaError(f"memory field {field.name} must be a calendar date")
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise MemorySchemaError(f"memory field {field.name} must use YYYY-MM-DD") from exc
        raise MemorySchemaError(f"memory field {field.name} must be a date")

    @staticmethod
    def _render_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _is_non_empty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True


__all__ = [
    "MemoryFieldRole",
    "MemoryFieldSchema",
    "MemoryFieldType",
    "MemoryMergeStrategy",
    "MemoryOperationMode",
    "MemorySchemaError",
    "MemoryTypeSchema",
]
