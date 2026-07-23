"""加载并严格校验随包发布的六类长期记忆 Schema。"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from typing import Any

import yaml

from memory.schema.model import (
    MemoryFieldRole,
    MemoryFieldSchema,
    MemoryFieldType,
    MemoryMergeStrategy,
    MemoryOperationMode,
    MemorySchemaError,
    MemoryTypeSchema,
)
from memory.tree.model import MemoryAddress, MemoryKind

_SCHEMA_FILES = {
    MemoryKind.PROFILE: "profile.yaml",
    MemoryKind.PREFERENCE: "preferences.yaml",
    MemoryKind.ENTITY: "entities.yaml",
    MemoryKind.TOOL: "tools.yaml",
    MemoryKind.EVENT: "events.yaml",
    MemoryKind.INTENTION: "intentions.yaml",
}
_TYPE_KEYS = {
    "memory_type",
    "description",
    "path_template",
    "markdown_template",
    "operation_mode",
    "fields",
}
_FIELD_KEYS = {"name", "type", "role", "required", "merge", "description"}


class MemorySchemaRegistry:
    """固定六类记忆类型的声明注册表，不约束动态 topic/category/name。"""

    def __init__(self, schemas: tuple[MemoryTypeSchema, ...]) -> None:
        by_kind: dict[MemoryKind, MemoryTypeSchema] = {}
        for schema in schemas:
            if schema.kind in by_kind:
                raise MemorySchemaError(f"duplicate memory schema: {schema.kind.value}")
            by_kind[schema.kind] = schema
        if set(by_kind) != set(MemoryKind):
            missing = sorted(kind.value for kind in set(MemoryKind) - set(by_kind))
            raise MemorySchemaError(f"memory schema registry is incomplete: {missing}")
        self._schemas = tuple(by_kind[kind] for kind in MemoryKind)
        self._by_kind = by_kind

    @classmethod
    def load_default(cls) -> MemorySchemaRegistry:
        definitions = resources.files("memory.schema.definitions")
        schemas = tuple(
            _load_schema(definitions.joinpath(filename).read_text(encoding="utf-8"), filename)
            for _kind, filename in _SCHEMA_FILES.items()
        )
        return cls(schemas)

    def all(self) -> tuple[MemoryTypeSchema, ...]:
        return self._schemas

    def get(self, kind: MemoryKind | str) -> MemoryTypeSchema:
        try:
            return self._by_kind[MemoryKind(kind)]
        except (KeyError, ValueError) as exc:
            raise MemorySchemaError(f"unknown memory type: {kind}") from exc

    def validate(self, kind: MemoryKind | str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.get(kind).validate_payload(payload)

    def address_for(self, kind: MemoryKind | str, payload: Mapping[str, Any]) -> MemoryAddress:
        return self.get(kind).address_for(payload)

    def render_markdown(self, kind: MemoryKind | str, payload: Mapping[str, Any]) -> str:
        return self.get(kind).render_markdown(payload)


def _load_schema(source: str, filename: str) -> MemoryTypeSchema:
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise MemorySchemaError(f"invalid YAML in {filename}") from exc
    payload = _mapping(raw, f"schema {filename}")
    _reject_unknown(payload, _TYPE_KEYS, f"schema {filename}")
    required = _TYPE_KEYS
    missing = required - set(payload)
    if missing:
        raise MemorySchemaError(f"schema {filename} is missing fields: {sorted(missing)}")
    raw_fields = payload["fields"]
    if not isinstance(raw_fields, list):
        raise MemorySchemaError(f"schema {filename} fields must be a list")
    fields = tuple(_load_field(item, filename) for item in raw_fields)
    return MemoryTypeSchema(
        kind=MemoryKind(str(payload["memory_type"])),
        description=_string(payload["description"], "memory type description"),
        path_template=_string(payload["path_template"], "memory path template"),
        markdown_template=_string(payload["markdown_template"], "memory markdown template"),
        operation_mode=MemoryOperationMode(str(payload["operation_mode"])),
        fields=fields,
    )


def _load_field(raw: Any, filename: str) -> MemoryFieldSchema:
    payload = _mapping(raw, f"field in {filename}")
    _reject_unknown(payload, _FIELD_KEYS, f"field in {filename}")
    missing = _FIELD_KEYS - set(payload)
    if missing:
        raise MemorySchemaError(f"field in {filename} is missing keys: {sorted(missing)}")
    return MemoryFieldSchema(
        name=_string(payload["name"], "memory field name"),
        field_type=MemoryFieldType(str(payload["type"])),
        role=MemoryFieldRole(str(payload["role"])),
        required=payload["required"],
        merge_strategy=MemoryMergeStrategy(str(payload["merge"])),
        description=_string(payload["description"], "memory field description"),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MemorySchemaError(f"{label} must be an object with string keys")
    return dict(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MemorySchemaError(f"{label} must be a non-empty string")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise MemorySchemaError(f"{label} contains unsupported keys: {sorted(unknown)}")


__all__ = ["MemorySchemaRegistry"]
