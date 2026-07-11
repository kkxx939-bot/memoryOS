"""技能系统里的工具注册表。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[[dict], dict]] = {}
        self._schemas: dict[str, dict] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool_name: str,
        handler: Callable[[dict], dict],
        input_schema: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._tools[tool_name] = handler
        self._schemas[tool_name] = input_schema or {}
        self._metadata[tool_name] = dict(metadata or {})

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def validate_args(self, tool_name: str, args: dict) -> bool:
        if tool_name not in self._tools:
            raise KeyError(f"Tool is not registered: {tool_name}")
        schema = self._schemas.get(tool_name) or {}
        required = schema.get("required", []) if isinstance(schema, dict) else []
        missing = [field for field in required if field not in args or args[field] in {None, ""}]
        if missing:
            raise ValueError(f"missing required args: {', '.join(missing)}")
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for field, spec in properties.items():
            if field not in args or not isinstance(spec, dict):
                continue
            expected = spec.get("type")
            if expected and not self._matches_type(args[field], str(expected)):
                raise ValueError(f"invalid type for {field}: expected {expected}")
        return True

    def execute(self, tool_name: str, args: dict, dry_run: bool = False) -> dict:
        if tool_name not in self._tools:
            raise KeyError(f"Tool is not registered: {tool_name}")
        self.validate_args(tool_name, args)
        if dry_run:
            return {"dry_run": True, "tool_name": tool_name, "args": args}
        return self._tools[tool_name](args)

    def metadata(self, tool_name: str) -> dict[str, Any]:
        if tool_name not in self._tools:
            raise KeyError(f"Tool is not registered: {tool_name}")
        return dict(self._metadata.get(tool_name, {}))

    def _matches_type(self, value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        return True
