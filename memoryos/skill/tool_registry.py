from __future__ import annotations

from collections.abc import Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[[dict], dict]] = {}

    def register(self, tool_name: str, handler: Callable[[dict], dict]) -> None:
        self._tools[tool_name] = handler

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def execute(self, tool_name: str, payload: dict) -> dict:
        if tool_name not in self._tools:
            raise KeyError(f"Tool is not registered: {tool_name}")
        return self._tools[tool_name](payload)
