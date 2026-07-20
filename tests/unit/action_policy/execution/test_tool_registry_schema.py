from __future__ import annotations

import pytest

from policy.action_policy.execution.tool_registry import ToolRegistry


def test_tool_registry_validates_schema_and_executes() -> None:
    registry = ToolRegistry()
    registry.register(
        "ac.turn_on",
        lambda args: {"ok": True, **args},
        input_schema={
            "type": "object",
            "required": ["device_id", "temperature"],
            "properties": {"device_id": {"type": "string"}, "temperature": {"type": "number"}},
        },
    )

    assert registry.validate_args("ac.turn_on", {"device_id": "ac", "temperature": 24})
    assert registry.execute("ac.turn_on", {"device_id": "ac", "temperature": 24})["ok"] is True
    with pytest.raises(ValueError):
        registry.validate_args("ac.turn_on", {"device_id": "ac"})
