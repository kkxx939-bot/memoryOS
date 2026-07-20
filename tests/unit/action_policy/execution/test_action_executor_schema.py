from __future__ import annotations

from policy.action_policy.decision.action_context import ActionContext
from policy.action_policy.decision.result import PolicyDecision
from policy.action_policy.execution.executor import ActionExecutor
from policy.action_policy.execution.tool_registry import ToolRegistry


def _context(temperature: int | None = 24) -> ActionContext:
    resource_metadata: dict[str, object] = {"device_id": "ac"}
    if temperature is not None:
        resource_metadata["temperature"] = temperature
    return ActionContext(
        user_id="u1",
        candidate_actions=["turn_on_ac"],
        packed_context={
            "slices": {
                "resource": {"items": [{"uri": "memoryos://resources/ac", "metadata": resource_metadata}]},
                "skill": {"items": [{"uri": "memoryos://skills/ac", "metadata": {"tool_name": "ac.turn_on", "executable": True}}]},
            }
        },
    )


def _registry() -> ToolRegistry:
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
    return registry


def test_executor_schema_success_and_invalid_args() -> None:
    decision = PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok")

    success = ActionExecutor(_registry()).execute(decision, _context())
    failed = ActionExecutor(_registry()).execute(decision, _context(temperature=None))

    assert success.status == "success"
    assert success.tool_args["device_id"] == "ac"
    assert failed.status == "failed"
    assert failed.reason == "invalid_args"


def test_executor_does_not_execute_non_execute_or_blocked_decisions() -> None:
    executor = ActionExecutor(_registry())

    assert executor.execute(PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="confirm"), _context()).status == "skipped"
    assert executor.execute(PolicyDecision(mode="blocked", allowed=False, action="do_nothing", reason="risk"), _context()).status == "skipped"
