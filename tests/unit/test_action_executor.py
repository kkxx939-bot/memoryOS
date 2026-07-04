from __future__ import annotations

from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.model.prediction_result import PolicyDecision
from memoryos.prediction.pipeline.executor import ActionExecutor
from memoryos.skill.tool_registry import ToolRegistry


def _context(with_skill: bool = True, with_resource: bool = True) -> ActionContext:
    return ActionContext(
        user_id="u1",
        candidate_actions=["turn_on_ac"],
        packed_context={
            "slices": {
                "resource": {"items": [{"uri": "memoryos://resources/ac", "metadata": {"available": True}}] if with_resource else []},
                "skill": {"items": [{"uri": "memoryos://skills/ac", "title": "ac_tool", "metadata": {"tool_name": "ac_tool", "executable": True}}] if with_skill else []},
            }
        },
    )


def test_execute_calls_registered_fake_skill_successfully() -> None:
    registry = ToolRegistry()
    calls = []

    def handler(payload: dict) -> dict:
        calls.append(payload)
        return {"ok": True}

    registry.register("ac_tool", handler)

    result = ActionExecutor(registry).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), _context())

    assert result.status == "success"
    assert result.executed is True
    assert calls


def test_execute_blocks_when_skill_or_resource_is_missing() -> None:
    executor = ActionExecutor()
    decision = PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok")

    assert executor.execute(decision, _context(with_skill=False)).status == "blocked"
    assert executor.execute(decision, _context(with_resource=False)).status == "blocked"


def test_ask_user_does_not_call_tool() -> None:
    registry = ToolRegistry()
    calls = []

    def handler(payload: dict) -> dict:
        calls.append(payload)
        return {"ok": True}

    registry.register("ac_tool", handler)

    result = ActionExecutor(registry).execute(PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="confirm"), _context())

    assert result.status == "skipped"
    assert calls == []
