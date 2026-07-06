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


def _context_items(resources: list[dict], skills: list[dict]) -> ActionContext:
    return ActionContext(
        user_id="u1",
        candidate_actions=["turn_on_ac"],
        packed_context={"slices": {"resource": {"items": resources}, "skill": {"items": skills}}},
    )


def _skill(
    uri: str,
    *,
    tool_name: str = "ac_tool",
    top_level_tool_name: str | None = None,
    action: str | None = None,
    supported_actions: list[str] | str | None = None,
    default_args: dict | None = None,
) -> dict:
    metadata = {"tool_name": tool_name, "executable": True}
    if action is not None:
        metadata["action"] = action
    if supported_actions is not None:
        metadata["supported_actions"] = supported_actions
    if default_args is not None:
        metadata["default_args"] = default_args
    item = {"uri": uri, "title": tool_name, "metadata": metadata}
    if top_level_tool_name is not None:
        item["tool_name"] = top_level_tool_name
    return item


def _resource(
    uri: str,
    *,
    action: str | None = None,
    supported_actions: list[str] | str | None = None,
    tool_name: str | None = None,
    top_level_tool_name: str | None = None,
    tool_args: dict | None = None,
    device_id: str | None = None,
) -> dict:
    metadata: dict[str, object] = {"available": True}
    if action is not None:
        metadata["action"] = action
    if supported_actions is not None:
        metadata["supported_actions"] = supported_actions
    if tool_name is not None:
        metadata["tool_name"] = tool_name
    if tool_args is not None:
        metadata["tool_args"] = tool_args
    if device_id is not None:
        metadata["device_id"] = device_id
    item = {"uri": uri, "metadata": metadata}
    if top_level_tool_name is not None:
        item["tool_name"] = top_level_tool_name
    return item


def _registry(calls: list[dict]) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(payload: dict) -> dict:
        calls.append(payload)
        return {"ok": True, **payload}

    registry.register("ac_tool", handler)
    return registry


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


def test_execute_selects_only_resource_supporting_action() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[
            _resource("memoryos://resources/fan", supported_actions=["turn_on_fan"], device_id="fan"),
            _resource("memoryos://resources/ac", supported_actions=["turn_on_ac"], device_id="ac"),
        ],
        skills=[_skill("memoryos://skills/ac")],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "success"
    assert result.resource_uris == ["memoryos://resources/ac"]
    assert calls[0]["device_id"] == "ac"


def test_execute_selects_only_skill_supporting_action() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[_resource("memoryos://resources/ac", device_id="ac")],
        skills=[
            _skill("memoryos://skills/fan", supported_actions=["turn_on_fan"]),
            _skill("memoryos://skills/ac", supported_actions=["turn_on_ac"]),
        ],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "success"
    assert result.skill_uris == ["memoryos://skills/ac"]
    assert calls


def test_execute_blocks_when_skill_and_resource_tool_names_do_not_match() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[_resource("memoryos://resources/ac", supported_actions=["turn_on_ac"], tool_name="fan_tool")],
        skills=[_skill("memoryos://skills/ac", supported_actions=["turn_on_ac"], tool_name="ac_tool")],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.executed is False
    assert result.reason == "resource/skill tool_name mismatch"
    assert calls == []


def test_execute_blocks_when_top_level_tool_names_do_not_match() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[
            _resource(
                "memoryos://resources/ac",
                supported_actions=["turn_on_ac"],
                top_level_tool_name="fan_tool",
            )
        ],
        skills=[
            _skill(
                "memoryos://skills/ac",
                supported_actions=["turn_on_ac"],
                top_level_tool_name="ac_tool",
            )
        ],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.reason == "resource/skill tool_name mismatch"
    assert calls == []


def test_execute_uses_resource_tool_name_when_skill_has_no_tool_name() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[
            _resource(
                "memoryos://resources/ac",
                supported_actions=["turn_on_ac"],
                top_level_tool_name="ac_tool",
                device_id="ac",
            )
        ],
        skills=[
            {
                "uri": "memoryos://skills/ac",
                "title": "",
                "metadata": {"supported_actions": ["turn_on_ac"], "executable": True},
            }
        ],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "success"
    assert result.tool_name == "ac_tool"
    assert calls


def test_execute_blocks_ambiguous_resources_for_action() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[
            _resource("memoryos://resources/ac-bedroom", supported_actions=["turn_on_ac"], device_id="bedroom"),
            _resource("memoryos://resources/ac-office", supported_actions=["turn_on_ac"], device_id="office"),
        ],
        skills=[_skill("memoryos://skills/ac", supported_actions=["turn_on_ac"])],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.reason == "Ambiguous resources for action turn_on_ac"
    assert calls == []


def test_execute_blocks_ambiguous_skills_for_action() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[_resource("memoryos://resources/ac", supported_actions=["turn_on_ac"], device_id="ac")],
        skills=[
            _skill("memoryos://skills/ac-primary", supported_actions=["turn_on_ac"]),
            _skill("memoryos://skills/ac-secondary", supported_actions=["turn_on_ac"]),
        ],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.reason == "Ambiguous skills for action turn_on_ac"
    assert calls == []


def test_execute_blocks_when_no_skill_supports_action() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[_resource("memoryos://resources/ac", supported_actions=["turn_on_ac"])],
        skills=[_skill("memoryos://skills/fan", supported_actions=["turn_on_fan"])],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.reason == "No skill supports action turn_on_ac"
    assert calls == []


def test_execute_blocks_when_no_resource_supports_action() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[_resource("memoryos://resources/fan", supported_actions=["turn_on_fan"])],
        skills=[_skill("memoryos://skills/ac", supported_actions=["turn_on_ac"])],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.reason == "No resource supports action turn_on_ac"
    assert calls == []


def test_execute_merges_skill_default_args_with_selected_resource_tool_args() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[
            _resource("memoryos://resources/fan", supported_actions=["turn_on_fan"], tool_args={"device_id": "fan"}),
            _resource(
                "memoryos://resources/ac",
                supported_actions="turn_on_ac",
                tool_args={"device_id": "ac", "temperature": 22},
            ),
        ],
        skills=[_skill("memoryos://skills/ac", action="turn_on_ac", default_args={"mode": "cool", "temperature": 24})],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "success"
    assert result.tool_args["mode"] == "cool"
    assert result.tool_args["temperature"] == 22
    assert result.tool_args["device_id"] == "ac"
    assert calls == [result.tool_args]


def test_execute_requires_all_explicit_action_metadata_to_match() -> None:
    calls: list[dict] = []
    context = _context_items(
        resources=[
            _resource(
                "memoryos://resources/ac",
                action="turn_on_fan",
                supported_actions=["turn_on_ac"],
                device_id="ac",
            )
        ],
        skills=[_skill("memoryos://skills/ac", supported_actions=["turn_on_ac"])],
    )

    result = ActionExecutor(_registry(calls)).execute(PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"), context)

    assert result.status == "blocked"
    assert result.reason == "No resource supports action turn_on_ac"
    assert calls == []
