from __future__ import annotations

from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.model.action_result import ActionResult
from memoryos.prediction.model.prediction_result import PolicyDecision
from memoryos.skill.tool_registry import ToolRegistry


class ActionExecutor:
    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or ToolRegistry()

    def execute(self, decision: PolicyDecision, action_context: ActionContext) -> ActionResult:
        if decision.mode != "execute" or not decision.allowed:
            return ActionResult(action=decision.action, status="skipped", executed=False, reason="Decision does not execute.")
        resources = self._section_items(action_context, "resource")
        skills = self._section_items(action_context, "skill")
        if not resources:
            return ActionResult(action=decision.action, status="blocked", executed=False, reason="Required resource is missing.")
        if not skills:
            return ActionResult(action=decision.action, status="blocked", executed=False, reason="Required skill is missing.")
        skill = skills[0]
        tool_name = self._tool_name(skill)
        payload = {"action": decision.action, "resources": resources, "skill": skill}
        if self.tool_registry.can_execute(tool_name):
            try:
                output = self.tool_registry.execute(tool_name, payload)
            except Exception as exc:  # pragma: no cover - exact tool failures are runtime data.
                return ActionResult(
                    action=decision.action,
                    status="failed",
                    executed=True,
                    reason=str(exc),
                    tool_name=tool_name,
                    resource_uris=[str(item.get("uri", "")) for item in resources],
                    skill_uris=[str(item.get("uri", "")) for item in skills],
                )
            return ActionResult(
                action=decision.action,
                status="success",
                executed=True,
                reason="Tool executed successfully.",
                tool_name=tool_name,
                resource_uris=[str(item.get("uri", "")) for item in resources],
                skill_uris=[str(item.get("uri", "")) for item in skills],
                output=output,
            )
        if skill.get("metadata", {}).get("executable") is True:
            return ActionResult(
                action=decision.action,
                status="success",
                executed=True,
                reason="Executable skill context accepted by fake registry.",
                tool_name=tool_name,
                resource_uris=[str(item.get("uri", "")) for item in resources],
                skill_uris=[str(item.get("uri", "")) for item in skills],
                output={"simulated": True},
            )
        return ActionResult(
            action=decision.action,
            status="failed",
            executed=False,
            reason=f"No executable tool registered for {tool_name}.",
            tool_name=tool_name,
            resource_uris=[str(item.get("uri", "")) for item in resources],
            skill_uris=[str(item.get("uri", "")) for item in skills],
        )

    def _section_items(self, action_context: ActionContext, section: str) -> list[dict]:
        return list(action_context.packed_context.get("slices", {}).get(section, {}).get("items", []))

    def _tool_name(self, skill: dict) -> str:
        metadata = skill.get("metadata", {}) if isinstance(skill.get("metadata", {}), dict) else {}
        return str(metadata.get("tool_name") or skill.get("title") or skill.get("uri") or "")


ExecutionResult = ActionResult
Executor = ActionExecutor
