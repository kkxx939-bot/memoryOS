from __future__ import annotations

from memoryos.core.time import utc_now
from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.model.action_result import ActionResult
from memoryos.prediction.model.prediction_result import PolicyDecision
from memoryos.skill.tool_registry import ToolRegistry


class ActionExecutor:
    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or ToolRegistry()

    def execute(self, decision: PolicyDecision, action_context: ActionContext) -> ActionResult:
        started_at = utc_now()
        if decision.mode != "execute" or not decision.allowed:
            return ActionResult(action=decision.action, status="skipped", executed=False, reason="Decision does not execute.", started_at=started_at)
        resources = self._section_items(action_context, "resource")
        skills = self._section_items(action_context, "skill")
        if not resources:
            return ActionResult(action=decision.action, status="blocked", executed=False, reason="Required resource is missing.", started_at=started_at)
        if not skills:
            return ActionResult(action=decision.action, status="blocked", executed=False, reason="Required skill is missing.", started_at=started_at)
        skill = skills[0]
        metadata = skill.get("metadata", {}) if isinstance(skill.get("metadata", {}), dict) else {}
        if metadata.get("executable") is not True:
            return ActionResult(action=decision.action, status="blocked", executed=False, reason="Required skill is not executable.", started_at=started_at)
        tool_name = self._tool_name(skill)
        tool_args = self._tool_args(decision.action, resources[0], skill)
        if self.tool_registry.can_execute(tool_name):
            try:
                self.tool_registry.validate_args(tool_name, tool_args)
            except ValueError as exc:
                return ActionResult(
                    action=decision.action,
                    status="failed",
                    executed=False,
                    reason="invalid_args",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    resource_uris=[str(item.get("uri", "")) for item in resources],
                    skill_uris=[str(item.get("uri", "")) for item in skills],
                    error=str(exc),
                    started_at=started_at,
                )
            try:
                output = self.tool_registry.execute(tool_name, tool_args, dry_run=bool(metadata.get("dry_run")))
            except Exception as exc:  # pragma: no cover - exact tool failures are runtime data.
                return ActionResult(
                    action=decision.action,
                    status="failed",
                    executed=True,
                    reason=str(exc),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    resource_uris=[str(item.get("uri", "")) for item in resources],
                    skill_uris=[str(item.get("uri", "")) for item in skills],
                    error=str(exc),
                    started_at=started_at,
                )
            return ActionResult(
                action=decision.action,
                status="success",
                executed=True,
                reason="Tool executed successfully.",
                tool_name=tool_name,
                tool_args=tool_args,
                resource_uris=[str(item.get("uri", "")) for item in resources],
                skill_uris=[str(item.get("uri", "")) for item in skills],
                output=output,
                started_at=started_at,
            )
        return ActionResult(
            action=decision.action,
            status="failed",
            executed=False,
            reason=f"No executable tool registered for {tool_name}.",
            tool_name=tool_name,
            tool_args=tool_args,
            resource_uris=[str(item.get("uri", "")) for item in resources],
            skill_uris=[str(item.get("uri", "")) for item in skills],
            error=f"tool_not_registered:{tool_name}",
            started_at=started_at,
        )

    def _section_items(self, action_context: ActionContext, section: str) -> list[dict]:
        return list(action_context.packed_context.get("slices", {}).get(section, {}).get("items", []))

    def _tool_name(self, skill: dict) -> str:
        metadata = skill.get("metadata", {}) if isinstance(skill.get("metadata", {}), dict) else {}
        return str(metadata.get("tool_name") or skill.get("title") or skill.get("uri") or "")

    def _tool_args(self, action: str, resource: dict, skill: dict) -> dict:
        resource_metadata = resource.get("metadata", {}) if isinstance(resource.get("metadata", {}), dict) else {}
        skill_metadata = skill.get("metadata", {}) if isinstance(skill.get("metadata", {}), dict) else {}
        args = dict(skill_metadata.get("default_args", {})) if isinstance(skill_metadata.get("default_args", {}), dict) else {}
        args.update(resource_metadata.get("tool_args", {}) if isinstance(resource_metadata.get("tool_args", {}), dict) else {})
        args.setdefault("action", action)
        args.setdefault("device_id", resource_metadata.get("device_id") or str(resource.get("uri", "")).rstrip("/").rsplit("/", 1)[-1])
        if "temperature" not in args and "temperature" in resource_metadata:
            args["temperature"] = resource_metadata["temperature"]
        if "temperature" not in args and "temperature" in skill_metadata:
            args["temperature"] = skill_metadata["temperature"]
        return args


ExecutionResult = ActionResult
Executor = ActionExecutor
