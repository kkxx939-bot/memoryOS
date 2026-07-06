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
        skill, resource, selection_error = self._select_skill_and_resource(decision.action, skills, resources)
        if selection_error is not None:
            return ActionResult(
                action=decision.action,
                status="blocked",
                executed=False,
                reason=selection_error,
                resource_uris=[str(item.get("uri", "")) for item in resources],
                skill_uris=[str(item.get("uri", "")) for item in skills],
                started_at=started_at,
            )
        if skill is None or resource is None:
            return ActionResult(
                action=decision.action,
                status="blocked",
                executed=False,
                reason=f"No unique skill/resource supports action {decision.action}.",
                resource_uris=[str(item.get("uri", "")) for item in resources],
                skill_uris=[str(item.get("uri", "")) for item in skills],
                started_at=started_at,
            )
        metadata = skill.get("metadata", {}) if isinstance(skill.get("metadata", {}), dict) else {}
        if metadata.get("executable") is not True:
            return ActionResult(action=decision.action, status="blocked", executed=False, reason="Required skill is not executable.", started_at=started_at)
        tool_name = self._tool_name(skill, resource)
        tool_args = self._tool_args(decision.action, resource, skill)
        resource_uris = [str(resource.get("uri", ""))]
        skill_uris = [str(skill.get("uri", ""))]
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
                    resource_uris=resource_uris,
                    skill_uris=skill_uris,
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
                    resource_uris=resource_uris,
                    skill_uris=skill_uris,
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
                resource_uris=resource_uris,
                skill_uris=skill_uris,
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
            resource_uris=resource_uris,
            skill_uris=skill_uris,
            error=f"tool_not_registered:{tool_name}",
            started_at=started_at,
        )

    def _section_items(self, action_context: ActionContext, section: str) -> list[dict]:
        return list(action_context.packed_context.get("slices", {}).get(section, {}).get("items", []))

    def _select_skill_and_resource(
        self,
        action: str,
        skills: list[dict],
        resources: list[dict],
    ) -> tuple[dict | None, dict | None, str | None]:
        skill_candidates, skill_error = self._matching_items(action, skills, "skill")
        if skill_error is not None:
            return None, None, skill_error
        resource_candidates, resource_error = self._matching_items(action, resources, "resource")
        if resource_error is not None:
            return None, None, resource_error
        if len(skill_candidates) > 1 and len(resource_candidates) > 1:
            return None, None, f"Ambiguous skills and resources for action {action}"
        if len(skill_candidates) > 1:
            return None, None, f"Ambiguous skills for action {action}"
        if len(resource_candidates) > 1:
            return None, None, f"Ambiguous resources for action {action}"
        if not skill_candidates:
            return None, None, f"No skill supports action {action}"
        if not resource_candidates:
            return None, None, f"No resource supports action {action}"
        skill = skill_candidates[0]
        resource = resource_candidates[0]
        skill_tool = self._declared_tool_name(skill)
        resource_tool = self._declared_tool_name(resource)
        if skill_tool and resource_tool and str(skill_tool) != str(resource_tool):
            return None, None, "resource/skill tool_name mismatch"
        return skill, resource, None

    def _matching_items(self, action: str, items: list[dict], item_name: str) -> tuple[list[dict], str | None]:
        matches = []
        generic = []
        has_explicit = False
        for item in items:
            matched, explicit = self._match_action(item, action)
            if explicit:
                has_explicit = True
            if matched:
                if explicit:
                    matches.append(item)
                else:
                    generic.append(item)
        if matches:
            return matches, None
        if generic:
            return generic, None
        if has_explicit:
            return [], f"No {item_name} supports action {action}"
        return [], None

    def _match_action(self, item: dict, action: str) -> tuple[bool, bool]:
        metadata = self._metadata(item)
        has_action = "action" in metadata
        has_supported_actions = "supported_actions" in metadata
        if has_action and str(metadata.get("action", "")) != action:
            return False, True
        supported = metadata.get("supported_actions")
        if has_supported_actions:
            if isinstance(supported, str):
                if supported != action:
                    return False, True
            elif isinstance(supported, list):
                if action not in {str(value) for value in supported}:
                    return False, True
            else:
                return False, True
        if has_action or has_supported_actions:
            return True, True
        return True, False

    def _metadata(self, item: dict) -> dict:
        metadata = item.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _declared_tool_name(self, item: dict) -> str:
        metadata = self._metadata(item)
        return str(item.get("tool_name") or metadata.get("tool_name") or "")

    def _tool_name(self, skill: dict, resource: dict) -> str:
        return self._declared_tool_name(skill) or self._declared_tool_name(resource) or str(skill.get("title") or skill.get("uri") or "")

    def _tool_args(self, action: str, resource: dict, skill: dict) -> dict:
        resource_metadata = self._metadata(resource)
        skill_metadata = self._metadata(skill)
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
