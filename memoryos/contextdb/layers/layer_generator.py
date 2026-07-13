"""上下文数据库里的分层生成器。"""

from __future__ import annotations

from typing import Any

from memoryos.contextdb.layers.semantic_templates import (
    action_lines,
    bullet_value,
    dominant_action,
    object_metadata,
    safe_context_type,
)
from memoryos.contextdb.model.context_type import ContextType


def l0_abstract(text: str, max_chars: int = 220) -> str:
    compact = " ".join(str(text).split())
    return compact[:max_chars]


def l1_overview(title: str, bullets: list[str], max_bullets: int = 12) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {bullet}" for bullet in bullets[:max_bullets] if str(bullet).strip())
    return "\n".join(lines).strip() + "\n"


def generate_l0_for_object(obj: Any, content: str = "") -> str:
    try:
        context_type = safe_context_type(obj)
        metadata = object_metadata(obj, content)
        title = str(getattr(obj, "title", metadata.get("title", "untitled")) or "untitled")
        if context_type == ContextType.BEHAVIOR_PATTERN:
            scene_key = bullet_value(metadata, "scene_key", "default")
            action = dominant_action(metadata)
            case_count = len(metadata.get("case_refs", []) or [])
            return f"用户在 {scene_key} 场景下多次表现出 {action} 倾向；该模式由 {case_count} 条行为证据支持。"
        if context_type == ContextType.BEHAVIOR_CLUSTER:
            scene_key = bullet_value(metadata, "scene_key", "default")
            case_count = len(metadata.get("case_refs", []) or [])
            return f"用户在 {scene_key} 场景下出现相似行为簇；该簇由 {case_count} 条行为证据支持。"
        if context_type == ContextType.BEHAVIOR_CASE:
            scene_key = bullet_value(metadata, "scene_key", "default")
            action = bullet_value(metadata, "selected_action", metadata.get("executed_action", "unknown"))
            return f"用户在 {scene_key} 场景下发生一次行为案例；记录动作={action}。"
        if context_type == ContextType.ACTION_POLICY:
            scene_key = bullet_value(metadata, "scene_key", "default")
            action = bullet_value(metadata, "action", "unknown")
            q_value = bullet_value(metadata, "q_value", "0.0")
            status = bullet_value(metadata, "status", "active")
            auto_execute = bullet_value(metadata, "auto_execute_allowed", "False")
            return f"在 {scene_key} 场景下，{action} 是候选动作；当前 q_value={q_value}，状态={status}，自动执行={auto_execute}。"
        if context_type == ContextType.MEMORY:
            kind = bullet_value(metadata, "memory_kind", metadata.get("kind", "explicit"))
            confidence = bullet_value(metadata, "confidence", "1.0")
            return f"用户相关记忆：{title}；类型={kind}，置信度={confidence}。"
        if context_type == ContextType.SESSION:
            return f"会话记录：{title}；包含用户交互、观察和预测上下文。"
        if context_type == ContextType.RESOURCE:
            return f"资源上下文：{title}；用于判断动作执行所需资源是否可用。"
        if context_type == ContextType.SKILL:
            return f"技能上下文：{title}；用于判断候选动作是否具备可执行能力。"
    except (AttributeError, KeyError, TypeError, ValueError):
        return l0_abstract(content or str(getattr(obj, "title", "")) or "context")
    return l0_abstract(content or str(getattr(obj, "title", "")) or "context")


def generate_l1_for_object(obj: Any, content: str = "") -> str:
    try:
        context_type = safe_context_type(obj)
        metadata = object_metadata(obj, content)
        title = str(getattr(obj, "title", metadata.get("title", "untitled")) or "untitled")
        if context_type == ContextType.BEHAVIOR_PATTERN:
            scene_key = bullet_value(metadata, "scene_key", "default")
            opportunity = dict(metadata.get("opportunity", {}) or {})
            lines = [
                f"# BehaviorPattern: {scene_key}",
                "",
                "Trigger Conditions:",
                f"- {metadata.get('trigger_conditions', {})}",
                "",
                "Evidence:",
                f"- case_count: {len(metadata.get('case_refs', []) or [])}",
                f"- positive_count: {bullet_value(opportunity, 'positive_count', 0)}",
                f"- negative_count: {bullet_value(opportunity, 'negative_feedback_count', 0)}",
                f"- opportunity_count: {bullet_value(opportunity, 'opportunity_count', 0)}",
                f"- activation_count: {bullet_value(opportunity, 'activation_count', 0)}",
                f"- missed_opportunity_count: {bullet_value(opportunity, 'missed_opportunity_count', 0)}",
                "",
                "Dominant Actions:",
                *action_lines(metadata),
                "",
                "Memory Anchor:",
                f"- {bullet_value(metadata, 'memory_anchor_uri', '')}",
                "",
                "Action Policies:",
                f"- {metadata.get('related_policy_uris', metadata.get('policy_uris', []))}",
            ]
            return "\n".join(lines).strip() + "\n"
        if context_type == ContextType.ACTION_POLICY:
            scene_key = bullet_value(metadata, "scene_key", "default")
            action = bullet_value(metadata, "action", "unknown")
            lines = [
                f"# ActionPolicy: {scene_key}/{action}",
                "",
                "State:",
                f"- status: {bullet_value(metadata, 'status', 'active')}",
                f"- auto_execute_allowed: {bullet_value(metadata, 'auto_execute_allowed', False)}",
                f"- cooldown_until: {bullet_value(metadata, 'cooldown_until', '')}",
                "",
                "Scores:",
                f"- q_value: {bullet_value(metadata, 'q_value', 0.0)}",
                f"- confidence: {bullet_value(metadata, 'confidence', 0.0)}",
                f"- reward_score: {bullet_value(metadata, 'reward_score', 0.0)}",
                f"- penalty_score: {bullet_value(metadata, 'penalty_score', 0.0)}",
                "",
                "Evidence:",
                f"- success_count: {bullet_value(metadata, 'success_count', 0)}",
                f"- failure_count: {bullet_value(metadata, 'failure_count', 0)}",
                f"- neutral_count: {bullet_value(metadata, 'neutral_count', 0)}",
                f"- negative_feedback_count: {bullet_value(metadata, 'negative_feedback_count', 0)}",
                "",
                "Relations:",
                f"- memory_anchor_uri: {bullet_value(metadata, 'memory_anchor_uri', '')}",
                f"- supported_behavior_pattern_uris: {metadata.get('supported_behavior_pattern_uris', [])}",
                f"- constrained_by_memory_uris: {metadata.get('constrained_by_memory_uris', [])}",
                f"- required_resource_uris: {metadata.get('required_resource_uris', [])}",
                f"- required_skill_uris: {metadata.get('required_skill_uris', [])}",
            ]
            return "\n".join(lines).strip() + "\n"
        if context_type == ContextType.MEMORY:
            lines = [
                f"# Memory: {title}",
                "",
                "Kind:",
                f"- {bullet_value(metadata, 'memory_kind', metadata.get('kind', 'explicit'))}",
                "",
                "Content:",
                f"- {l0_abstract(content or bullet_value(metadata, 'summary', title), 500)}",
                "",
                "Relations:",
                f"- supports behavior: {metadata.get('supporting_behavior_uris', [])}",
                f"- constrains policy: {metadata.get('constrains_policy_uris', [])}",
            ]
            return "\n".join(lines).strip() + "\n"
        if context_type is not None and context_type in {ContextType.BEHAVIOR_CASE, ContextType.BEHAVIOR_CLUSTER, ContextType.SESSION, ContextType.RESOURCE, ContextType.SKILL}:
            return l1_overview(
                f"{context_type.value}: {title}",
                [
                    f"context_type: {context_type.value}",
                    f"summary: {generate_l0_for_object(obj, content)}",
                    f"metadata: {metadata}",
                ],
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        return l1_overview(
            str(getattr(obj, "title", "Context")),
            [content[:240] if content else "No content available."],
        )
    return l1_overview(str(getattr(obj, "title", "Context")), [content[:240] if content else "No content available."])
