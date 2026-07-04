from __future__ import annotations

from typing import Any

from memoryos.contextdb.model.context_type import ContextType


def object_metadata(obj: Any, content: str = "") -> dict:
    metadata = dict(getattr(obj, "metadata", {}) or {})
    metadata.setdefault("title", str(getattr(obj, "title", "") or "untitled"))
    metadata.setdefault("content", content)
    return metadata


def safe_context_type(obj: Any) -> ContextType | None:
    try:
        value = getattr(obj, "context_type", None)
        return value if isinstance(value, ContextType) else ContextType(str(value))
    except (TypeError, ValueError):
        return None


def dominant_action(metadata: dict) -> str:
    distribution = metadata.get("action_distribution", []) or []
    if not isinstance(distribution, list) or not distribution:
        return str(metadata.get("action", "unknown"))
    return str(max(distribution, key=lambda item: int(item.get("count", 0) or 0)).get("action", "unknown"))


def action_lines(metadata: dict) -> list[str]:
    distribution = metadata.get("action_distribution", []) or []
    if not isinstance(distribution, list) or not distribution:
        action = metadata.get("action")
        return [f"- {action}"] if action else ["- unknown"]
    return [f"- {item.get('action', 'unknown')}: {item.get('count', 0)}" for item in distribution]


def bullet_value(metadata: dict, key: str, default: object = "") -> str:
    value = metadata.get(key, default)
    if value is None:
        value = default
    return str(value)
