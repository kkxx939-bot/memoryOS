from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REQUIRED_METADATA_KEYS = {
    "id",
    "user_id",
    "type",
    "title",
    "path",
    "tags",
    "source",
    "confidence",
    "created_at",
    "updated_at",
    "last_accessed_at",
    "active_count",
    "hotness",
    "lifecycle_state",
    "temporal_scope",
    "base_weight",
    "evidence_count",
    "positive_count",
    "negative_count",
    "effective_weight",
    "abstract",
    "status",
    "supersedes",
    "superseded_by",
    "valid_from",
    "valid_until",
    "last_confirmed_at",
    "source_episode_id",
}

MEMORY_TYPE_DESCRIPTIONS = {
    "profile": "Stable identity and background facts about the user.",
    "preference": "Explicit likes, dislikes, defaults, and response preferences.",
    "habit": "Repeated behavior patterns that can support prediction.",
    "trigger": "Conditions that tend to precede a need, action, or intervention.",
    "intervention": "Robot or assistant actions taken in past situations.",
    "feedback": "User acceptance, rejection, correction, or annoyance signals.",
    "policy": "Permission and risk boundaries for autonomous action.",
    "event": "Timestamped or episodic facts that should remain auditable.",
    "case": "Reusable problem, context, action, and outcome examples.",
}


@dataclass(frozen=True)
class MemoryTypeSpec:
    memory_type: str
    directory: str
    description: str
    operation_mode: str = "upsert"
    merge_op: str = "patch"
    route_group: str = "relevant"
    overview_title: str = "Overview"
    content_template: str = "{{ content }}"
    embedding_template: str = "{{ memory_type }}\n{{ title }}\n{{ tags }}\n{{ abstract }}\n{{ content }}"
    overview_template: str = ""


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "domain" / "memory" / "schemas"


@lru_cache(maxsize=1)
def load_memory_type_specs() -> dict[str, MemoryTypeSpec]:
    specs = {}
    for path in sorted(SCHEMA_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = MemoryTypeSpec(
            memory_type=str(data["memory_type"]),
            directory=str(data["directory"]),
            description=str(data.get("description", "")),
            operation_mode=str(data.get("operation_mode", "upsert")),
            merge_op=str(data.get("merge_op", "patch")),
            route_group=str(data.get("route_group", "relevant")),
            overview_title=str(data.get("overview_title", "Overview")),
            content_template=str(data.get("content_template", "{{ content }}")),
            embedding_template=str(data.get("embedding_template", MemoryTypeSpec("", "", "").embedding_template)),
            overview_template=str(data.get("overview_template", "")),
        )
        specs[spec.memory_type] = spec
    if not specs:
        raise RuntimeError(f"No memory schemas found in {SCHEMA_DIR}")
    return specs


MEMORY_TYPE_SPECS = load_memory_type_specs()


def memory_type_spec(memory_type: str) -> MemoryTypeSpec:
    if memory_type not in MEMORY_TYPE_SPECS:
        raise ValueError(f"Unknown memory type: {memory_type}")
    return MEMORY_TYPE_SPECS[memory_type]


def memory_specs_by_route_group(route_group: str) -> dict[str, MemoryTypeSpec]:
    return {
        memory_type: spec
        for memory_type, spec in MEMORY_TYPE_SPECS.items()
        if spec.route_group == route_group
    }


def validate_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_METADATA_KEYS - metadata.keys())
    if missing:
        raise ValueError(f"Memory metadata missing keys: {', '.join(missing)}")
    if not isinstance(metadata.get("tags"), list):
        raise ValueError("Memory metadata field 'tags' must be a list")
    confidence = metadata.get("confidence")
    if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
        raise ValueError("Memory metadata field 'confidence' must be a number in [0, 1]")
    active_count = metadata.get("active_count")
    if not isinstance(active_count, int) or active_count < 0:
        raise ValueError("Memory metadata field 'active_count' must be a non-negative integer")
    hotness = metadata.get("hotness")
    if not isinstance(hotness, int | float) or not 0 <= hotness <= 1:
        raise ValueError("Memory metadata field 'hotness' must be a number in [0, 1]")
    if metadata.get("lifecycle_state") not in {"hot", "warm", "cold"}:
        raise ValueError("Memory metadata field 'lifecycle_state' must be hot, warm, or cold")
    if metadata.get("temporal_scope") not in {"stable", "rolling_7d", "rolling_30d", "episodic", "seasonal"}:
        raise ValueError("Memory metadata field 'temporal_scope' is invalid")
    if metadata.get("status") not in {"active", "obsolete", "deleted", "pending"}:
        raise ValueError("Memory metadata field 'status' must be active, obsolete, deleted, or pending")
    if not isinstance(metadata.get("supersedes"), list):
        raise ValueError("Memory metadata field 'supersedes' must be a list")
    superseded_by = metadata.get("superseded_by")
    if superseded_by is not None and not isinstance(superseded_by, str):
        raise ValueError("Memory metadata field 'superseded_by' must be a string or null")
    for key in ("base_weight", "effective_weight"):
        value = metadata.get(key)
        if not isinstance(value, int | float) or not 0 <= value <= 1:
            raise ValueError(f"Memory metadata field '{key}' must be a number in [0, 1]")
    for key in ("evidence_count", "positive_count", "negative_count"):
        value = metadata.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Memory metadata field '{key}' must be a non-negative integer")


def type_description(memory_type: str) -> str:
    return MEMORY_TYPE_DESCRIPTIONS.get(memory_type, "Unknown memory type.")


def render_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
