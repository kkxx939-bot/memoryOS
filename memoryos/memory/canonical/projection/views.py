"""Views responsibilities for canonical projection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.durable_io import atomic_write_json
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
)
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import (
    materialized_current_revision_payload,
)

if TYPE_CHECKING:
    from .service import CanonicalMemoryProjector


def _write_scope_views(self: CanonicalMemoryProjector, obj: ContextObject, record: ProjectionRecord) -> None:
    metadata = dict(obj.metadata or {})
    raw_scope = metadata.get("scope")
    if not isinstance(raw_scope, dict):
        return
    try:
        canonical_scope = MemoryScope.from_dict(raw_scope)
    except (KeyError, TypeError, ValueError):
        return
    for scope_ref in canonical_scope.applicability.all_of:
        directory = (
            self.root
            / "views"
            / "scope"
            / self._segment(obj.tenant_id or "default")
            / self._segment(scope_ref.namespace)
            / self._segment(scope_ref.kind)
        )
        parent_path = list(scope_ref.parent_path)
        directory = directory / ("path" if parent_path else "root")
        for parent in parent_path:
            directory = directory / self._segment(parent)
        directory = directory / self._segment(scope_ref.id) / self._segment(metadata.get("claim_id", "unknown"))
        self._write_revisioned_view(directory, self._view_reference(obj, record))


def _write_taxonomy_view(self: CanonicalMemoryProjector, obj: ContextObject, record: ProjectionRecord) -> None:
    metadata = dict(obj.metadata or {})
    directory = (
        self.root
        / "views"
        / "taxonomy"
        / self._segment(obj.tenant_id or "default")
        / self._taxonomy_path(metadata)
        / self._segment(metadata.get("claim_id", "unknown"))
    )
    self._write_revisioned_view(directory, self._view_reference(obj, record))


def _write_revisioned_view(self: CanonicalMemoryProjector, directory: Path, payload: dict[str, Any]) -> None:
    revision = int(payload["source_revision"])
    attempt_id = str(payload["projection_attempt_id"])
    self._write_json_atomic(directory / f"rev-{revision}-attempt-{attempt_id}.json", payload)


def _publish_view_currents(self: CanonicalMemoryProjector, record: ProjectionRecord) -> None:
    pattern = f"views/**/rev-{record.source_revision}-attempt-{record.projection_attempt_id}.json"
    for path in self.root.glob(pattern):
        payload = self._read_json_optional(path)
        if (
            payload is None
            or str(payload.get("claim_uri", "")) != record.claim_uri
            or str(payload.get("projection_attempt_id", "")) != record.projection_attempt_id
            or str(payload.get("input_effect_hash", "")) != record.input_effect_hash
        ):
            continue
        current_path = path.parent / "current.json"
        current = self._read_json_optional(current_path) or {}
        current_revision = int(current.get("source_revision", 0) or 0)
        if current_revision > record.source_revision:
            continue
        if (
            current_revision == record.source_revision
            and current
            and str(current.get("input_effect_hash", "")) != record.input_effect_hash
        ):
            raise ProjectionIntegrityError("same revision view has a different input effect")
        self._write_json_atomic(current_path, payload)


def _view_reference(self: CanonicalMemoryProjector, obj: ContextObject, record: ProjectionRecord) -> dict[str, Any]:
    metadata = dict(obj.metadata or {})
    return dict(
        self.sanitizer.sanitize_trace(
            {
                "claim_uri": obj.uri,
                "slot_uri": record.slot_uri,
                "tenant_id": obj.tenant_id or "default",
                "owner_user_id": obj.owner_user_id or "",
                "canonical_kind": metadata.get("canonical_kind"),
                "claim_state": metadata.get("claim_state"),
                "canonical_head_digest": metadata.get("canonical_head_digest"),
                "current_transaction_id": metadata.get("current_transaction_id"),
                "current_receipt_digest": metadata.get("current_receipt_digest"),
                "slot_id": metadata.get("slot_id"),
                "claim_id": metadata.get("claim_id"),
                "source_revision": record.source_revision,
                "projection_revision": record.projection_revision,
                "projection_attempt_id": record.projection_attempt_id,
                "input_effect_hash": record.input_effect_hash,
                "publish_token": record.publish_token,
                "projected_content_digest": record.projected_content_digest,
                "projected_relation_digest": record.projected_relation_digest,
                "current_claim_revision": record.current_claim_revision,
                "projection_record_path": str(self.record_store.attempt_path_for(record)),
            }
        )
    )


def _taxonomy_path(self: CanonicalMemoryProjector, metadata: dict[str, Any]) -> Path:
    memory_type = str(metadata.get("memory_type", "memory"))
    current = materialized_current_revision_payload(metadata)
    values = dict(current.get("value_fields", {}) or {})
    identity = dict(metadata.get("identity_fields", {}) or {})
    category = {
        "project_decision": "decisions",
        "project_rule": "rules",
        "preference": "preferences",
        "agent_experience": "experiences",
        "profile": "profiles",
        "entity": "entities",
        "event": "events",
    }.get(memory_type, "memory")
    topic = str(
        identity.get("decision_topic")
        or identity.get("rule_topic")
        or identity.get("dimension")
        or identity.get("task_pattern")
        or identity.get("attribute_key")
        or identity.get("canonical_entity_id")
        or metadata.get("canonical_value")
        or values.get("topic")
        or values.get("dimension")
        or "general"
    )
    return Path(category) / self._segment(topic)


def _remove_view_currents(self: CanonicalMemoryProjector, record: ProjectionRecord) -> None:
    for path in self.root.glob("views/**/current.json"):
        payload = self._read_json_optional(path)
        if payload is None:
            continue
        if (
            str(payload.get("claim_uri", "")) == record.claim_uri
            and int(payload.get("source_revision", 0) or 0) == record.source_revision
            and str(payload.get("projection_attempt_id", "")) == record.projection_attempt_id
            and str(payload.get("publish_token", "")) == record.publish_token
        ):
            path.unlink(missing_ok=True)


def _segment(self: CanonicalMemoryProjector, value: Any) -> str:
    safe_value = str(self.sanitizer.sanitize_trace(str(value)))
    cleaned = re.sub(r"[^a-zA-Z0-9._:-]+", "-", safe_value).strip("-.")
    return cleaned[:120] or "unknown"


def _read_json_optional(self: CanonicalMemoryProjector, path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise ProjectionIntegrityError(f"projection view state cannot be a symbolic link: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectionIntegrityError(f"invalid projection view state: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProjectionIntegrityError(f"invalid projection view state: {path.name}")
    return value


def _write_json_atomic(self: CanonicalMemoryProjector, path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ProjectionIntegrityError(f"projection view state cannot be a symbolic link: {path.name}")
    try:
        atomic_write_json(path, payload, artifact_root=self.root)
    except ValueError as exc:
        raise ProjectionIntegrityError(f"projection view state publication is unsafe: {path.name}") from exc
