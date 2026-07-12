"""Revision-bound derived projections for canonical memory."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexStore, QueueJob, QueueStore, SourceStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.memory.canonical.projection_state import (
    ProjectionRecord,
    ProjectionRecordStore,
    ProjectionStatus,
    ProjectionStepStatus,
)
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.visibility import read_committed_canonical
from memoryos.providers.embedding import EmbeddingProvider, HashingEmbeddingProvider


@dataclass(frozen=True)
class ProjectionResult:
    claim_uri: str
    source_revision: int
    status: str
    record_path: str = ""


class CanonicalMemoryProjector:
    """Build disposable projections without ever writing a canonical object."""

    GENERATOR = "deterministic-template-v2"
    PROMPT_VERSION = "none"

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str | Path,
        *,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        record_store: ProjectionRecordStore | None = None,
        test_hook: Callable[[str, str, int], None] | None = None,
        status_callback: Callable[[ProjectionRecord], None] | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.root = Path(root)
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self.record_store = record_store or ProjectionRecordStore(self.root)
        self.test_hook = test_hook
        self.status_callback = status_callback

    def project(
        self,
        claim_uri: str,
        source_revision: int | None = None,
        *,
        force: bool = False,
    ) -> ProjectionResult:
        committed = read_committed_canonical(self.source_store, claim_uri)
        obj = committed.object
        metadata = dict(obj.metadata or {})
        current_revision = int(metadata.get("revision", 0))
        if committed.from_before_image:
            return ProjectionResult(claim_uri, current_revision, "skipped_uncommitted")
        if metadata.get("canonical_kind") != "claim":
            return ProjectionResult(claim_uri, current_revision, "skipped_non_claim")
        raw_scope = metadata.get("scope")
        try:
            canonical_scope = MemoryScope.from_dict(raw_scope) if isinstance(raw_scope, dict) else None
        except (KeyError, TypeError, ValueError):
            canonical_scope = None
        asserted_by = str(metadata.get("asserted_by") or "")
        asserted_by_service = str(metadata.get("asserted_by_service") or "")
        if (
            canonical_scope is None
            or canonical_scope.canonical_subject is None
            or canonical_scope.visibility.tenant_id != str(obj.tenant_id or "default")
            or canonical_scope.authority.inferred
            or (
                (canonical_scope.authority.principal_ids or canonical_scope.authority.service_ids)
                and asserted_by not in set(canonical_scope.authority.principal_ids)
                and asserted_by_service not in set(canonical_scope.authority.service_ids)
            )
        ):
            return ProjectionResult(claim_uri, current_revision, "skipped_invalid_scope")
        requested = current_revision if source_revision is None else int(source_revision)
        if requested < current_revision:
            stale_record = self.record_store.load(claim_uri, requested)
            if stale_record is not None and stale_record.current:
                self._remove_record_current(stale_record)
                self.record_store.clear_current_if(
                    claim_uri,
                    requested,
                    reason="canonical revision advanced before stale task was consumed",
                )
                stale_record = self.record_store.load(claim_uri, requested)
                if stale_record is not None:
                    self._emit(stale_record)
            return ProjectionResult(claim_uri, requested, "skipped_stale")
        if requested > current_revision:
            raise ValueError("projection source revision is newer than canonical claim")
        slot_uri = claim_uri.rsplit("/claims/", 1)[0]
        current_claim_revision = int(metadata.get("current_revision", requested))
        existing = self.record_store.load_current(claim_uri, source_revision=requested)
        if existing is not None and not force:
            self._emit(existing)
            return ProjectionResult(
                claim_uri,
                requested,
                "projected",
                str(self.record_store.record_path(claim_uri, requested)),
            )

        base = f"{claim_uri}/projections/rev-{requested}"
        l0_uri = f"{base}/l0.md"
        l1_uri = f"{base}/l1.md"
        l2_uri = f"{base}/l2.json"
        relations_uri = f"{base}/relations.json"
        manifest_uri = f"{base}/manifest.json"
        record = self.record_store.start(
            claim_uri=claim_uri,
            slot_uri=slot_uri,
            source_revision=requested,
            projection_revision=requested,
            l0_uri=l0_uri,
            l1_uri=l1_uri,
            l2_uri=l2_uri,
            manifest_uri=manifest_uri,
            current_claim_revision=current_claim_revision,
        )
        if force:
            record = self.record_store.update(
                record,
                index_status=ProjectionStepStatus.PENDING.value,
                vector_status=ProjectionStepStatus.PENDING.value,
                relation_status=ProjectionStepStatus.PENDING.value,
                scope_status=ProjectionStepStatus.PENDING.value,
                taxonomy_status=ProjectionStepStatus.PENDING.value,
            )
        published_view_currents = False
        self._notify("after_read", claim_uri, requested)
        try:
            revision = self._revision_payload(metadata, current_claim_revision)
            l0, l1, l2 = self._layers(obj, metadata, revision, requested)
            self._notify("before_artifacts", claim_uri, requested)
            self.source_store.write_content(l0_uri, l0)
            self.source_store.write_content(l1_uri, l1)
            self.source_store.write_content(l2_uri, l2)
            if record.relation_status != ProjectionStepStatus.COMPLETED.value:
                record = self.record_store.update(record, relation_status=ProjectionStepStatus.RUNNING.value)
                try:
                    self.source_store.write_content(
                        relations_uri,
                        json.dumps(
                            {
                                "claim_uri": claim_uri,
                                "slot_uri": slot_uri,
                                "source_revision": requested,
                                "relations": [relation.to_dict() for relation in obj.relations],
                            },
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        ),
                    )
                except Exception:
                    record = self.record_store.update(
                        record,
                        relation_status=ProjectionStepStatus.FAILED.value,
                    )
                    raise

            with self.record_store.claim_lock(claim_uri):
                if not self._is_current(claim_uri, requested, metadata):
                    stale = self.record_store.stale(record, "canonical revision changed before publication")
                    return self._result(stale, "skipped_stale")
                newer = self.record_store.load_current(claim_uri)
                if newer is not None and newer.source_revision > requested:
                    stale = self.record_store.stale(record, "newer projection revision is already current")
                    return self._result(stale, "skipped_stale")

                self._notify("before_publish", claim_uri, requested)
                projection_obj = self._projection_object(
                    obj,
                    metadata,
                    record,
                    layers=ContextLayers(l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri),
                )
                content = "\n".join((l0, l1, l2))
                if record.index_status != ProjectionStepStatus.COMPLETED.value:
                    record = self.record_store.update(record, index_status=ProjectionStepStatus.RUNNING.value)
                    try:
                        self.index_store.upsert_index(projection_obj, content=content)
                    except Exception:
                        record = self.record_store.update(record, index_status=ProjectionStepStatus.FAILED.value)
                        raise
                    record = self.record_store.update(record, index_status=ProjectionStepStatus.COMPLETED.value)

                if self.vector_store is None:
                    if record.vector_status != ProjectionStepStatus.SKIPPED.value:
                        record = self.record_store.update(record, vector_status=ProjectionStepStatus.SKIPPED.value)
                elif record.vector_status != ProjectionStepStatus.COMPLETED.value:
                    record = self.record_store.update(record, vector_status=ProjectionStepStatus.RUNNING.value)
                    try:
                        self._project_vector(projection_obj, "\n".join((l0, l1)), requested)
                    except Exception:
                        record = self.record_store.update(record, vector_status=ProjectionStepStatus.FAILED.value)
                        raise
                    record = self.record_store.update(record, vector_status=ProjectionStepStatus.COMPLETED.value)

                if record.relation_status != ProjectionStepStatus.COMPLETED.value:
                    record = self.record_store.update(
                        record,
                        relation_status=ProjectionStepStatus.COMPLETED.value,
                    )
                if record.scope_status != ProjectionStepStatus.COMPLETED.value:
                    record = self.record_store.update(record, scope_status=ProjectionStepStatus.RUNNING.value)
                    try:
                        self._write_scope_views(projection_obj, record)
                    except Exception:
                        record = self.record_store.update(
                            record,
                            scope_status=ProjectionStepStatus.FAILED.value,
                        )
                        raise
                    record = self.record_store.update(record, scope_status=ProjectionStepStatus.COMPLETED.value)
                if record.taxonomy_status != ProjectionStepStatus.COMPLETED.value:
                    record = self.record_store.update(record, taxonomy_status=ProjectionStepStatus.RUNNING.value)
                    try:
                        self._write_taxonomy_view(projection_obj, record)
                    except Exception:
                        record = self.record_store.update(
                            record,
                            taxonomy_status=ProjectionStepStatus.FAILED.value,
                        )
                        raise
                    record = self.record_store.update(record, taxonomy_status=ProjectionStepStatus.COMPLETED.value)

                if not self._is_current(claim_uri, requested, metadata):
                    self._remove_published_current(projection_obj, record)
                    stale = self.record_store.stale(record, "canonical revision changed during publication")
                    return self._result(stale, "skipped_stale")

                completed_preview = self.record_store.update(
                    record,
                    status=ProjectionStatus.COMPLETED.value,
                    failure_reason="",
                    retryable=False,
                    current=False,
                )
                self.source_store.write_content(
                    manifest_uri,
                    json.dumps(
                        self._manifest(completed_preview, metadata, relations_uri),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                )
                published_view_currents = True
                self._publish_view_currents(record)
                record = self.record_store.promote(completed_preview)
                if not record.current:
                    self._remove_published_current(projection_obj, record)
                    return self._result(record, "skipped_stale")
                self._notify("after_publish", claim_uri, requested)
                if not self._is_current(claim_uri, requested, metadata):
                    self._remove_published_current(projection_obj, record)
                    self.record_store.clear_current_if(
                        claim_uri,
                        requested,
                        reason="canonical revision changed after publication",
                    )
                    stale = self.record_store.load(claim_uri, requested) or record
                    return self._result(stale, "skipped_stale")
            return self._result(record, "projected")
        except Exception as exc:
            latest = self.record_store.load(claim_uri, requested) or record
            if latest.current:
                removed = self._remove_record_current(latest)
                if removed:
                    latest = self._reset_removed_components(latest)
                self.record_store.clear_current_if(
                    claim_uri,
                    requested,
                    reason=f"projection failed after publication: {type(exc).__name__}",
                )
                latest = self.record_store.load(claim_uri, requested) or latest
            elif published_view_currents:
                self._remove_view_currents(latest)
            failed = self.record_store.fail(latest, f"{type(exc).__name__}: {exc}", retryable=True)
            self._emit(failed)
            raise

    def rebuild(self, *, clear_views: bool = True) -> dict[str, int]:
        if clear_views:
            for name in ("scope", "taxonomy"):
                path = self.root / "views" / name
                if path.exists():
                    shutil.rmtree(path)
        projected = 0
        skipped = 0
        for obj in self.source_store.list_objects():
            if dict(obj.metadata or {}).get("canonical_kind") != "claim":
                continue
            result = self.project(obj.uri, force=True)
            if result.status == "projected":
                projected += 1
            else:
                skipped += 1
        return {"projected": projected, "skipped": skipped}

    def _layers(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        revision: dict[str, Any],
        source_revision: int,
    ) -> tuple[str, str, str]:
        revision_values = dict(revision.get("value_fields", {}) or {})
        value = str(
            revision_values.get("canonical_value")
            or revision_values.get("value")
            or metadata.get("canonical_value", obj.title)
        )
        state = str(revision.get("state") or metadata.get("state", ""))
        memory_type = str(metadata.get("memory_type", "memory"))
        l0 = f"{value} [{state}]"
        qualifiers = dict(revision.get("qualifiers", {}) or {})
        display_fields = dict(metadata.get("display_fields", {}) or {})
        l1_lines = [
            f"# {value}",
            f"- type: {memory_type}",
            f"- state: {state}",
            f"- source revision: {source_revision}",
            f"- current claim revision: {revision.get('revision', source_revision)}",
            f"- epistemic: {revision.get('epistemic_status', '')}",
            f"- relation: {revision.get('relation', '')}",
        ]
        display_text = next(
            (
                str(display_fields[name])
                for name in ("display_text", "summary", "decision", "rule", "rationale", "details", "reason")
                if display_fields.get(name)
            ),
            "",
        )
        if display_text:
            l1_lines.append(f"- display: {display_text}")
        if qualifiers:
            l1_lines.append(f"- qualifiers: {json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)}")
        l1 = "\n".join(l1_lines)
        l2 = json.dumps(
            {
                "claim_uri": obj.uri,
                "slot_id": metadata.get("slot_id"),
                "claim_id": metadata.get("claim_id"),
                "source_revision": source_revision,
                "current_claim_revision": revision.get("revision", source_revision),
                "canonical_value": value,
                "revision": revision,
                "display_fields": display_fields,
                "display_field_evidence_refs": dict(metadata.get("display_field_evidence_refs", {}) or {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return l0, l1, l2

    def _revision_payload(self, metadata: dict[str, Any], revision: int) -> dict[str, Any]:
        revisions = [
            dict(item) for item in metadata.get("revisions", []) or [] if int(item.get("revision", 0)) == revision
        ]
        if not revisions:
            raise ValueError("canonical claim revision payload is missing")
        return revisions[-1]

    def _projection_object(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        record: ProjectionRecord,
        *,
        layers: ContextLayers,
    ) -> ContextObject:
        projected = ContextObject.from_dict(obj.to_dict())
        projected.layers = layers
        projected.metadata = {
            **metadata,
            "projection_source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "current_claim_revision": record.current_claim_revision,
            "projection_manifest_uri": record.manifest_uri,
            "projection_record_path": str(self.record_store.record_path(record.claim_uri, record.source_revision)),
        }
        return projected

    def _project_vector(self, obj: ContextObject, content: str, source_revision: int) -> None:
        assert self.vector_store is not None
        embedding = self.embedding_provider.embed(content)
        self.vector_store.upsert_vector(
            obj.uri,
            embedding,
            metadata={
                "tenant_id": obj.tenant_id,
                "owner_user_id": obj.owner_user_id,
                "context_type": obj.context_type.value,
                "claim_id": obj.metadata.get("claim_id"),
                "slot_id": obj.metadata.get("slot_id"),
                "source_revision": source_revision,
                "projection_revision": source_revision,
                "embedding_model": self.embedding_provider.model_name,
                "schema_version": "canonical_vector_projection_v2",
            },
        )

    def _write_scope_views(self, obj: ContextObject, record: ProjectionRecord) -> None:
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
            directory = (
                directory
                / self._segment(scope_ref.id)
                / self._segment(metadata.get("claim_id", "unknown"))
            )
            self._write_revisioned_view(directory, self._view_reference(obj, record))

    def _write_taxonomy_view(self, obj: ContextObject, record: ProjectionRecord) -> None:
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

    def _write_revisioned_view(self, directory: Path, payload: dict[str, Any]) -> None:
        revision = int(payload["source_revision"])
        self._write_json_atomic(directory / f"rev-{revision}.json", payload)

    def _publish_view_currents(self, record: ProjectionRecord) -> None:
        for path in self.root.glob(f"views/**/rev-{record.source_revision}.json"):
            payload = self._read_json_optional(path)
            if payload is None or str(payload.get("claim_uri", "")) != record.claim_uri:
                continue
            current_path = path.parent / "current.json"
            current = self._read_json_optional(current_path) or {}
            if int(current.get("source_revision", 0) or 0) <= record.source_revision:
                self._write_json_atomic(current_path, payload)

    def _view_reference(self, obj: ContextObject, record: ProjectionRecord) -> dict[str, Any]:
        metadata = dict(obj.metadata or {})
        return {
            "claim_uri": obj.uri,
            "slot_uri": record.slot_uri,
            "tenant_id": obj.tenant_id or "default",
            "slot_id": metadata.get("slot_id"),
            "claim_id": metadata.get("claim_id"),
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "current_claim_revision": record.current_claim_revision,
            "projection_record_path": str(self.record_store.record_path(record.claim_uri, record.source_revision)),
        }

    def _taxonomy_path(self, metadata: dict[str, Any]) -> Path:
        memory_type = str(metadata.get("memory_type", "memory"))
        revisions = metadata.get("revisions", []) or []
        current = dict(revisions[-1]) if revisions else {}
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

    def _manifest(
        self,
        record: ProjectionRecord,
        metadata: dict[str, Any],
        relations_uri: str,
    ) -> dict[str, Any]:
        return {
            **record.to_dict(),
            "memory_id": metadata.get("claim_id"),
            "slot_id": metadata.get("slot_id"),
            "claim_id": metadata.get("claim_id"),
            "projection_levels": ["L0", "L1", "L2"],
            "projections": [
                {
                    "claim_uri": record.claim_uri,
                    "slot_uri": record.slot_uri,
                    "source_revision": record.source_revision,
                    "projection_revision": record.projection_revision,
                    "projection_level": level,
                    "uri": uri,
                    "generator": self.GENERATOR,
                    "model_id": None,
                    "prompt_version": self.PROMPT_VERSION,
                    "created_at": record.created_at,
                }
                for level, uri in (("L0", record.l0_uri), ("L1", record.l1_uri), ("L2", record.l2_uri))
            ],
            "relation_projection_uri": relations_uri,
            "generator": self.GENERATOR,
            "model_id": None,
            "prompt_version": self.PROMPT_VERSION,
        }

    def _is_current(self, claim_uri: str, revision: int, expected_metadata: dict[str, Any]) -> bool:
        try:
            committed = read_committed_canonical(self.source_store, claim_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False
        if committed.from_before_image:
            return False
        metadata = dict(committed.object.metadata or {})
        return bool(
            metadata.get("canonical_kind") == "claim"
            and int(metadata.get("revision", 0)) == revision
            and str(metadata.get("claim_id", "")) == str(expected_metadata.get("claim_id", ""))
            and str(metadata.get("slot_id", "")) == str(expected_metadata.get("slot_id", ""))
        )

    def _remove_published_current(self, obj: ContextObject, record: ProjectionRecord) -> None:
        current = self.record_store.load_current(record.claim_uri)
        if current is not None and current.source_revision > record.source_revision:
            return
        self.index_store.delete_index(obj.uri)
        if self.vector_store is not None:
            self.vector_store.delete_vector(obj.uri)
        for path in self.root.glob("views/**/current.json"):
            payload = self._read_json_optional(path)
            if payload is None:
                continue
            if (
                str(payload.get("claim_uri", "")) == record.claim_uri
                and int(payload.get("source_revision", 0) or 0) == record.source_revision
            ):
                path.unlink(missing_ok=True)

    def _remove_record_current(self, record: ProjectionRecord) -> bool:
        current = self.record_store.load_current(record.claim_uri)
        if current is not None and current.source_revision > record.source_revision:
            return False
        self.index_store.delete_index(record.claim_uri)
        if self.vector_store is not None:
            self.vector_store.delete_vector(record.claim_uri)
        self._remove_view_currents(record)
        return True

    def _remove_view_currents(self, record: ProjectionRecord) -> None:
        for path in self.root.glob("views/**/current.json"):
            payload = self._read_json_optional(path)
            if payload is None:
                continue
            if (
                str(payload.get("claim_uri", "")) == record.claim_uri
                and int(payload.get("source_revision", 0) or 0) == record.source_revision
            ):
                path.unlink(missing_ok=True)

    def _reset_removed_components(self, record: ProjectionRecord) -> ProjectionRecord:
        resettable = {ProjectionStepStatus.RUNNING.value, ProjectionStepStatus.COMPLETED.value}
        return self.record_store.update(
            record,
            index_status=(
                ProjectionStepStatus.PENDING.value if record.index_status in resettable else record.index_status
            ),
            vector_status=(
                ProjectionStepStatus.PENDING.value if record.vector_status in resettable else record.vector_status
            ),
            scope_status=(
                ProjectionStepStatus.PENDING.value if record.scope_status in resettable else record.scope_status
            ),
            taxonomy_status=(
                ProjectionStepStatus.PENDING.value if record.taxonomy_status in resettable else record.taxonomy_status
            ),
        )

    def _notify(self, stage: str, claim_uri: str, revision: int) -> None:
        if self.test_hook is not None:
            self.test_hook(stage, claim_uri, revision)

    def _result(self, record: ProjectionRecord, status: str) -> ProjectionResult:
        self._emit(record)
        return ProjectionResult(
            record.claim_uri,
            record.source_revision,
            status,
            str(self.record_store.record_path(record.claim_uri, record.source_revision)),
        )

    def _emit(self, record: ProjectionRecord) -> None:
        if self.status_callback is not None:
            self.status_callback(record)

    def _segment(self, value: Any) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._:-]+", "-", str(value)).strip("-.")
        return cleaned[:120] or "unknown"

    def _read_json_optional(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)


class MemoryProjectionWorker:
    """Consume durable MemoryCommitted outbox entries idempotently."""

    def __init__(self, projector: CanonicalMemoryProjector, queue_store: QueueStore) -> None:
        self.projector = projector
        self.queue_store = queue_store

    def process_pending(self, limit: int = 10) -> dict[str, list[str]]:
        self.dispatch_outbox()
        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        for job in self.queue_store.lease("memory_projection", limit=limit):
            try:
                outbox = self._read_outbox(Path(str(job.payload["outbox_path"])))
                self._project_event(outbox, job.job_id, stale)
            except Exception as exc:
                retry = getattr(self.queue_store, "retry", None)
                if callable(retry):
                    retry(job.job_id, str(exc), max_retries=3, retryable=True)
                else:
                    self.queue_store.fail(job.job_id, str(exc))
                failed.append(job.job_id)
                continue
            self.queue_store.ack(job.job_id)
            processed.append(job.job_id)
        return {"processed": processed, "stale": stale, "failed": failed}

    def process_commit_group(
        self,
        group_id: str,
        *,
        transaction_ids: tuple[str, ...] = (),
    ) -> dict[str, list[str]]:
        """Project only one durable commit group, independently of unrelated queue jobs."""

        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        outbox_root = self.projector.root / "system" / "outbox"
        paths = (
            [outbox_root / f"{transaction_id}.json" for transaction_id in transaction_ids]
            if transaction_ids
            else sorted(outbox_root.glob("*.json"))
            if outbox_root.exists()
            else []
        )
        for path in paths:
            event_id = path.stem
            try:
                outbox = self._read_outbox(path)
                if str(outbox.get("commit_group_id", "")) != group_id:
                    if transaction_ids:
                        raise ValueError("projection outbox is not bound to the requested commit group")
                    continue
                job_id = f"outbox_{event_id}"
                self._project_event(outbox, job_id, stale)
                self.queue_store.ack(job_id)
                processed.append(job_id)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                failed.append(f"{event_id}:{type(exc).__name__}")
        return {"processed": processed, "stale": stale, "failed": failed}

    def _read_outbox(self, path: Path) -> dict[str, Any]:
        outbox = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(outbox, dict):
            raise ValueError("projection outbox must be an object")
        if outbox.get("event_type") != "MemoryCommitted" or outbox.get("status") != "committed":
            raise ValueError("projection job references a non-committed outbox event")
        return outbox

    def _project_event(self, outbox: dict[str, Any], job_id: str, stale: list[str]) -> None:
        for item in outbox.get("claim_revisions", []) or []:
            if not isinstance(item, dict) or not item.get("uri") or item.get("revision") is None:
                raise ValueError("projection outbox contains an invalid claim revision")
            result = self.projector.project(str(item["uri"]), int(item["revision"]))
            if result.status == "skipped_stale":
                stale.append(job_id)

    def dispatch_outbox(self) -> list[str]:
        dispatched: list[str] = []
        outbox_root = self.projector.root / "system" / "outbox"
        if not outbox_root.exists():
            return dispatched
        for path in sorted(outbox_root.glob("*.json")):
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if event.get("event_type") != "MemoryCommitted" or event.get("status") != "committed":
                continue
            transaction_id = str(event.get("transaction_id", ""))
            if not transaction_id:
                continue
            claim_revisions = event.get("claim_revisions", []) or []
            target_uri = str(claim_revisions[0].get("uri", "")) if claim_revisions else ""
            self.queue_store.enqueue(
                QueueJob(
                    job_id=f"outbox_{transaction_id}",
                    queue_name="memory_projection",
                    action="project_memory_committed",
                    target_uri=target_uri,
                    payload={
                        "transaction_id": transaction_id,
                        "outbox_path": str(path),
                        "operation_ids": [str(item) for item in event.get("operation_ids", []) or []],
                    },
                )
            )
            dispatched.append(transaction_id)
        return dispatched
