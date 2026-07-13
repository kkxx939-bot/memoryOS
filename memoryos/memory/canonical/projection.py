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
from memoryos.contextdb.store.source_store import (
    IndexStore,
    QueueIdempotencyConflictError,
    QueueJob,
    QueueLeaseIdentityError,
    QueueStore,
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.core.ids import require_safe_path_segment
from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    artifact_root_for,
    iter_current_head_uris,
    load_current_head,
)
from memoryos.memory.canonical.event import canonical_digest, canonical_json
from memoryos.memory.canonical.projection_proof import (
    PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION,
    PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    AuthoritativeProjectionIntegrityError,
    ProjectionProofStore,
    projection_publication_record_digest,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
    ProjectionRecordStore,
    ProjectionStatus,
    ProjectionStepStatus,
)
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalRead,
    CommittedStateIntegrityError,
    committed_content,
    committed_relations,
    read_committed_canonical,
)
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.operations.commit.outbox_envelope import (
    OUTBOX_EVENT_TYPE,
    OutboxIntegrityError,
    prepared_intent_digest,
    validate_outbox,
)
from memoryos.operations.commit.quarantine import quarantine_control_file
from memoryos.operations.commit.receipt import (
    ReceiptIntegrityError,
    load_transaction_receipt,
    receipt_snapshot,
)
from memoryos.providers.embedding import EmbeddingProvider, HashingEmbeddingProvider
from memoryos.workers.readiness import (
    readiness_for_source_store,
    require_source_store_ready,
    require_source_store_recovering,
)


@dataclass(frozen=True)
class ProjectionResult:
    claim_uri: str
    source_revision: int
    status: str
    record_path: str = ""
    projection_attempt_id: str = ""
    input_effect_hash: str = ""


class ProjectionOutboxIntegrityError(RuntimeError):
    """A projection outbox control file is corrupt or missing."""


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
        relation_store: RelationStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        record_store: ProjectionRecordStore | None = None,
        test_hook: Callable[[str, str, int], None] | None = None,
        status_callback: Callable[[ProjectionRecord], None] | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.root = Path(root)
        self.relation_store = relation_store
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
        try:
            committed = read_committed_canonical(
                self.source_store,
                claim_uri,
                self.relation_store,
            )
        except FileNotFoundError as exc:
            current = self.record_store.load_current(claim_uri)
            if current is not None:
                raise ProjectionIntegrityError(
                    "same revision has a different input effect or invalid commit proof"
                ) from exc
            raise
        obj = committed.object
        metadata = dict(obj.metadata or {})
        current_revision = int(metadata.get("revision", 0))
        if committed.from_before_image:
            return ProjectionResult(claim_uri, current_revision, "skipped_uncommitted")
        if metadata.get("canonical_kind") != "claim":
            return ProjectionResult(claim_uri, current_revision, "skipped_non_claim")
        materialized_current = materialized_current_revision_payload(metadata)
        domain_identity = self._projection_domain_identity(committed, materialized_current)
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
            with self.record_store.claim_lock(claim_uri):
                stale_current = self.record_store.load_current(claim_uri, source_revision=requested)
                if stale_current is not None:
                    self._remove_view_currents(stale_current)
                    self.record_store.clear_current_if(
                        claim_uri,
                        requested,
                        projection_attempt_id=stale_current.projection_attempt_id,
                        publish_token=stale_current.publish_token,
                        reason="canonical revision advanced beyond this projection",
                    )
            return ProjectionResult(claim_uri, requested, "skipped_stale")
        if requested > current_revision:
            raise ValueError("projection source revision is newer than canonical claim")

        input_effect_hash = self._input_effect_hash(committed, requested)
        existing = self.record_store.load_current(claim_uri, source_revision=requested)
        if existing is not None and not force:
            if existing.input_effect_hash != input_effect_hash:
                raise ProjectionIntegrityError("same projection revision has a different input effect")
            self._emit(existing)
            return self._result(existing, "projected")

        slot_uri = claim_uri.rsplit("/claims/", 1)[0]
        current_claim_revision = int(materialized_current["revision"])
        attempt_id = uuid.uuid4().hex
        base = f"{claim_uri}/projections/rev-{requested}/attempt-{attempt_id}"
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
            projection_attempt_id=attempt_id,
            input_effect_hash=input_effect_hash,
            l0_uri=l0_uri,
            l1_uri=l1_uri,
            l2_uri=l2_uri,
            relations_uri=relations_uri,
            manifest_uri=manifest_uri,
            current_claim_revision=current_claim_revision,
        )
        published_view_currents = False
        self._notify("after_read", claim_uri, requested)
        try:
            revision = self._revision_payload(metadata, current_claim_revision)
            l0, l1, l2 = self._layers(obj, metadata, revision, requested)
            relation_payload = [relation.to_dict() for relation in committed_relations(committed)]
            record = self.record_store.update(
                record,
                projected_content_digest=canonical_digest({"L0": l0, "L1": l1, "L2": l2}),
                projected_relation_digest=canonical_digest(relation_payload),
            )
            self._notify("before_artifacts", claim_uri, requested)
            self.source_store.write_content(l0_uri, l0)
            self.source_store.write_content(l1_uri, l1)
            self.source_store.write_content(l2_uri, l2)
            record = self.record_store.update(record, relation_status=ProjectionStepStatus.RUNNING.value)
            self.source_store.write_content(
                relations_uri,
                json.dumps(
                    {
                        **domain_identity,
                        "claim_uri": claim_uri,
                        "slot_uri": slot_uri,
                        "source_revision": requested,
                        "projection_revision": record.projection_revision,
                        "projection_attempt_id": record.projection_attempt_id,
                        "input_effect_hash": record.input_effect_hash,
                        "publish_token": record.publish_token,
                        "projected_content_digest": record.projected_content_digest,
                        "projected_relation_digest": record.projected_relation_digest,
                        "relations": relation_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
            record = self.record_store.update(record, relation_status=ProjectionStepStatus.COMPLETED.value)
            vector_embedding: list[float] | None = None
            if self.vector_store is None:
                record = self.record_store.update(record, vector_status=ProjectionStepStatus.SKIPPED.value)
            else:
                record = self.record_store.update(record, vector_status=ProjectionStepStatus.RUNNING.value)
                vector_embedding = self.embedding_provider.embed("\n".join((l0, l1)))
            self._notify("after_artifacts", claim_uri, requested)

            with self.record_store.claim_lock(claim_uri):
                if not self._is_current(claim_uri, requested, input_effect_hash):
                    stale = self.record_store.stale(record, "canonical revision or effect changed before publication")
                    return self._result(stale, "skipped_stale")
                current = self.record_store.load_current(claim_uri)
                if current is not None:
                    if current.source_revision > requested:
                        stale = self.record_store.stale(record, "newer projection revision is already current")
                        return self._result(stale, "skipped_stale")
                    if current.source_revision == requested:
                        if current.input_effect_hash != input_effect_hash:
                            raise ProjectionIntegrityError("same projection revision has a different input effect")
                        if current.projection_attempt_id != record.projection_attempt_id and not force:
                            self.record_store.stale(record, "equivalent projection attempt is already current")
                            self._emit(current)
                            return self._result(current, "projected")

                owned = self.record_store.load(
                    claim_uri,
                    requested,
                    projection_attempt_id=record.projection_attempt_id,
                )
                if owned is None or owned.status != ProjectionStatus.RUNNING.value:
                    raise ProjectionIntegrityError("projection attempt lost publication eligibility")
                record = owned
                self._notify("before_publish", claim_uri, requested)
                projection_obj = self._projection_object(
                    obj,
                    metadata,
                    record,
                    domain_identity=domain_identity,
                    layers=ContextLayers(l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri),
                )
                if self.vector_store is not None:
                    assert vector_embedding is not None
                    try:
                        self._publish_vector(projection_obj, vector_embedding, record)
                    except Exception:
                        record = self.record_store.update(record, vector_status=ProjectionStepStatus.FAILED.value)
                        raise
                    record = self.record_store.update(record, vector_status=ProjectionStepStatus.COMPLETED.value)

                record = self.record_store.update(record, index_status=ProjectionStepStatus.RUNNING.value)
                try:
                    self.index_store.upsert_index(projection_obj, content="\n".join((l0, l1, l2)))
                except Exception:
                    record = self.record_store.update(record, index_status=ProjectionStepStatus.FAILED.value)
                    raise
                record = self.record_store.update(record, index_status=ProjectionStepStatus.COMPLETED.value)
                self._notify("after_index", claim_uri, requested)

                record = self.record_store.update(record, scope_status=ProjectionStepStatus.RUNNING.value)
                self._write_scope_views(projection_obj, record)
                record = self.record_store.update(record, scope_status=ProjectionStepStatus.COMPLETED.value)
                record = self.record_store.update(record, taxonomy_status=ProjectionStepStatus.RUNNING.value)
                self._write_taxonomy_view(projection_obj, record)
                record = self.record_store.update(record, taxonomy_status=ProjectionStepStatus.COMPLETED.value)

                if not self._is_current(claim_uri, requested, input_effect_hash):
                    stale = self.record_store.stale(record, "canonical revision or effect changed during publication")
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
                        self._manifest(
                            completed_preview,
                            metadata,
                            relations_uri,
                            domain_identity=domain_identity,
                        ),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                )
                self._notify("before_view_publish", claim_uri, requested)
                self._publish_view_currents(completed_preview)
                published_view_currents = True
                self._notify("after_view_publish", claim_uri, requested)
                record = self.record_store.promote(completed_preview, replace_same_effect=force)
                if record.projection_attempt_id != completed_preview.projection_attempt_id:
                    self._remove_view_currents(completed_preview)
                    return self._result(record, "projected")
                self._notify("after_publish", claim_uri, requested)
                if not self._is_current(claim_uri, requested, input_effect_hash):
                    self._remove_view_currents(record)
                    self.record_store.clear_current_if(
                        claim_uri,
                        requested,
                        projection_attempt_id=record.projection_attempt_id,
                        publish_token=record.publish_token,
                        reason="canonical revision or effect changed after publication",
                    )
                    stale = (
                        self.record_store.load(
                            claim_uri,
                            requested,
                            projection_attempt_id=record.projection_attempt_id,
                        )
                        or record
                    )
                    return self._result(stale, "skipped_stale")
            return self._result(record, "projected")
        except Exception as exc:
            latest = (
                self.record_store.load(
                    claim_uri,
                    requested,
                    projection_attempt_id=record.projection_attempt_id,
                )
                or record
            )
            current = self.record_store.load_current(claim_uri)
            if current is not None and current.projection_attempt_id == record.projection_attempt_id:
                self._emit(current)
                raise
            if published_view_currents:
                self._remove_view_currents(latest)
            failed = self.record_store.fail(latest, f"{type(exc).__name__}: {exc}", retryable=True)
            self._emit(failed)
            raise

    def rebuild(self, *, clear_views: bool = True) -> dict[str, int]:
        projected = 0
        skipped = 0
        # Rebuild is also called by recovery/admin code outside ContextDB.
        # Complete every non-mutating authoritative/derived-input check before
        # deleting a view, retiring a pointer, or replacing an index/vector
        # row.  Outbox/queue boundary validation remains the worker/caller's
        # responsibility because the projector deliberately has no queue.
        ProjectionProofStore(self.root).validate_all()
        artifact_root = artifact_root_for(self.source_store)
        claim_uris = tuple(iter_current_head_uris(artifact_root, kinds=("claim",)) if artifact_root is not None else ())
        committed_claims = set(claim_uris)

        def uncommitted_canonical_row(uri: str, metadata: dict[str, Any] | None) -> bool:
            if uri in committed_claims:
                return False
            row_metadata = dict(metadata or {})
            if (
                is_canonical_memory_uri(uri)
                or str(row_metadata.get("canonical_kind") or "") in {"slot", "claim", "pending_proposal"}
                or str(row_metadata.get("schema_version") or "").startswith("canonical_")
            ):
                return True
            try:
                return is_canonical_memory_object(self.source_store.read_object(uri))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return False

        committed_revisions: dict[str, int] = {}
        for claim_uri in claim_uris:
            committed = read_committed_canonical(
                self.source_store,
                claim_uri,
                self.relation_store,
            )
            committed_revisions[claim_uri] = int(dict(committed.object.metadata or {}).get("revision", 0))
        current_records = self.record_store.iter_current()
        indexed_to_remove = tuple(
            indexed_uri
            for indexed_uri in tuple(self.index_store.indexed_uris())
            if uncommitted_canonical_row(
                indexed_uri,
                self.index_store.get_index_metadata(indexed_uri),
            )
        )
        vector_to_remove = (
            tuple(
                vector_uri
                for vector_uri in tuple(self.vector_store.vector_uris())
                if uncommitted_canonical_row(
                    vector_uri,
                    self.vector_store.get_vector_metadata(vector_uri),
                )
            )
            if self.vector_store is not None
            else ()
        )

        if clear_views:
            for name in ("scope", "taxonomy"):
                path = self.root / "views" / name
                if path.exists():
                    shutil.rmtree(path)
        retired = 0
        for record in current_records:
            if committed_revisions.get(record.claim_uri) == record.source_revision:
                continue
            with self.record_store.claim_lock(record.claim_uri):
                self._remove_view_currents(record)
                if self.record_store.clear_current_if(
                    record.claim_uri,
                    record.source_revision,
                    projection_attempt_id=record.projection_attempt_id,
                    publish_token=record.publish_token,
                    reason="projection current does not have an equal committed Claim head",
                ):
                    retired += 1
        for indexed_uri in indexed_to_remove:
            self.index_store.delete_index(indexed_uri)
            if self.vector_store is not None:
                self.vector_store.delete_vector(indexed_uri)
        if self.vector_store is not None:
            for vector_uri in vector_to_remove:
                self.vector_store.delete_vector(vector_uri)
        for claim_uri in claim_uris:
            result = self.project(claim_uri, force=True)
            if result.status == "projected":
                projected += 1
            else:
                skipped += 1
        return {"projected": projected, "skipped": skipped, "retired": retired}

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
        display_fields = dict(qualifiers.get("display_fields", {}) or {})
        display_field_evidence_refs = dict(qualifiers.get("display_field_evidence_refs", {}) or {})
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
                "display_field_evidence_refs": display_field_evidence_refs,
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

    def _projection_domain_identity(
        self,
        committed: CommittedCanonicalRead,
        current_revision: dict[str, Any],
    ) -> dict[str, Any]:
        obj = committed.object
        metadata = dict(obj.metadata or {})
        head = dict(committed.head or {})
        if (
            not head
            or head.get("uri") != obj.uri
            or head.get("canonical_kind") != "claim"
            or head.get("tenant_id") != str(obj.tenant_id or "default")
            or head.get("owner_user_id") != str(obj.owner_user_id or "")
            or not str(head.get("current_transaction_id") or "")
            or not str(head.get("receipt_digest") or "")
        ):
            raise ProjectionIntegrityError("projection Source has no complete current-head identity")
        state = str(current_revision.get("state") or "")
        if not state or state != str(metadata.get("state") or ""):
            raise ProjectionIntegrityError("projection Claim state mirror is inconsistent")
        return {
            "claim_uri": obj.uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "owner_user_id": str(obj.owner_user_id or ""),
            "canonical_kind": "claim",
            "claim_state": state,
            "current_transaction_id": str(head["current_transaction_id"]),
            "current_receipt_digest": str(head["receipt_digest"]),
            "current_claim_revision": int(current_revision["revision"]),
        }

    def _projection_object(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        record: ProjectionRecord,
        *,
        domain_identity: dict[str, Any],
        layers: ContextLayers,
    ) -> ContextObject:
        projected = ContextObject.from_dict(obj.to_dict())
        projected.layers = layers
        projected.metadata = {
            **metadata,
            **domain_identity,
            "projection_source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "projection_input_effect_hash": record.input_effect_hash,
            "projection_publish_token": record.publish_token,
            "projection_content_digest": record.projected_content_digest,
            "projection_relation_digest": record.projected_relation_digest,
            "current_claim_revision": record.current_claim_revision,
            "projection_manifest_uri": record.manifest_uri,
            "projection_record_path": str(self.record_store.attempt_path_for(record)),
        }
        return projected

    def _publish_vector(
        self,
        obj: ContextObject,
        embedding: list[float],
        record: ProjectionRecord,
    ) -> None:
        assert self.vector_store is not None
        self.vector_store.upsert_vector(
            obj.uri,
            embedding,
            metadata={
                "claim_uri": obj.uri,
                "tenant_id": obj.tenant_id or "default",
                "owner_user_id": obj.owner_user_id or "",
                "context_type": obj.context_type.value,
                "claim_id": obj.metadata.get("claim_id"),
                "slot_id": obj.metadata.get("slot_id"),
                "canonical_kind": obj.metadata.get("canonical_kind"),
                "claim_state": obj.metadata.get("claim_state"),
                "current_transaction_id": obj.metadata.get("current_transaction_id"),
                "current_receipt_digest": obj.metadata.get("current_receipt_digest"),
                "current_claim_revision": obj.metadata.get("current_claim_revision"),
                "source_revision": record.source_revision,
                "projection_revision": record.projection_revision,
                "projection_attempt_id": record.projection_attempt_id,
                "input_effect_hash": record.input_effect_hash,
                "publish_token": record.publish_token,
                "projected_content_digest": record.projected_content_digest,
                "projected_relation_digest": record.projected_relation_digest,
                "embedding_model": self.embedding_provider.model_name,
                "schema_version": "canonical_vector_projection_v4",
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
            directory = directory / self._segment(scope_ref.id) / self._segment(metadata.get("claim_id", "unknown"))
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
        attempt_id = str(payload["projection_attempt_id"])
        self._write_json_atomic(directory / f"rev-{revision}-attempt-{attempt_id}.json", payload)

    def _publish_view_currents(self, record: ProjectionRecord) -> None:
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

    def _view_reference(self, obj: ContextObject, record: ProjectionRecord) -> dict[str, Any]:
        metadata = dict(obj.metadata or {})
        return {
            "claim_uri": obj.uri,
            "slot_uri": record.slot_uri,
            "tenant_id": obj.tenant_id or "default",
            "owner_user_id": obj.owner_user_id or "",
            "canonical_kind": metadata.get("canonical_kind"),
            "claim_state": metadata.get("claim_state"),
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

    def _taxonomy_path(self, metadata: dict[str, Any]) -> Path:
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

    def _manifest(
        self,
        record: ProjectionRecord,
        metadata: dict[str, Any],
        relations_uri: str,
        *,
        domain_identity: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **record.to_dict(),
            **domain_identity,
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
                    "projection_attempt_id": record.projection_attempt_id,
                    "input_effect_hash": record.input_effect_hash,
                    "publish_token": record.publish_token,
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

    def _is_current(self, claim_uri: str, revision: int, expected_effect_hash: str) -> bool:
        try:
            committed = read_committed_canonical(
                self.source_store,
                claim_uri,
                self.relation_store,
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False
        if committed.from_before_image:
            return False
        metadata = dict(committed.object.metadata or {})
        return bool(
            not committed.from_before_image
            and metadata.get("canonical_kind") == "claim"
            and int(metadata.get("revision", 0)) == revision
            and self._input_effect_hash(committed, revision) == expected_effect_hash
        )

    def _remove_view_currents(self, record: ProjectionRecord) -> None:
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

    def _input_effect_hash(
        self,
        committed: CommittedCanonicalRead,
        source_revision: int,
    ) -> str:
        obj = committed.object
        content = committed_content(committed)
        relations = sorted(
            (relation.to_dict() for relation in committed_relations(committed)),
            key=canonical_json,
        )
        return canonical_digest(
            {
                "claim_uri": obj.uri,
                "source_revision": source_revision,
                "object": obj.to_dict(),
                "content": content,
                "relations": relations,
            }
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
            str(self.record_store.attempt_path_for(record)),
            record.projection_attempt_id,
            record.input_effect_hash,
        )

    def _emit(self, record: ProjectionRecord) -> None:
        if self.status_callback is not None:
            self.status_callback(record)

    def _segment(self, value: Any) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._:-]+", "-", str(value)).strip("-.")
        return cleaned[:120] or "unknown"

    def _read_json_optional(self, path: Path) -> dict[str, Any] | None:
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

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        if path.is_symlink():
            raise ProjectionIntegrityError(f"projection view state cannot be a symbolic link: {path.name}")
        try:
            atomic_write_json(path, payload, artifact_root=self.root)
        except ValueError as exc:
            raise ProjectionIntegrityError(f"projection view state publication is unsafe: {path.name}") from exc


class MemoryProjectionWorker:
    """Consume durable MemoryCommitted outbox entries idempotently."""

    def __init__(
        self,
        projector: CanonicalMemoryProjector,
        queue_store: QueueStore,
        *,
        worker_id: str | None = None,
    ) -> None:
        self.projector = projector
        self.queue_store = queue_store
        self.proof_store = ProjectionProofStore(projector.root)
        self.worker_id = worker_id or f"memory-projection:{os.getpid()}:{uuid.uuid4().hex}"
        self.last_quarantined: list[str] = []

    def process_pending(
        self,
        limit: int = 10,
        *,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, list[str]]:
        require_source_store_ready(self.projector.source_store)
        return self._process_pending(
            limit,
            lease_seconds=lease_seconds,
            max_retries=max_retries,
        )

    def _process_pending_during_startup(
        self,
        limit: int = 10,
        *,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, list[str]]:
        require_source_store_recovering(self.projector.source_store)
        return self._process_pending(
            limit,
            lease_seconds=lease_seconds,
            max_retries=max_retries,
        )

    def _process_pending(
        self,
        limit: int,
        *,
        lease_seconds: int,
        max_retries: int,
    ) -> dict[str, list[str]]:
        self.last_quarantined = []
        self._validate_authoritative_projection_proofs()
        self.dispatch_outbox()
        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        dead_letter: list[str] = []
        quarantine: list[str] = []
        released: list[str] = []
        jobs = self.queue_store.lease(
            "memory_projection",
            lease_owner=self.worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        for position, job in enumerate(jobs):
            try:
                outbox = self._load_projection_job_outbox(job)
                self._project_event(outbox, job.job_id, stale)
                self._assert_projection_job_identity_unchanged(job)
                self._ensure_projection_publication(outbox, job)
                self._assert_projection_job_identity_unchanged(job)
                self.queue_store.ack(job)
            except QueueLeaseIdentityError as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue",
                    identifiers={"job_id": job.job_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause=type(exc).__name__,
                    )
                )
                self._quarantine_projection_identity_conflict(job, exc)
                failed.append(job.job_id)
                quarantine.append(job.job_id)
                break
            except (ProjectionOutboxIntegrityError, AuthoritativeProjectionIntegrityError) as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact=(
                        "projection_proof"
                        if isinstance(exc, AuthoritativeProjectionIntegrityError)
                        else "projection_outbox_or_queue"
                    ),
                    identifiers={"job_id": job.job_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause=type(exc).__name__,
                    )
                )
                self.queue_store.quarantine(job, type(exc).__name__)
                failed.append(job.job_id)
                quarantine.append(job.job_id)
                break
            except Exception as exc:
                settled = self.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=True,
                )
                failed.append(job.job_id)
                if settled.status == "dead_letter":
                    dead_letter.append(job.job_id)
                    self._mark_authoritative_integrity_failure(
                        exc,
                        artifact="projection_queue_dead_letter",
                        identifiers={"job_id": job.job_id},
                    )
                    released.extend(
                        self._release_unattempted_projection_jobs(
                            jobs[position + 1 :],
                            cause="projection_queue_dead_letter",
                        )
                    )
                    break
                self._extend_unattempted_projection_leases(
                    jobs[position + 1 :],
                    lease_seconds=lease_seconds,
                )
                continue
            processed.append(job.job_id)
            self._extend_unattempted_projection_leases(
                jobs[position + 1 :],
                lease_seconds=lease_seconds,
            )
        return {
            "processed": processed,
            "stale": stale,
            "failed": failed,
            "dead_letter": dead_letter,
            "quarantine": [*self.last_quarantined, *quarantine],
            "released": released,
        }

    def _validate_authoritative_projection_proofs(self) -> None:
        """Reverse-bind immutable proofs without inspecting rebuildable views."""

        try:
            self.proof_store.validate_all()
            for publication in self.proof_store.iter_publications():
                transaction_id = str(publication["transaction_id"])
                job = self.queue_store.get(f"outbox_{transaction_id}")
                if job is None:
                    raise AuthoritativeProjectionIntegrityError(
                        "projection publication receipt has no durable queue identity"
                    )
                outbox = self._load_projection_job_outbox(
                    job,
                    expected_transaction_id=transaction_id,
                )
                receipt = self._load_bound_receipt(
                    outbox,
                    transaction_id,
                    str(publication["commit_group_id"]),
                )
                self._verify_projection_publication_boundary(
                    publication,
                    outbox,
                    receipt,
                    job,
                )
                completion = self.proof_store.load_completion(transaction_id)
                if completion is not None and job.status != "done":
                    raise AuthoritativeProjectionIntegrityError(
                        "projection completion proof is detached from terminal queue state"
                    )
        except (AuthoritativeProjectionIntegrityError, ProjectionOutboxIntegrityError) as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact=(
                    "projection_proof"
                    if isinstance(exc, AuthoritativeProjectionIntegrityError)
                    else "projection_outbox_or_queue"
                ),
            )
            raise

    def _mark_authoritative_integrity_failure(
        self,
        error: BaseException,
        *,
        artifact: str,
        identifiers: dict[str, Any] | None = None,
    ) -> None:
        readiness = readiness_for_source_store(self.projector.source_store)
        mark_not_ready = getattr(readiness, "mark_not_ready", None)
        if not callable(mark_not_ready):
            return
        details: dict[str, Any] = {
            "artifact": artifact,
            "error_type": type(error).__name__,
            **dict(identifiers or {}),
        }
        mark_not_ready(
            f"authoritative projection integrity failure: {type(error).__name__}: {error}",
            details=details,
        )

    def _release_unattempted_projection_jobs(
        self,
        jobs: list[QueueJob],
        *,
        cause: str,
    ) -> list[str]:
        """Release the remainder of an already-leased batch without retry cost."""

        released: list[str] = []
        for job in jobs:
            try:
                settled = self.queue_store.release(
                    job,
                    f"batch aborted before attempt after {cause}",
                )
            except Exception as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue_release",
                    identifiers={"job_id": job.job_id},
                )
                raise ProjectionOutboxIntegrityError(
                    "projection batch abort could not release an unattempted lease"
                ) from exc
            if settled.status != "pending":
                error = ProjectionOutboxIntegrityError("projection batch abort released a job to a non-pending state")
                self._mark_authoritative_integrity_failure(
                    error,
                    artifact="projection_queue_release",
                    identifiers={"job_id": job.job_id},
                )
                raise error
            released.append(job.job_id)
        return released

    def _extend_unattempted_projection_leases(
        self,
        jobs: list[QueueJob],
        *,
        lease_seconds: int,
    ) -> None:
        """Keep an already leased fail-stop batch owned while earlier work runs."""

        for job in jobs:
            self.queue_store.extend(job, lease_seconds=lease_seconds)

    def _assert_projection_job_identity_unchanged(self, job: QueueJob) -> None:
        """Re-read a leased job so post-preflight queue tamper cannot publish."""

        try:
            persisted = self.queue_store.get(job.job_id)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise QueueLeaseIdentityError(
                f"projection queue identity is unreadable while leased: {job.job_id}"
            ) from exc
        if persisted is None or (
            persisted.queue_name != job.queue_name
            or persisted.action != job.action
            or persisted.target_uri != job.target_uri
            or persisted.payload != job.payload
        ):
            raise QueueLeaseIdentityError(f"projection queue immutable identity changed while leased: {job.job_id}")

    def _quarantine_projection_identity_conflict(
        self,
        job: QueueJob,
        error: QueueLeaseIdentityError,
    ) -> None:
        try:
            settled = self.queue_store.quarantine_identity_conflict(
                job,
                type(error).__name__,
            )
        except Exception as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="projection_queue_quarantine",
                identifiers={"job_id": job.job_id},
            )
            raise ProjectionOutboxIntegrityError("corrupt projection queue identity could not be quarantined") from exc
        if settled.status != "quarantine":
            failure = ProjectionOutboxIntegrityError("corrupt projection queue identity was not quarantined")
            self._mark_authoritative_integrity_failure(
                failure,
                artifact="projection_queue_quarantine",
                identifiers={"job_id": job.job_id},
            )
            raise failure

    def verify_current_projections(self) -> dict[str, int]:
        artifact_root = artifact_root_for(self.projector.source_store)
        claim_uris = tuple(iter_current_head_uris(artifact_root, kinds=("claim",)) if artifact_root is not None else ())
        current_records = {record.claim_uri: record for record in self.projector.record_store.iter_current()}
        if set(current_records) != set(claim_uris):
            dangling = sorted(set(current_records) - set(claim_uris))
            missing = sorted(set(claim_uris) - set(current_records))
            raise ProjectionIntegrityError(
                f"projection current/head closure mismatch; dangling={dangling}; missing={missing}"
            )
        verified = 0
        for claim_uri in claim_uris:
            committed = read_committed_canonical(
                self.projector.source_store,
                claim_uri,
                self.projector.relation_store,
            )
            revision = int(dict(committed.object.metadata or {}).get("revision", 0))
            if current_records[claim_uri].source_revision != revision:
                raise ProjectionIntegrityError(
                    f"projection current revision does not match committed Claim head: {claim_uri}"
                )
            self._verify_claim_projection(claim_uri, revision)
            verified += 1
        return {"verified": verified}

    def validate_projection_proofs(self) -> dict[str, int]:
        """Reverse-bind every proof artifact, including crash-orphaned receipts."""

        structural = self.proof_store.validate_all()
        verified = 0
        for publication in self.proof_store.iter_publications():
            transaction_id = str(publication["transaction_id"])
            group_id = str(publication["commit_group_id"])
            result = self.verify_commit_group_completion(group_id, (transaction_id,))
            failures = [str(item) for item in result["failures"]]
            if failures:
                raise ProjectionIntegrityError(
                    f"projection publication has no durable completion: {transaction_id}: {failures}"
                )
            if len(result["proofs"]) != 1:
                raise ProjectionIntegrityError("projection transaction has an invalid completion proof count")
            verified += 1
        final = self.proof_store.validate_all()
        return {
            "publications": final["publications"],
            "completions": final["completions"],
            "verified": verified,
            "completed_during_validation": final["completions"] - structural["completions"],
        }

    def migrate_legacy_completion_proof(
        self,
        group_id: str,
        transaction_id: str,
        legacy_proof: dict[str, Any],
    ) -> bool:
        """Promote a validated v1 group result into the create-only proof DAG.

        The v1 result was emitted only after the old verifier had checked the
        live index/vector/views, but it did not persist their metadata digests.
        Migration binds that attestation digest to the immutable receipt and
        all still-present revision-scoped projection artifacts.  It never
        invents proof when the legacy result or historical artifacts are
        missing.
        """

        if self.proof_store.load_publication(transaction_id) is not None:
            return False
        if not isinstance(legacy_proof, dict) or legacy_proof.get("schema_version") != "projection_completion_proof_v1":
            raise ProjectionIntegrityError("legacy projection completion proof schema is unsupported")
        legacy_core = {key: value for key, value in legacy_proof.items() if key != "proof_digest"}
        legacy_digest = str(legacy_proof.get("proof_digest") or "")
        if legacy_digest != canonical_digest(legacy_core):
            raise ProjectionIntegrityError("legacy projection completion proof digest is corrupt")
        job_id = f"outbox_{transaction_id}"
        job = self.queue_store.get(job_id)
        if job is None or job.status != "done":
            raise ProjectionIntegrityError("legacy projection completion queue state is not durable")
        outbox = self._load_projection_job_outbox(job, expected_transaction_id=transaction_id)
        receipt = self._load_bound_receipt(outbox, transaction_id, group_id)
        if (
            legacy_proof.get("commit_group_id") != group_id
            or legacy_proof.get("transaction_id") != transaction_id
            or legacy_proof.get("job_id") != job_id
            or legacy_proof.get("queue_status") != "done"
            or legacy_proof.get("outbox_digest") != outbox.get("outbox_digest")
            or legacy_proof.get("receipt_digest") != receipt.get("receipt_digest")
        ):
            raise ProjectionIntegrityError("legacy projection completion proof crosses its transaction boundary")
        raw_claims = legacy_proof.get("claims")
        if not isinstance(raw_claims, list):
            raise ProjectionIntegrityError("legacy projection completion proof has no Claim set")
        legacy_by_identity = {
            (str(item.get("claim_uri") or ""), int(item.get("source_revision", 0))): item
            for item in raw_claims
            if isinstance(item, dict)
        }
        expected_identities = {(str(item["uri"]), int(item["revision"])) for item in self._claim_revisions(outbox)}
        if set(legacy_by_identity) != expected_identities:
            raise ProjectionIntegrityError("legacy projection completion Claim set differs from outbox")
        claims = [
            self._migrated_legacy_claim_proof(
                legacy_by_identity[identity],
                receipt,
                legacy_digest,
            )
            for identity in sorted(expected_identities)
        ]
        publication_core = {
            "schema_version": PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "commit_group_id": group_id,
            "transaction_id": transaction_id,
            "job_id": job_id,
            "tenant_id": str(receipt["tenant_id"]),
            "user_id": str(receipt["user_id"]),
            "queue_identity_digest": self._queue_identity_digest(job),
            "outbox_digest": str(outbox["outbox_digest"]),
            "receipt_digest": str(receipt["receipt_digest"]),
            "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
            "operation_ids": [str(item) for item in outbox["operation_ids"]],
            "claim_revisions": [{"uri": item["claim_uri"], "revision": item["source_revision"]} for item in claims],
            "claims": claims,
            "migration_source_schema": "projection_completion_proof_v1",
            "legacy_completion_proof_digest": legacy_digest,
        }
        publication = self.proof_store.ensure_publication(
            {
                **publication_core,
                "publication_digest": canonical_digest(publication_core),
            }
        )
        self._verify_projection_publication(publication, outbox, receipt, job)
        return True

    def _migrated_legacy_claim_proof(
        self,
        legacy: dict[str, Any],
        receipt: dict[str, Any],
        legacy_digest: str,
    ) -> dict[str, Any]:
        claim_uri = str(legacy.get("claim_uri") or "")
        source_revision = int(legacy.get("source_revision", 0))
        attempt_id = str(legacy.get("projection_attempt_id") or "")
        record = self.projector.record_store.load(
            claim_uri,
            source_revision,
            projection_attempt_id=attempt_id,
        )
        if record is None or record.status not in {
            ProjectionStatus.COMPLETED.value,
            ProjectionStatus.STALE.value,
        }:
            raise ProjectionIntegrityError("legacy projection attempt record is missing")
        for key, expected in (
            ("claim_uri", record.claim_uri),
            ("source_revision", record.source_revision),
            ("projection_revision", record.projection_revision),
            ("projection_attempt_id", record.projection_attempt_id),
            ("input_effect_hash", record.input_effect_hash),
            ("publish_token", record.publish_token),
            ("projected_content_digest", record.projected_content_digest),
            ("projected_relation_digest", record.projected_relation_digest),
        ):
            if legacy.get(key) != expected:
                raise ProjectionIntegrityError("legacy projection Claim proof differs from attempt record")
        legacy_record_digest = str(legacy.get("record_digest") or "")
        if len(legacy_record_digest) != 64:
            raise ProjectionIntegrityError("legacy projection record digest is invalid")
        try:
            snapshot = receipt_snapshot(receipt, claim_uri)
            obj = ContextObject.from_dict(dict(snapshot["object"]))
            metadata = dict(obj.metadata or {})
            materialized = materialized_current_revision_payload(metadata)
        except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
            raise ProjectionIntegrityError("legacy projection has no immutable Source snapshot") from exc
        domain_identity = {
            "claim_uri": claim_uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "owner_user_id": str(obj.owner_user_id or ""),
            "canonical_kind": "claim",
            "claim_state": str(materialized.get("state") or ""),
            "current_transaction_id": str(receipt["transaction_id"]),
            "current_receipt_digest": str(receipt["receipt_digest"]),
            "current_claim_revision": int(materialized["revision"]),
        }
        historical_committed = CommittedCanonicalRead(obj, receipt=receipt)
        if (
            int(metadata.get("revision", 0)) != source_revision
            or self.projector._input_effect_hash(historical_committed, source_revision) != record.input_effect_hash
        ):
            raise ProjectionIntegrityError("legacy projection input effect differs from receipt")
        layer_values = {
            "L0": self.projector.source_store.read_content(record.l0_uri),
            "L1": self.projector.source_store.read_content(record.l1_uri),
            "L2": self.projector.source_store.read_content(record.l2_uri),
        }
        try:
            relation_payload = json.loads(self.projector.source_store.read_content(record.relations_uri))
            manifest = json.loads(self.projector.source_store.read_content(record.manifest_uri))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProjectionIntegrityError("legacy projection artifact is malformed") from exc
        if not isinstance(relation_payload, dict) or not isinstance(manifest, dict):
            raise ProjectionIntegrityError("legacy projection artifact is not an object")
        if record.projected_content_digest != canonical_digest(
            layer_values
        ) or record.projected_relation_digest != canonical_digest(relation_payload.get("relations", [])):
            raise ProjectionIntegrityError("legacy projection artifact digest is corrupt")
        self._assert_projection_identity(
            relation_payload,
            record,
            label="legacy relation",
            domain_identity=domain_identity,
        )
        self._assert_projection_identity(
            manifest,
            record,
            label="legacy manifest",
            domain_identity=domain_identity,
        )
        attested = lambda component: canonical_digest(  # noqa: E731
            {
                "schema_version": "legacy_projection_component_attestation_v1",
                "legacy_completion_proof_digest": legacy_digest,
                "component": component,
            }
        )
        claim_core = {
            "claim_uri": record.claim_uri,
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
            "record_digest": legacy_record_digest,
            "publication_record_digest": projection_publication_record_digest(record),
            "layer_uris": {
                "L0": record.l0_uri,
                "L1": record.l1_uri,
                "L2": record.l2_uri,
                "manifest": record.manifest_uri,
                "relations": record.relations_uri,
            },
            "layer_digests": {name: canonical_digest(value) for name, value in layer_values.items()},
            "relation_artifact_digest": canonical_digest(relation_payload),
            "manifest_digest": canonical_digest(manifest),
            "index_metadata_digest": attested("index"),
            "vector_metadata_digest": attested("vector"),
            "scope_view_digests": [attested("scope")],
            "taxonomy_view_digests": [attested("taxonomy")],
            "domain_identity": domain_identity,
            "migration_source_schema": "projection_completion_proof_v1",
            "legacy_completion_proof_digest": legacy_digest,
        }
        return {**claim_core, "claim_proof_digest": canonical_digest(claim_core)}

    def process_commit_group(
        self,
        group_id: str,
        *,
        transaction_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Project only one durable commit group, independently of unrelated queue jobs."""

        readiness = readiness_for_source_store(self.projector.source_store)
        state_obj = getattr(readiness, "state", None)
        state = str(getattr(state_obj, "value", state_obj or ""))
        if state != "RECOVERING":
            require_source_store_ready(self.projector.source_store)
        self._validate_authoritative_projection_proofs()

        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        quarantine: list[str] = []
        released: list[str] = []
        completion_proofs: list[dict[str, Any]] = []
        terminal_abort = False
        self.last_quarantined = []
        self.dispatch_outbox()
        if transaction_ids:
            job_ids = tuple(f"outbox_{transaction_id}" for transaction_id in transaction_ids)
        else:
            outbox_root = self.projector.root / "system" / "outbox"
            selected: list[str] = []
            for path in sorted(outbox_root.glob("*.json")) if outbox_root.exists() else []:
                try:
                    event = self._read_outbox(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if str(event.get("commit_group_id", "")) == group_id:
                    selected.append(f"outbox_{path.stem}")
            job_ids = tuple(selected)
        if not job_ids:
            return {
                "processed": processed,
                "stale": stale,
                "failed": failed,
                "quarantine": self.last_quarantined,
                "released": released,
            }
        lease_seconds = 300
        jobs = self.queue_store.lease(
            "memory_projection",
            lease_owner=self.worker_id,
            limit=len(job_ids),
            lease_seconds=lease_seconds,
            job_ids=job_ids,
        )
        for position, job in enumerate(jobs):
            try:
                outbox = self._load_projection_job_outbox(job)
                if str(outbox.get("commit_group_id", "")) != group_id:
                    if transaction_ids:
                        raise ValueError("projection outbox is not bound to the requested commit group")
                    released.extend(
                        self._release_unattempted_projection_jobs(
                            [job],
                            cause="commit_group_filter_mismatch",
                        )
                    )
                    continue
                self._project_event(outbox, job.job_id, stale)
                self._assert_projection_job_identity_unchanged(job)
                self._ensure_projection_publication(outbox, job)
                self._assert_projection_job_identity_unchanged(job)
                self.queue_store.ack(job)
                processed.append(job.job_id)
            except QueueLeaseIdentityError as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue",
                    identifiers={"job_id": job.job_id, "commit_group_id": group_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause=type(exc).__name__,
                    )
                )
                self._quarantine_projection_identity_conflict(job, exc)
                failed.append(f"{job.job_id}:{type(exc).__name__}")
                quarantine.append(job.job_id)
                terminal_abort = True
                break
            except (ProjectionOutboxIntegrityError, AuthoritativeProjectionIntegrityError) as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact=(
                        "projection_proof"
                        if isinstance(exc, AuthoritativeProjectionIntegrityError)
                        else "projection_outbox_or_queue"
                    ),
                    identifiers={"job_id": job.job_id, "commit_group_id": group_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause=type(exc).__name__,
                    )
                )
                self.queue_store.quarantine(job, type(exc).__name__)
                failed.append(f"{job.job_id}:{type(exc).__name__}")
                quarantine.append(job.job_id)
                terminal_abort = True
                break
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                settled = self.queue_store.retry(job, type(exc).__name__, max_retries=3, retryable=True)
                if settled.status == "dead_letter":
                    self._mark_authoritative_integrity_failure(
                        exc,
                        artifact="projection_queue_dead_letter",
                        identifiers={"job_id": job.job_id, "commit_group_id": group_id},
                    )
                    released.extend(
                        self._release_unattempted_projection_jobs(
                            jobs[position + 1 :],
                            cause="projection_queue_dead_letter",
                        )
                    )
                    terminal_abort = True
                failed.append(f"{job.job_id}:{type(exc).__name__}")
                if terminal_abort:
                    failed.append(f"{job.job_id}:queue_dead_letter")
                    break
                self._extend_unattempted_projection_leases(
                    jobs[position + 1 :],
                    lease_seconds=lease_seconds,
                )
            except Exception as exc:
                settled = self.queue_store.retry(job, type(exc).__name__, max_retries=3, retryable=False)
                if settled.status == "dead_letter":
                    self._mark_authoritative_integrity_failure(
                        exc,
                        artifact="projection_queue_dead_letter",
                        identifiers={"job_id": job.job_id, "commit_group_id": group_id},
                    )
                    released.extend(
                        self._release_unattempted_projection_jobs(
                            jobs[position + 1 :],
                            cause="projection_queue_dead_letter",
                        )
                    )
                    terminal_abort = True
                failed.append(f"{job.job_id}:{type(exc).__name__}")
                if terminal_abort:
                    failed.append(f"{job.job_id}:queue_dead_letter")
                    break
                self._extend_unattempted_projection_leases(
                    jobs[position + 1 :],
                    lease_seconds=lease_seconds,
                )
            else:
                self._extend_unattempted_projection_leases(
                    jobs[position + 1 :],
                    lease_seconds=lease_seconds,
                )
        if transaction_ids and not terminal_abort:
            completion = self.verify_commit_group_completion(group_id, transaction_ids)
            failed.extend(completion["failures"])
            completion_proofs.extend(completion["proofs"])
        return {
            "processed": processed,
            "stale": stale,
            "failed": failed,
            "quarantine": [*self.last_quarantined, *quarantine],
            "completion_proofs": completion_proofs,
            "released": released,
        }

    def _verify_projection_completion(
        self,
        group_id: str,
        transaction_ids: tuple[str, ...],
    ) -> list[str]:
        """Prove durable queue and every derived publication before completion."""

        return self.verify_commit_group_completion(group_id, transaction_ids)["failures"]

    def verify_commit_group_completion(
        self,
        group_id: str,
        transaction_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Return immutable publication-bound proofs for one commit group.

        A current transaction is checked against every live derived row.  Once
        its Claim revision advances, the create-only publication receipt and
        revision-scoped projection artifacts become the historical proof;
        current index/vector/view pointers are then validated by the newer
        transaction instead of being incorrectly compared with the old one.
        """

        failures: list[str] = []
        proofs: list[dict[str, Any]] = []
        for transaction_id in transaction_ids:
            job_id = f"outbox_{transaction_id}"
            job = self.queue_store.get(job_id)
            if job is None:
                failures.append(f"{job_id}:missing_job")
                continue
            if job.status != "done":
                failures.append(f"{job_id}:queue_{job.status}")
                continue
            try:
                outbox = self._load_projection_job_outbox(
                    job,
                    expected_transaction_id=transaction_id,
                )
                if (
                    str(outbox.get("transaction_id") or "") != transaction_id
                    or str(outbox.get("commit_group_id") or "") != group_id
                ):
                    raise ProjectionIntegrityError("projection outbox crosses commit group")
                receipt = self._load_bound_receipt(outbox, transaction_id, group_id)
                publication = self.proof_store.load_publication(transaction_id)
                if publication is None:
                    # Compatibility/recovery path for a job ACKed by an older
                    # process or a crash between ACK and completion proof.  It
                    # remains safe only while every projected revision can
                    # still pass the strict current-state verifier.
                    publication = self._ensure_projection_publication(outbox, job)
                self._verify_projection_publication(publication, outbox, receipt, job)
                proof_core = {
                    "schema_version": PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION,
                    "commit_group_id": group_id,
                    "transaction_id": transaction_id,
                    "job_id": job_id,
                    "tenant_id": str(receipt["tenant_id"]),
                    "user_id": str(receipt["user_id"]),
                    "queue_status": job.status,
                    "queue_identity_digest": self._queue_identity_digest(job),
                    "outbox_digest": str(outbox["outbox_digest"]),
                    "receipt_digest": str(receipt["receipt_digest"]),
                    "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
                    "operation_ids": [str(item) for item in outbox["operation_ids"]],
                    "claim_revisions": list(publication["claim_revisions"]),
                    "claims": list(publication["claims"]),
                    "publication_digest": str(publication["publication_digest"]),
                }
                completion = self.proof_store.ensure_completion(
                    {**proof_core, "proof_digest": canonical_digest(proof_core)}
                )
                proofs.append(completion)
            except AuthoritativeProjectionIntegrityError as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_proof",
                    identifiers={"job_id": job_id, "commit_group_id": group_id},
                )
                failures.append(f"{job_id}:ProjectionIntegrityError")
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                ProjectionIntegrityError,
                ProjectionOutboxIntegrityError,
                CommittedStateIntegrityError,
                CurrentHeadIntegrityError,
                ReceiptIntegrityError,
            ) as exc:
                failures.append(f"{job_id}:{type(exc).__name__}")
        return {"failures": failures, "proofs": proofs}

    def _ensure_projection_publication(
        self,
        outbox: dict[str, Any],
        job: QueueJob,
    ) -> dict[str, Any]:
        transaction_id = str(outbox.get("transaction_id") or "")
        group_id = str(outbox.get("commit_group_id") or "")
        receipt = self._load_bound_receipt(outbox, transaction_id, group_id)
        existing = self.proof_store.load_publication(transaction_id)
        if existing is not None:
            self._verify_projection_publication(existing, outbox, receipt, job)
            return existing
        claim_proofs: list[dict[str, Any]] = []
        for item in self._claim_revisions(outbox):
            claim_uri = str(item["uri"])
            source_revision = int(item["revision"])
            head, _current_receipt, _current_snapshot = load_current_head(
                self.projector.root,
                claim_uri,
                canonical_kind="claim",
            )
            current_revision = int(head.get("current_revision", 0))
            if current_revision == source_revision:
                claim_proofs.append(self._verify_claim_projection(claim_uri, source_revision))
            elif current_revision > source_revision:
                claim_proofs.append(
                    self._materialize_historical_claim_projection(
                        receipt,
                        claim_uri,
                        source_revision,
                    )
                )
            else:
                raise ProjectionIntegrityError("projection outbox refers to a future Claim revision")
        publication_core = {
            "schema_version": PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "commit_group_id": group_id,
            "transaction_id": transaction_id,
            "job_id": job.job_id,
            "tenant_id": str(receipt["tenant_id"]),
            "user_id": str(receipt["user_id"]),
            "queue_identity_digest": self._queue_identity_digest(job),
            "outbox_digest": str(outbox["outbox_digest"]),
            "receipt_digest": str(receipt["receipt_digest"]),
            "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
            "operation_ids": [str(item) for item in outbox["operation_ids"]],
            "claim_revisions": [
                {"uri": item["claim_uri"], "revision": item["source_revision"]} for item in claim_proofs
            ],
            "claims": claim_proofs,
        }
        return self.proof_store.ensure_publication(
            {
                **publication_core,
                "publication_digest": canonical_digest(publication_core),
            }
        )

    def _verify_projection_publication(
        self,
        publication: dict[str, Any],
        outbox: dict[str, Any],
        receipt: dict[str, Any],
        job: QueueJob,
    ) -> None:
        self._verify_projection_publication_boundary(publication, outbox, receipt, job)
        claim_proofs = publication.get("claims")
        assert isinstance(claim_proofs, list)
        by_identity = {
            (str(item.get("claim_uri") or ""), int(item.get("source_revision", 0))): item
            for item in claim_proofs
            if isinstance(item, dict)
        }
        expected_identities = {(str(item["uri"]), int(item["revision"])) for item in self._claim_revisions(outbox)}
        for claim_uri, source_revision in sorted(expected_identities):
            claim_proof = by_identity[(claim_uri, source_revision)]
            head, _current_receipt, _current_snapshot = load_current_head(
                self.projector.root,
                claim_uri,
                canonical_kind="claim",
            )
            current_revision = int(head.get("current_revision", 0))
            if current_revision == source_revision:
                # Rebuild may legitimately publish a new attempt for the same
                # committed Source effect.  Verify that mutable current attempt
                # in full, then independently verify the original immutable
                # publication instead of requiring their attempt ids to match.
                self._verify_claim_projection(claim_uri, source_revision)
                self._verify_historical_claim_projection(claim_proof, receipt)
            elif current_revision > source_revision:
                self._verify_historical_claim_projection(claim_proof, receipt)
            else:
                raise ProjectionIntegrityError("projection publication refers to a future Claim revision")

    def _verify_projection_publication_boundary(
        self,
        publication: dict[str, Any],
        outbox: dict[str, Any],
        receipt: dict[str, Any],
        job: QueueJob,
    ) -> None:
        transaction_id = str(outbox["transaction_id"])
        group_id = str(outbox["commit_group_id"])
        expected_boundary = {
            "commit_group_id": group_id,
            "transaction_id": transaction_id,
            "job_id": job.job_id,
            "tenant_id": str(receipt["tenant_id"]),
            "user_id": str(receipt["user_id"]),
            "queue_identity_digest": self._queue_identity_digest(job),
            "outbox_digest": str(outbox["outbox_digest"]),
            "receipt_digest": str(receipt["receipt_digest"]),
            "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
            "operation_ids": [str(item) for item in outbox["operation_ids"]],
            "claim_revisions": self._claim_revisions(outbox),
        }
        actual_boundary = {
            **{key: publication.get(key) for key in expected_boundary if key != "claim_revisions"},
            "claim_revisions": [
                {"uri": str(item.get("uri") or ""), "revision": int(item["revision"])}
                for item in publication.get("claim_revisions", [])
                if isinstance(item, dict) and item.get("revision") is not None
            ],
        }
        if actual_boundary != expected_boundary:
            raise AuthoritativeProjectionIntegrityError(
                "projection publication receipt crosses its transaction boundary"
            )
        claim_proofs = publication.get("claims")
        if not isinstance(claim_proofs, list):
            raise AuthoritativeProjectionIntegrityError("projection publication receipt has no Claim proofs")
        by_identity = {
            (str(item.get("claim_uri") or ""), int(item.get("source_revision", 0))): item
            for item in claim_proofs
            if isinstance(item, dict)
        }
        expected_identities = {(str(item["uri"]), int(item["revision"])) for item in self._claim_revisions(outbox)}
        if set(by_identity) != expected_identities:
            raise AuthoritativeProjectionIntegrityError("projection publication Claim proof set differs from outbox")

    def _load_bound_receipt(
        self,
        outbox: dict[str, Any],
        transaction_id: str,
        group_id: str,
    ) -> dict[str, Any]:
        raw_relative = str(outbox.get("receipt_path") or "")
        try:
            idempotency_key = require_safe_path_segment(
                outbox.get("idempotency_key"),
                "projection outbox idempotency_key",
            )
        except (TypeError, ValueError) as exc:
            raise AuthoritativeProjectionIntegrityError(
                "projection outbox does not identify its unique immutable receipt"
            ) from exc
        normalized_relative = Path(raw_relative)
        expected_relative = Path("system") / "transactions" / f"{idempotency_key}.json"
        candidate = self.projector.root / normalized_relative
        if (
            normalized_relative.is_absolute()
            or normalized_relative.as_posix() != expected_relative.as_posix()
            or candidate.is_symlink()
        ):
            raise AuthoritativeProjectionIntegrityError(
                "projection outbox does not reference its unique immutable receipt"
            )
        try:
            receipt_path = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise AuthoritativeProjectionIntegrityError(
                "projection outbox immutable receipt is missing or unreadable"
            ) from exc
        root = self.projector.root.resolve()
        if receipt_path == root or root not in receipt_path.parents:
            raise AuthoritativeProjectionIntegrityError("projection receipt path escapes tenant root")
        try:
            receipt = load_transaction_receipt(receipt_path)
        except (OSError, ReceiptIntegrityError) as exc:
            raise AuthoritativeProjectionIntegrityError("projection immutable receipt is corrupt") from exc
        if (
            receipt.get("receipt_digest") != outbox.get("receipt_digest")
            or receipt.get("transaction_id") != transaction_id
            or receipt.get("commit_group_id") != group_id
            or receipt.get("prepared_intent_digest") != prepared_intent_digest(outbox)
        ):
            raise AuthoritativeProjectionIntegrityError("projection receipt does not bind its outbox")
        return receipt

    @staticmethod
    def _queue_identity_digest(job: QueueJob) -> str:
        return canonical_digest(
            {
                "job_id": job.job_id,
                "queue_name": job.queue_name,
                "action": job.action,
                "target_uri": job.target_uri,
                "payload": job.payload,
            }
        )

    @staticmethod
    def _claim_revisions(outbox: dict[str, Any]) -> list[dict[str, Any]]:
        revisions: list[dict[str, Any]] = []
        for item in outbox.get("claim_revisions", []) or []:
            if not isinstance(item, dict) or not item.get("uri") or item.get("revision") is None:
                raise ProjectionIntegrityError("projection claim revision is invalid")
            revisions.append({"uri": str(item["uri"]), "revision": int(item["revision"])})
        return sorted(revisions, key=lambda item: (item["uri"], item["revision"]))

    def _verify_claim_projection(self, claim_uri: str, source_revision: int) -> dict[str, Any]:
        if not claim_uri:
            raise ProjectionIntegrityError("projection claim URI is missing")
        committed = read_committed_canonical(
            self.projector.source_store,
            claim_uri,
            self.projector.relation_store,
        )
        metadata = dict(committed.object.metadata or {})
        if committed.from_before_image or int(metadata.get("revision", 0)) != source_revision:
            raise ProjectionIntegrityError("projection source revision is not current committed state")
        record = self.projector.record_store.load_current(
            claim_uri,
            source_revision=source_revision,
        )
        if record is None or not record.usable:
            raise ProjectionIntegrityError("projection current record is missing or incomplete")
        try:
            materialized_current = materialized_current_revision_payload(metadata)
        except CanonicalMemoryInvariantError as exc:
            raise ProjectionIntegrityError("projection source domain state is invalid") from exc
        domain_identity = self.projector._projection_domain_identity(
            committed,
            materialized_current,
        )
        expected_effect = self.projector._input_effect_hash(committed, source_revision)
        if record.input_effect_hash != expected_effect or record.projection_revision != source_revision:
            raise ProjectionIntegrityError("projection input effect does not match committed Source")
        layer_values = {
            "L0": self.projector.source_store.read_content(record.l0_uri),
            "L1": self.projector.source_store.read_content(record.l1_uri),
            "L2": self.projector.source_store.read_content(record.l2_uri),
        }
        if record.projected_content_digest != canonical_digest(layer_values):
            raise ProjectionIntegrityError("projection layer content digest does not match record")
        relation_payload = json.loads(self.projector.source_store.read_content(record.relations_uri))
        if not isinstance(relation_payload, dict) or record.projected_relation_digest != canonical_digest(
            relation_payload.get("relations", [])
        ):
            raise ProjectionIntegrityError("projection relation artifact does not match record")
        self._assert_projection_identity(
            relation_payload,
            record,
            label="relation",
            domain_identity=domain_identity,
        )
        manifest = json.loads(self.projector.source_store.read_content(record.manifest_uri))
        self._assert_projection_identity(
            manifest,
            record,
            label="manifest",
            domain_identity=domain_identity,
        )
        index_metadata = self.projector.index_store.get_index_metadata(claim_uri)
        if index_metadata is None:
            raise ProjectionIntegrityError("projection index row is missing")
        self._assert_projection_identity(
            index_metadata,
            record,
            label="index",
            domain_identity=domain_identity,
        )
        if index_metadata.get("index_content_digest") != canonical_digest(
            "\n".join((layer_values["L0"], layer_values["L1"], layer_values["L2"]))
        ):
            raise ProjectionIntegrityError("projection index content digest does not match layers")
        vector_metadata: dict[str, Any] | None = None
        if self.projector.vector_store is not None:
            vector_metadata = self.projector.vector_store.get_vector_metadata(claim_uri)
            if vector_metadata is None:
                raise ProjectionIntegrityError("projection vector row is missing")
            self._assert_projection_identity(
                vector_metadata,
                record,
                label="vector",
                domain_identity=domain_identity,
            )
        elif record.vector_status != ProjectionStepStatus.SKIPPED.value:
            raise ProjectionIntegrityError("projection vector status cannot be completed without a store")
        scope_views = self._matching_current_views("scope", record, domain_identity)
        taxonomy_views = self._matching_current_views("taxonomy", record, domain_identity)
        if not scope_views or not taxonomy_views:
            raise ProjectionIntegrityError("projection scope or taxonomy publication is missing")
        claim_core = {
            "claim_uri": record.claim_uri,
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
            "record_digest": str(record.to_dict()["record_digest"]),
            "publication_record_digest": projection_publication_record_digest(record),
            "layer_uris": {
                "L0": record.l0_uri,
                "L1": record.l1_uri,
                "L2": record.l2_uri,
                "manifest": record.manifest_uri,
                "relations": record.relations_uri,
            },
            "layer_digests": {name: canonical_digest(value) for name, value in layer_values.items()},
            "relation_artifact_digest": canonical_digest(relation_payload),
            "manifest_digest": canonical_digest(manifest),
            "index_metadata_digest": canonical_digest(index_metadata),
            "vector_metadata_digest": (canonical_digest(vector_metadata) if vector_metadata is not None else ""),
            "scope_view_digests": sorted(canonical_digest(item) for item in scope_views),
            "taxonomy_view_digests": sorted(canonical_digest(item) for item in taxonomy_views),
            "domain_identity": domain_identity,
        }
        return {**claim_core, "claim_proof_digest": canonical_digest(claim_core)}

    def _verify_historical_claim_projection(
        self,
        claim_proof: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        """Validate a retired projection without consulting mutable current rows."""

        claim_uri = str(claim_proof.get("claim_uri") or "")
        source_revision = int(claim_proof.get("source_revision", 0))
        try:
            snapshot = receipt_snapshot(receipt, claim_uri)
            obj = ContextObject.from_dict(dict(snapshot["object"]))
        except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
            raise ProjectionIntegrityError("historical projection has no matching immutable Source effect") from exc
        metadata = dict(obj.metadata or {})
        if (
            str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "") != "claim"
            or int(metadata.get("revision", 0)) != source_revision
            or int(snapshot.get("after_revision", 0)) != source_revision
        ):
            raise ProjectionIntegrityError("historical projection Source revision is inconsistent")
        try:
            materialized_current = materialized_current_revision_payload(metadata)
        except CanonicalMemoryInvariantError as exc:
            raise ProjectionIntegrityError("historical projection Source domain state is invalid") from exc
        expected_domain_identity = {
            "claim_uri": claim_uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "owner_user_id": str(obj.owner_user_id or ""),
            "canonical_kind": "claim",
            "claim_state": str(materialized_current.get("state") or ""),
            "current_transaction_id": str(receipt["transaction_id"]),
            "current_receipt_digest": str(receipt["receipt_digest"]),
            "current_claim_revision": int(materialized_current["revision"]),
        }
        if claim_proof.get("domain_identity") != expected_domain_identity or expected_domain_identity[
            "claim_state"
        ] != str(metadata.get("state") or ""):
            raise ProjectionIntegrityError("historical projection domain identity differs from receipt")
        historical_committed = CommittedCanonicalRead(obj, receipt=receipt)
        expected_effect_hash = self.projector._input_effect_hash(
            historical_committed,
            source_revision,
        )
        if claim_proof.get("input_effect_hash") != expected_effect_hash:
            raise ProjectionIntegrityError("historical projection input effect differs from receipt")
        attempt_id = str(claim_proof.get("projection_attempt_id") or "")
        expected_base = f"{claim_uri}/projections/rev-{source_revision}/attempt-{attempt_id}"
        expected_uris = {
            "L0": f"{expected_base}/l0.md",
            "L1": f"{expected_base}/l1.md",
            "L2": f"{expected_base}/l2.json",
            "manifest": f"{expected_base}/manifest.json",
            "relations": f"{expected_base}/relations.json",
        }
        if not attempt_id or claim_proof.get("layer_uris") != expected_uris:
            raise ProjectionIntegrityError("historical projection artifact URIs are inconsistent")
        layer_values = {
            "L0": self.projector.source_store.read_content(expected_uris["L0"]),
            "L1": self.projector.source_store.read_content(expected_uris["L1"]),
            "L2": self.projector.source_store.read_content(expected_uris["L2"]),
        }
        layer_digests = {name: canonical_digest(value) for name, value in layer_values.items()}
        if claim_proof.get("layer_digests") != layer_digests or claim_proof.get(
            "projected_content_digest"
        ) != canonical_digest(layer_values):
            raise ProjectionIntegrityError("historical projection layer content is corrupt")
        try:
            relation_payload = json.loads(self.projector.source_store.read_content(expected_uris["relations"]))
            manifest = json.loads(self.projector.source_store.read_content(expected_uris["manifest"]))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProjectionIntegrityError("historical projection artifact is malformed") from exc
        if not isinstance(relation_payload, dict) or not isinstance(manifest, dict):
            raise ProjectionIntegrityError("historical projection artifact is not an object")
        record_payload = {key: manifest.get(key) for key in ProjectionRecord.__dataclass_fields__}
        record_payload["record_digest"] = manifest.get("record_digest")
        try:
            record = ProjectionRecord.from_dict(record_payload)
        except (KeyError, TypeError, ValueError, ProjectionIntegrityError) as exc:
            raise ProjectionIntegrityError("historical projection manifest has no valid attempt snapshot") from exc
        if record.status not in {ProjectionStatus.COMPLETED.value, ProjectionStatus.STALE.value}:
            raise ProjectionIntegrityError("historical projection attempt has an invalid terminal state")
        if projection_publication_record_digest(record) != claim_proof.get("publication_record_digest"):
            raise ProjectionIntegrityError("historical projection attempt differs from publication receipt")
        expected_record_fields = {
            "claim_uri": record.claim_uri,
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
        }
        if any(claim_proof.get(key) != value for key, value in expected_record_fields.items()):
            raise ProjectionIntegrityError("historical projection record identity is inconsistent")
        if (
            claim_proof.get("relation_artifact_digest") != canonical_digest(relation_payload)
            or claim_proof.get("manifest_digest") != canonical_digest(manifest)
            or record.projected_relation_digest != canonical_digest(relation_payload.get("relations", []))
        ):
            raise ProjectionIntegrityError("historical projection artifact digest is corrupt")
        self._assert_projection_identity(
            relation_payload,
            record,
            label="historical relation",
            domain_identity=expected_domain_identity,
        )
        self._assert_projection_identity(
            manifest,
            record,
            label="historical manifest",
            domain_identity=expected_domain_identity,
        )
        if claim_proof.get("historical_only") is True:
            expected_attestations = {
                component: self._historical_component_attestation(
                    component,
                    claim_uri=claim_uri,
                    source_revision=source_revision,
                    transaction_id=str(receipt["transaction_id"]),
                    receipt_digest=str(receipt["receipt_digest"]),
                )
                for component in ("index", "vector", "scope", "taxonomy")
            }
            if (
                claim_proof.get("index_metadata_digest") != expected_attestations["index"]
                or claim_proof.get("vector_metadata_digest") != expected_attestations["vector"]
                or claim_proof.get("scope_view_digests") != [expected_attestations["scope"]]
                or claim_proof.get("taxonomy_view_digests") != [expected_attestations["taxonomy"]]
            ):
                raise ProjectionIntegrityError("historical projection component attestation is inconsistent")
        self._restore_historical_projection_record(record)

    def _materialize_historical_claim_projection(
        self,
        receipt: dict[str, Any],
        claim_uri: str,
        source_revision: int,
    ) -> dict[str, Any]:
        """Project an immutable receipt snapshot without replacing current rows."""

        try:
            snapshot = receipt_snapshot(receipt, claim_uri)
            obj = ContextObject.from_dict(dict(snapshot["object"]))
        except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
            raise ProjectionIntegrityError("historical projection has no matching immutable Source effect") from exc
        metadata = dict(obj.metadata or {})
        if (
            str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "") != "claim"
            or int(metadata.get("revision", 0)) != source_revision
            or int(snapshot.get("after_revision", 0)) != source_revision
        ):
            raise ProjectionIntegrityError("historical projection Source revision is inconsistent")
        try:
            materialized = materialized_current_revision_payload(metadata)
            revision = self.projector._revision_payload(
                metadata,
                int(materialized["revision"]),
            )
        except (CanonicalMemoryInvariantError, KeyError, TypeError, ValueError) as exc:
            raise ProjectionIntegrityError("historical projection Source domain state is invalid") from exc
        domain_identity = {
            "claim_uri": claim_uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "owner_user_id": str(obj.owner_user_id or ""),
            "canonical_kind": "claim",
            "claim_state": str(materialized.get("state") or ""),
            "current_transaction_id": str(receipt["transaction_id"]),
            "current_receipt_digest": str(receipt["receipt_digest"]),
            "current_claim_revision": int(materialized["revision"]),
        }
        if not domain_identity["claim_state"] or domain_identity["claim_state"] != str(metadata.get("state") or ""):
            raise ProjectionIntegrityError("historical projection Claim state is inconsistent")
        committed = CommittedCanonicalRead(obj, receipt=receipt)
        input_effect_hash = self.projector._input_effect_hash(committed, source_revision)
        attempt_id = canonical_digest(
            {
                "schema_version": "historical_projection_attempt_v1",
                "transaction_id": str(receipt["transaction_id"]),
                "receipt_digest": str(receipt["receipt_digest"]),
                "claim_uri": claim_uri,
                "source_revision": source_revision,
            }
        )[:32]
        slot_uri = claim_uri.rsplit("/claims/", 1)[0]
        base = f"{claim_uri}/projections/rev-{source_revision}/attempt-{attempt_id}"
        record = self.projector.record_store.start(
            claim_uri=claim_uri,
            slot_uri=slot_uri,
            source_revision=source_revision,
            projection_revision=source_revision,
            projection_attempt_id=attempt_id,
            input_effect_hash=input_effect_hash,
            l0_uri=f"{base}/l0.md",
            l1_uri=f"{base}/l1.md",
            l2_uri=f"{base}/l2.json",
            relations_uri=f"{base}/relations.json",
            manifest_uri=f"{base}/manifest.json",
            current_claim_revision=int(materialized["revision"]),
        )
        record = self.projector.record_store.update(
            record,
            index_status=ProjectionStepStatus.SKIPPED.value,
            vector_status=ProjectionStepStatus.SKIPPED.value,
            relation_status=ProjectionStepStatus.RUNNING.value,
            scope_status=ProjectionStepStatus.SKIPPED.value,
            taxonomy_status=ProjectionStepStatus.SKIPPED.value,
            status=ProjectionStatus.RUNNING.value,
            failure_reason="",
            retryable=True,
            current=False,
        )
        l0, l1, l2 = self.projector._layers(
            obj,
            metadata,
            revision,
            source_revision,
        )
        relations = [item.to_dict() for item in committed_relations(committed)]
        record = self.projector.record_store.update(
            record,
            projected_content_digest=canonical_digest({"L0": l0, "L1": l1, "L2": l2}),
            projected_relation_digest=canonical_digest(relations),
        )
        self.projector.source_store.write_content(record.l0_uri, l0)
        self.projector.source_store.write_content(record.l1_uri, l1)
        self.projector.source_store.write_content(record.l2_uri, l2)
        relation_payload = {
            **domain_identity,
            "claim_uri": claim_uri,
            "slot_uri": slot_uri,
            "source_revision": source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
            "relations": relations,
        }
        self.projector.source_store.write_content(
            record.relations_uri,
            json.dumps(relation_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        record = self.projector.record_store.update(
            record,
            relation_status=ProjectionStepStatus.COMPLETED.value,
        )
        record = self.projector.record_store.stale(
            record,
            "canonical revision advanced before projection publication",
        )
        manifest = self.projector._manifest(
            record,
            metadata,
            record.relations_uri,
            domain_identity=domain_identity,
        )
        self.projector.source_store.write_content(
            record.manifest_uri,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )
        layer_values = {"L0": l0, "L1": l1, "L2": l2}
        attestations = {
            component: self._historical_component_attestation(
                component,
                claim_uri=claim_uri,
                source_revision=source_revision,
                transaction_id=str(receipt["transaction_id"]),
                receipt_digest=str(receipt["receipt_digest"]),
            )
            for component in ("index", "vector", "scope", "taxonomy")
        }
        claim_core = {
            "claim_uri": record.claim_uri,
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
            "record_digest": str(record.to_dict()["record_digest"]),
            "publication_record_digest": projection_publication_record_digest(record),
            "layer_uris": {
                "L0": record.l0_uri,
                "L1": record.l1_uri,
                "L2": record.l2_uri,
                "manifest": record.manifest_uri,
                "relations": record.relations_uri,
            },
            "layer_digests": {name: canonical_digest(value) for name, value in layer_values.items()},
            "relation_artifact_digest": canonical_digest(relation_payload),
            "manifest_digest": canonical_digest(manifest),
            "index_metadata_digest": attestations["index"],
            "vector_metadata_digest": attestations["vector"],
            "scope_view_digests": [attestations["scope"]],
            "taxonomy_view_digests": [attestations["taxonomy"]],
            "domain_identity": domain_identity,
            "historical_only": True,
        }
        proof = {**claim_core, "claim_proof_digest": canonical_digest(claim_core)}
        self._verify_historical_claim_projection(proof, receipt)
        return proof

    @staticmethod
    def _historical_component_attestation(
        component: str,
        *,
        claim_uri: str,
        source_revision: int,
        transaction_id: str,
        receipt_digest: str,
    ) -> str:
        return canonical_digest(
            {
                "schema_version": "historical_projection_component_attestation_v1",
                "component": component,
                "status": "skipped_superseded",
                "claim_uri": claim_uri,
                "source_revision": source_revision,
                "transaction_id": transaction_id,
                "receipt_digest": receipt_digest,
            }
        )

    def _restore_historical_projection_record(self, record: ProjectionRecord) -> None:
        """Restore disposable attempt state from its immutable manifest snapshot."""

        path = self.projector.record_store.attempt_path_for(record)
        try:
            persisted = self.projector.record_store.load(
                record.claim_uri,
                record.source_revision,
                projection_attempt_id=record.projection_attempt_id,
            )
        except ProjectionIntegrityError:
            persisted = None
        if persisted is not None:
            if projection_publication_record_digest(persisted) == (projection_publication_record_digest(record)):
                return
            quarantine_control_file(
                self.projector.root,
                path,
                kind="projection_record",
                error=ValueError("projection attempt differs from immutable publication manifest"),
                identifiers={
                    "record_id": path.stem,
                    "claim_uri": record.claim_uri,
                    "projection_attempt_id": record.projection_attempt_id,
                },
            )
        self.projector.record_store.save(record)

    def _matching_current_views(
        self,
        kind: str,
        record: ProjectionRecord,
        domain_identity: dict[str, Any],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for path in self.projector.root.glob(f"views/{kind}/**/current.json"):
            payload = self.projector._read_json_optional(path)
            if payload is None or str(payload.get("claim_uri") or "") != record.claim_uri:
                continue
            self._assert_projection_identity(
                payload,
                record,
                label=kind,
                domain_identity=domain_identity,
            )
            matches.append(payload)
        return matches

    @staticmethod
    def _assert_projection_identity(
        payload: dict[str, Any],
        record: ProjectionRecord,
        *,
        label: str,
        domain_identity: dict[str, Any],
    ) -> None:
        expected = {
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
            **domain_identity,
        }
        aliases = {
            "source_revision": ("source_revision", "projection_source_revision"),
            "projection_revision": ("projection_revision",),
            "projection_attempt_id": ("projection_attempt_id",),
            "input_effect_hash": ("input_effect_hash", "projection_input_effect_hash"),
            "publish_token": ("publish_token", "projection_publish_token"),
            "projected_content_digest": (
                "projected_content_digest",
                "projection_content_digest",
            ),
            "projected_relation_digest": (
                "projected_relation_digest",
                "projection_relation_digest",
            ),
            "claim_uri": ("claim_uri",),
            "tenant_id": ("tenant_id",),
            "owner_user_id": ("owner_user_id",),
            "canonical_kind": ("canonical_kind",),
            "claim_state": ("claim_state",),
            "current_transaction_id": ("current_transaction_id",),
            "current_receipt_digest": ("current_receipt_digest",),
            "current_claim_revision": ("current_claim_revision",),
        }
        for field_name, expected_value in expected.items():
            actual = next((payload.get(key) for key in aliases[field_name] if key in payload), None)
            if actual != expected_value:
                raise ProjectionIntegrityError(f"projection {label} {field_name} mismatch")

    def _read_outbox(self, path: Path) -> dict[str, Any]:
        try:
            if path.is_symlink():
                raise OutboxIntegrityError("canonical outbox path cannot be a symbolic link")
            return validate_outbox(
                json.loads(path.read_text(encoding="utf-8")),
                allowed_statuses={"committed"},
            )
        except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
            if path.exists():
                quarantine_control_file(
                    self.projector.root,
                    path,
                    kind="outbox",
                    error=exc,
                    identifiers={"transaction_id": path.stem},
                )
            raise ProjectionOutboxIntegrityError("projection job references an invalid committed outbox event") from exc

    def _load_projection_job_outbox(
        self,
        job: QueueJob,
        *,
        expected_transaction_id: str = "",
    ) -> dict[str, Any]:
        """Bind a durable queue identity to exactly one committed outbox."""

        self._assert_projection_job_identity_unchanged(job)
        declared_transaction = str(job.payload.get("transaction_id") or "")
        if expected_transaction_id and declared_transaction != expected_transaction_id:
            raise ProjectionOutboxIntegrityError(
                "projection queue transaction identity does not match completion request"
            )
        if (
            not declared_transaction
            or job.job_id != f"outbox_{declared_transaction}"
            or job.queue_name != "memory_projection"
            or job.action != "project_memory_committed"
        ):
            raise ProjectionOutboxIntegrityError("projection queue job identity is invalid")
        expected_candidate = self.projector.root / "system" / "outbox" / f"{declared_transaction}.json"
        expected_path = expected_candidate.resolve()
        raw_path = job.payload.get("outbox_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ProjectionOutboxIntegrityError("projection queue job has no outbox path")
        try:
            raw_candidate = Path(raw_path)
            if raw_candidate.is_symlink() or expected_candidate.is_symlink():
                raise ProjectionOutboxIntegrityError("projection queue outbox path cannot be a symbolic link")
            actual_path = raw_candidate.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProjectionOutboxIntegrityError("projection queue outbox path is invalid") from exc
        if actual_path != expected_path:
            raise ProjectionOutboxIntegrityError("projection queue job is detached from its tenant outbox path")
        outbox = self._read_outbox(actual_path)
        operation_ids = job.payload.get("operation_ids")
        if (
            outbox.get("transaction_id") != declared_transaction
            or not isinstance(operation_ids, list)
            or operation_ids != outbox.get("operation_ids")
        ):
            raise ProjectionOutboxIntegrityError("projection queue job is detached from its immutable operation set")
        return outbox

    def _project_event(self, outbox: dict[str, Any], job_id: str, stale: list[str]) -> None:
        for item in outbox.get("claim_revisions", []) or []:
            if not isinstance(item, dict) or not item.get("uri") or item.get("revision") is None:
                raise ValueError("projection outbox contains an invalid claim revision")
            result = self.projector.project(str(item["uri"]), int(item["revision"]))
            if result.status == "skipped_stale":
                stale.append(job_id)

    def dispatch_outbox(self) -> list[str]:
        outbox_root = self.projector.root / "system" / "outbox"
        if not outbox_root.exists():
            return []
        validated: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(outbox_root.glob("*.json")):
            try:
                if path.is_symlink():
                    raise OutboxIntegrityError("canonical outbox path cannot be a symbolic link")
                event = validate_outbox(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                quarantine_control_file(
                    self.projector.root,
                    path,
                    kind="outbox",
                    error=exc,
                    identifiers={"transaction_id": path.stem},
                )
                self.last_quarantined.append(path.stem)
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="committed_outbox",
                    identifiers={"transaction_id": path.stem},
                )
                raise ProjectionOutboxIntegrityError(
                    "authoritative outbox scan failed before projection dispatch"
                ) from exc
            validated.append((path, event))

        pending_jobs: list[tuple[str, QueueJob]] = []
        for path, event in validated:
            if event.get("event_type") != OUTBOX_EVENT_TYPE or event.get("status") != "committed":
                continue
            transaction_id = str(event.get("transaction_id", ""))
            if not transaction_id or path.stem != transaction_id:
                failure = ProjectionOutboxIntegrityError(
                    "committed outbox path is detached from its transaction identity"
                )
                self._mark_authoritative_integrity_failure(
                    failure,
                    artifact="committed_outbox",
                    identifiers={"transaction_id": transaction_id or path.stem},
                )
                raise failure
            claim_revisions = event.get("claim_revisions", []) or []
            operations = [item for item in event.get("operations", []) or [] if isinstance(item, dict)]
            target_uri = next(
                (
                    str(payload.get("uri", ""))
                    for item in operations
                    if isinstance((payload := item.get("payload", {}).get("context_object")), dict)
                    and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
                ),
                str(claim_revisions[0].get("uri", "")).rsplit("/claims/", 1)[0] if claim_revisions else transaction_id,
            )
            pending_jobs.append(
                (
                    transaction_id,
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
                    ),
                )
            )

        # Validate every existing queue identity before publishing any new
        # job.  A corrupt member cannot allow later valid work in this scan to
        # reach lease or derived projection writes.
        for transaction_id, expected in pending_jobs:
            existing = self.queue_store.get(expected.job_id)
            if existing is None:
                continue
            if (
                existing.queue_name != expected.queue_name
                or existing.action != expected.action
                or existing.target_uri != expected.target_uri
                or existing.payload != expected.payload
            ):
                queue_conflict = QueueIdempotencyConflictError(
                    "projection queue identity conflicts with its committed outbox"
                )
                self._mark_authoritative_integrity_failure(
                    queue_conflict,
                    artifact="projection_queue",
                    identifiers={"transaction_id": transaction_id},
                )
                raise ProjectionOutboxIntegrityError(str(queue_conflict)) from queue_conflict
            if existing.status in {"dead_letter", "quarantine"}:
                terminal_failure = ProjectionOutboxIntegrityError(
                    f"projection queue is terminal before publication: {existing.status}"
                )
                self._mark_authoritative_integrity_failure(
                    terminal_failure,
                    artifact=f"projection_queue_{existing.status}",
                    identifiers={
                        "transaction_id": transaction_id,
                        "job_id": existing.job_id,
                    },
                )
                raise terminal_failure

        dispatched: list[str] = []
        for transaction_id, expected in pending_jobs:
            try:
                self.queue_store.enqueue(expected)
            except QueueIdempotencyConflictError as exc:
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue",
                    identifiers={"transaction_id": transaction_id},
                )
                raise ProjectionOutboxIntegrityError(
                    "projection queue identity conflicts with its committed outbox"
                ) from exc
            dispatched.append(transaction_id)
        return dispatched
