"""Service Logic responsibilities for canonical projection."""

from __future__ import annotations

import json
import shutil
import uuid
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.catalog import (
    CatalogRecordKind,
)
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.current_head import (
    artifact_root_for,
    iter_current_head_uris,
)
from memoryos.memory.canonical.projection_proof import (
    ProjectionProofStore,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
    ProjectionStatus,
    ProjectionStepStatus,
)
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import (
    materialized_current_revision_payload,
    revision_payload_with_effective_validity,
)
from memoryos.memory.canonical.visibility import (
    committed_relations,
    read_committed_canonical,
)
from memoryos.memory.integration.classification import (
    is_canonical_memory_object,
    is_canonical_memory_uri,
)

from .models import (
    ProjectionResult,
)

if TYPE_CHECKING:
    from .service import CanonicalMemoryProjector


def project(
    self: CanonicalMemoryProjector,
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
        revisions = self._bounded_claim_revisions(metadata)
        # Legacy/offline artifacts deliberately describe the materialized
        # current assertion.  A late historical transaction advances the
        # Source tail without replacing that effective current value.
        # Unified Claim Revision serving rows are built separately below
        # from the requested immutable revision.
        revision = revision_payload_with_effective_validity(
            revisions,
            current_claim_revision,
        )
        l0, l1, l2 = self._sanitized_revision_layers(
            obj,
            metadata,
            revision,
            requested,
        )
        catalog_upsert = getattr(self.index_store, "upsert_catalog", None)
        unified_catalog = callable(catalog_upsert)
        requested_revision = revision_payload_with_effective_validity(
            revisions,
            requested,
        )
        serving_l0, serving_l1, serving_l2 = (
            self._sanitized_revision_layers(
                obj,
                metadata,
                requested_revision,
                requested,
            )
            if unified_catalog
            else (l0, l1, l2)
        )
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
            vector_embedding = self.embedding_provider.embed("\n".join((serving_l0, serving_l1)))
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
            catalog_record = (
                self._claim_revision_catalog_record(
                    obj,
                    metadata,
                    record,
                    requested_revision,
                    proof_metadata=dict(projection_obj.metadata or {}),
                    l0_text=serving_l0,
                    l1_text=serving_l1,
                    l2_text=serving_l2,
                )
                if unified_catalog
                else None
            )
            if self.vector_store is not None:
                assert vector_embedding is not None
                try:
                    if catalog_record is not None:
                        self._publish_catalog_vector(catalog_record, vector_embedding, record)
                    else:
                        self._publish_vector(projection_obj, vector_embedding, record)
                except Exception:
                    record = self.record_store.update(record, vector_status=ProjectionStepStatus.FAILED.value)
                    raise
                record = self.record_store.update(record, vector_status=ProjectionStepStatus.COMPLETED.value)

            record = self.record_store.update(record, index_status=ProjectionStepStatus.RUNNING.value)
            try:
                index_content = "\n".join((l0, l1, l2))
                if catalog_record is not None:
                    assert callable(catalog_upsert)
                    catalog_upsert(catalog_record)
                    self._reconcile_claim_catalog_projections(
                        obj,
                        metadata,
                        published_revision=requested,
                    )
                else:
                    self.index_store.upsert_index(projection_obj, content=index_content)
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


def rebuild(self: CanonicalMemoryProjector, *, clear_views: bool = True) -> dict[str, int]:
    projected = 0
    skipped = 0
    # Rebuild is also called by recovery/admin code outside ContextDB.
    # Complete every non-mutating authoritative/derived-input check before
    # deleting a view, retiring a pointer, or replacing an index/vector
    # row.  Outbox/queue boundary validation remains the worker/caller's
    # responsibility because the projector deliberately has no queue.
    proof_store = ProjectionProofStore(self.root)
    proof_store.validate_all()
    historical_proofs = self._verified_rebuild_claim_proofs(proof_store)
    artifact_root = artifact_root_for(self.source_store)
    claim_uris = tuple(iter_current_head_uris(artifact_root, kinds=("claim",)) if artifact_root is not None else ())
    committed_claims = set(claim_uris)

    source_tenant = str(getattr(self.source_store, "tenant_id", "") or "")

    def uncommitted_canonical_row(
        row_id: str,
        metadata: dict[str, Any] | None,
        *,
        allow_source_read: bool,
    ) -> bool:
        row_metadata = dict(metadata or {})
        metadata_tenant = str(row_metadata.get("tenant_id") or "")
        if source_tenant and metadata_tenant and metadata_tenant != source_tenant:
            return False
        logical_uris = tuple(
            dict.fromkeys(
                value
                for value in (
                    str(row_metadata.get("public_uri") or ""),
                    str(row_metadata.get("uri") or ""),
                    str(row_metadata.get("source_uri") or ""),
                    str(row_metadata.get("claim_uri") or ""),
                    str(row_id),
                )
                if value.startswith("memoryos://")
            )
        )
        if any(uri in committed_claims for uri in logical_uris):
            return False
        if (
            any(is_canonical_memory_uri(uri) for uri in logical_uris)
            or str(row_metadata.get("canonical_kind") or "") in {"slot", "claim", "pending_proposal"}
            or str(row_metadata.get("record_kind") or "")
            in {CatalogRecordKind.CLAIM_REVISION.value, CatalogRecordKind.CURRENT_SLOT.value}
            or str(row_metadata.get("schema_version") or "").startswith("canonical_")
        ):
            return True
        # Backend row IDs (for example ``memoryos-vector://`` hashes) are
        # never Source URIs.  Only the SQLite URI path may perform this
        # offline repair read, and only after proving a logical URI.
        if not allow_source_read or len(logical_uris) != 1:
            return False
        try:
            return is_canonical_memory_object(self.source_store.read_object(logical_uris[0]))
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
            allow_source_read=True,
        )
    )
    vector_to_remove = (
        tuple(
            vector_uri
            for vector_uri in tuple(self.vector_store.vector_uris())
            if uncommitted_canonical_row(
                vector_uri,
                self.vector_store.get_vector_metadata(vector_uri),
                allow_source_read=vector_uri.startswith("memoryos://"),
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
    historical_restored = 0
    for claim_uri in claim_uris:
        result = self.project(claim_uri, force=True)
        if result.status == "projected":
            projected += 1
        else:
            skipped += 1
        historical_restored += self._rebuild_claim_revision_catalog(
            claim_uri,
            historical_proofs,
        )
    return {
        "projected": projected,
        "skipped": skipped,
        "retired": retired,
        "historical_restored": historical_restored,
    }


def _notify(self: CanonicalMemoryProjector, stage: str, claim_uri: str, revision: int) -> None:
    if self.test_hook is not None:
        self.test_hook(stage, claim_uri, revision)


def _result(self: CanonicalMemoryProjector, record: ProjectionRecord, status: str) -> ProjectionResult:
    self._emit(record)
    return ProjectionResult(
        record.claim_uri,
        record.source_revision,
        status,
        str(self.record_store.attempt_path_for(record)),
        record.projection_attempt_id,
        record.input_effect_hash,
    )


def _emit(self: CanonicalMemoryProjector, record: ProjectionRecord) -> None:
    if self.status_callback is not None:
        self.status_callback(record)
