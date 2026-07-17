"""Catalog responsibilities for canonical projection."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    validate_tree_paths,
)
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.vector import vector_row_id
from memoryos.core.integrity import canonical_digest, canonical_json
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
    revision_payload_with_effective_validity,
)
from memoryos.memory.canonical.visibility import (
    read_committed_canonical,
)

from .models import (
    _PROJECTION_ATTEMPT_IDENTITY_FIELDS,
    _PROJECTION_DOMAIN_IDENTITY_FIELDS,
)

if TYPE_CHECKING:
    from .service import CanonicalMemoryProjector


def _rebuild_claim_revision_catalog(
    self: CanonicalMemoryProjector,
    claim_uri: str,
    proofs: dict[tuple[str, int], tuple[dict[str, Any], ProjectionRecord]],
) -> int:
    upsert_catalog = getattr(self.index_store, "upsert_catalog", None)
    if not callable(upsert_catalog):
        return 0
    committed = read_committed_canonical(self.source_store, claim_uri, self.relation_store)
    obj = committed.object
    metadata = dict(obj.metadata or {})
    revisions = self._bounded_claim_revisions(metadata)
    tail_revision = int(metadata.get("revision", 0) or 0)
    available_for_claim = {revision for uri, revision in proofs if uri == claim_uri}
    restored = 0
    for raw_revision in revisions:
        revision_number = int(raw_revision.get("revision", 0) or 0)
        if revision_number == tail_revision:
            continue
        proof_entry = proofs.get((claim_uri, revision_number))
        if proof_entry is None:
            if available_for_claim:
                raise ProjectionIntegrityError("projection rebuild is missing an immutable historical proof")
            continue
        claim_proof, record = proof_entry
        snapshot_revision = next(
            (item for item in revisions if int(item.get("revision", 0) or 0) == revision_number),
            None,
        )
        if snapshot_revision is None:
            raise ProjectionIntegrityError("projection rebuild Source revision disappeared")
        effective = revision_payload_with_effective_validity(revisions, revision_number)
        l0_text, l1_text, l2_text = self._sanitized_revision_layers(
            obj,
            metadata,
            effective,
            revision_number,
        )
        domain = dict(claim_proof.get("domain_identity", {}) or {})
        proof_metadata = {
            **domain,
            # ACL ownership is current Source state; transaction/head
            # fields below still name the original immutable publication.
            "claim_uri": obj.uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "owner_user_id": str(obj.owner_user_id or ""),
            "canonical_kind": "claim",
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "projection_input_effect_hash": record.input_effect_hash,
            "projection_publish_token": record.publish_token,
            "projection_content_digest": record.projected_content_digest,
            "projection_relation_digest": record.projected_relation_digest,
            "projection_manifest_uri": record.manifest_uri,
            "projection_record_path": str(self.record_store.attempt_path_for(record)),
        }
        catalog = self._claim_revision_catalog_record(
            obj,
            metadata,
            record,
            effective,
            proof_metadata=proof_metadata,
            l0_text=l0_text,
            l1_text=l1_text,
            l2_text=l2_text,
        )
        upsert_catalog(catalog)
        self._refresh_claim_vector(catalog)
        restored += 1
    return restored


def _claim_revision_catalog_record(
    self: CanonicalMemoryProjector,
    obj: ContextObject,
    metadata: dict[str, Any],
    record: ProjectionRecord,
    revision: dict[str, Any],
    *,
    proof_metadata: dict[str, Any],
    l0_text: str,
    l1_text: str,
    l2_text: str,
) -> CatalogRecord:
    """Build one requested-revision serving row without changing legacy artifacts."""

    source_revision = int(revision.get("revision", 0) or 0)
    if source_revision != record.source_revision:
        raise ProjectionIntegrityError("Claim Catalog revision does not match projection proof")
    # These fields are not searchable business state.  They are the exact
    # immutable domain/attempt identity consumed by the publication
    # verifier.  Keep them in metadata_json as well as typed Catalog
    # columns: reconstructing them from mutable columns would weaken the
    # receipt -> publication -> serving-row binding.
    required_proof_fields = (
        *_PROJECTION_DOMAIN_IDENTITY_FIELDS,
        *_PROJECTION_ATTEMPT_IDENTITY_FIELDS,
    )
    missing_proof_fields = [key for key in required_proof_fields if key not in proof_metadata]
    if missing_proof_fields:
        raise ProjectionIntegrityError(
            "Claim Catalog proof metadata is incomplete: " + ", ".join(sorted(missing_proof_fields))
        )
    proof_fields = {key: proof_metadata[key] for key in required_proof_fields}
    expected_domain_identity: dict[str, object] = {
        "claim_uri": obj.uri,
        "tenant_id": str(obj.tenant_id or "default"),
        "owner_user_id": str(obj.owner_user_id or ""),
        "canonical_kind": "claim",
    }
    for field, expected in expected_domain_identity.items():
        if proof_fields.get(field) != expected:
            raise ProjectionIntegrityError(f"Claim Catalog proof {field} differs from Source identity")
    expected_attempt_identity: dict[str, object] = {
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "projection_input_effect_hash": record.input_effect_hash,
        "projection_publish_token": record.publish_token,
        "projection_content_digest": record.projected_content_digest,
        "projection_relation_digest": record.projected_relation_digest,
        "projection_manifest_uri": record.manifest_uri,
    }
    for field, expected in expected_attempt_identity.items():
        if proof_fields.get(field) != expected:
            raise ProjectionIntegrityError(f"Claim Catalog proof {field} differs from projection attempt")
    values = dict(revision.get("value_fields", {}) or {})
    raw_value = values.get("canonical_value", values.get("value", metadata.get("canonical_value", obj.title)))
    title = raw_value if isinstance(raw_value, str) else canonical_json(raw_value)
    qualifiers = dict(revision.get("qualifiers", {}) or {})
    display_fields = dict(qualifiers.get("display_fields", {}) or {})
    display_evidence = dict(qualifiers.get("display_field_evidence_refs", {}) or {})
    valid_from = str(revision.get("valid_from") or "")
    valid_to = str(revision.get("valid_to") or "")
    created_at = str(revision.get("created_at") or obj.created_at)
    transaction_time = str(revision.get("transaction_time") or created_at or obj.updated_at)
    event_time = str(revision.get("event_time") or revision.get("occurred_at") or valid_from or created_at)
    tree_paths = self._canonical_tree_paths(metadata)
    catalog_l2_uri = f"{record.l2_uri.rsplit('/', 1)[0]}/catalog-l2.json"
    serving_digest = canonical_digest({"L0": l0_text, "L1": l1_text, "L2": l2_text})
    self.source_store.write_content(catalog_l2_uri, l2_text)
    projected = ContextObject.from_dict(obj.to_dict())
    projected.title = title
    projected.created_at = created_at
    # ``updated_at`` is the derived row refresh time/source head time;
    # transaction_time below remains revision-specific.
    projected.updated_at = str(obj.updated_at or transaction_time or created_at)
    projected.layers = ContextLayers(
        l0_uri=record.l0_uri,
        l1_uri=record.l1_uri,
        l2_uri=catalog_l2_uri,
    )
    safe = self.sanitizer.sanitize(
        title=title,
        l0_text=l0_text,
        l1_text=l1_text,
        metadata={
            **metadata,
            **proof_fields,
            # The Catalog row is one immutable Claim revision, not a
            # second copy of the mutable Claim aggregate.  Keep the
            # current Source scope/visibility/authority above, but bind
            # every business payload mirror to the requested revision so
            # public HISTORY results cannot accidentally expose the tail
            # revision as if it belonged to this row.
            "revision": source_revision,
            "revisions": [dict(revision)],
            "value_fields": values,
            "evidence_refs": list(revision.get("evidence_refs", ()) or ()),
            "proposal_id": str(revision.get("proposal_id") or ""),
            "relation": str(revision.get("relation") or ""),
            "qualifiers": qualifiers,
            "previous_revision": revision.get("previous_revision"),
            "record_kind": CatalogRecordKind.CLAIM_REVISION.value,
            "source_kind": "canonical_claim",
            "catalog_record_key": self._claim_catalog_record_key(metadata, source_revision),
            "tree_paths": list(tree_paths),
            "primary_tree_path": tree_paths[0],
            "source_uri": projected.uri,
            "source_digest": serving_digest,
            "source_revision": source_revision,
            "state": str(revision.get("state") or ""),
            "canonical_value": raw_value,
            "epistemic_status": str(revision.get("epistemic_status") or ""),
            "created_at": created_at,
            "updated_at": transaction_time,
            "semantic_relation": str(revision.get("relation") or ""),
            "display_fields": display_fields,
            "display_field_evidence_refs": display_evidence,
            "event_time": event_time,
            "ingested_at": created_at,
            "transaction_time": transaction_time,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "validity_end_derived": bool(
                not self._revision_payload(metadata, source_revision).get("valid_to") and valid_to
            ),
            "l0_text": l0_text,
            "l1_text": l1_text,
            "l2_uri": catalog_l2_uri,
            "serving_tier": ServingTier.HOT.value,
            "projection_status": CatalogProjectionStatus.PROJECTED.value,
            "projection_effect_hash": record.input_effect_hash,
            "projection_source_revision": source_revision,
        },
        source_kind="canonical_claim",
    )
    for field, expected in proof_fields.items():
        if safe.metadata.get(field) != expected:
            raise ProjectionIntegrityError(f"Claim Catalog sanitizer did not preserve proof field {field}")
    projected.title = safe.title
    projected.metadata = safe.metadata
    catalog = CatalogRecord.from_context_object(
        projected,
        content=safe.l1_text,
        record_key=self._claim_catalog_record_key(metadata, source_revision),
        record_kind=CatalogRecordKind.CLAIM_REVISION.value,
        tree_paths=tree_paths,
    )
    return replace(
        catalog,
        title=safe.title,
        l0_text=safe.l0_text,
        l1_text=safe.l1_text,
        l2_uri=catalog_l2_uri,
        source_digest=serving_digest,
        canonical_revision=source_revision,
        canonical_state=str(revision.get("state") or ""),
    )


def _reconcile_claim_catalog_projections(
    self: CanonicalMemoryProjector,
    obj: ContextObject,
    metadata: dict[str, Any],
    *,
    published_revision: int,
) -> None:
    """Refresh every existing row from one bounded current Claim read.

    Revision payload, event/transaction time, and validity remain bound to
    each immutable revision.  Tenant/owner/scope/authority/tree placement
    instead follows the current authoritative Claim so a repair or
    reclassification cannot leave stale ACL/path/FTS/vector candidates.
    """

    get_catalog = getattr(self.index_store, "get_catalog", None)
    upsert_catalog = getattr(self.index_store, "upsert_catalog", None)
    if not callable(get_catalog) or not callable(upsert_catalog):
        return
    revisions = self._bounded_claim_revisions(metadata)
    for raw_revision in revisions:
        revision_number = int(raw_revision.get("revision", 0) or 0)
        if revision_number == published_revision:
            continue
        effective = revision_payload_with_effective_validity(revisions, revision_number)
        record_key = self._claim_catalog_record_key(metadata, revision_number)
        loaded = get_catalog(record_key)
        if loaded is None:
            continue
        if not isinstance(loaded, CatalogRecord):
            raise ProjectionIntegrityError("canonical Claim refresh loaded an invalid Catalog record")
        existing = loaded
        proof = self._revision_bound_projection_proof(existing)
        l0_text, l1_text, l2_text = self._sanitized_revision_layers(
            obj,
            metadata,
            effective,
            revision_number,
        )
        refreshed = self._claim_revision_catalog_record(
            obj,
            metadata,
            proof,
            effective,
            proof_metadata={
                **dict(existing.metadata),
                # Serving ACL/scope identity follows the current Source.
                # The original receipt/publication remains immutable and
                # is validated by ``_revision_bound_projection_proof``.
                "claim_uri": obj.uri,
                "tenant_id": str(obj.tenant_id or "default"),
                "owner_user_id": str(obj.owner_user_id or ""),
                "canonical_kind": "claim",
                # A crash/rebuild may have left the disposable Catalog row
                # naming an equivalent but unpublished retry.  The proof
                # returned above is the exact immutable publication
                # attempt; always restore every attempt-owned field from
                # it instead of carrying the retry identity forward.
                "projection_revision": proof.projection_revision,
                "projection_attempt_id": proof.projection_attempt_id,
                "projection_input_effect_hash": proof.input_effect_hash,
                "projection_publish_token": proof.publish_token,
                "projection_content_digest": proof.projected_content_digest,
                "projection_relation_digest": proof.projected_relation_digest,
                "projection_manifest_uri": proof.manifest_uri,
                "projection_record_path": str(self.record_store.attempt_path_for(proof)),
            },
            l0_text=l0_text,
            l1_text=l1_text,
            l2_text=l2_text,
        )
        if existing.tenant_id != refreshed.tenant_id:
            if self.vector_store is not None:
                self.vector_store.delete_vector(vector_row_id(existing.tenant_id, existing.record_key))
            delete_catalog = getattr(self.index_store, "delete_catalog", None)
            if not callable(delete_catalog) or not delete_catalog(
                existing.record_key,
                tenant_id=existing.tenant_id,
            ):
                raise ProjectionIntegrityError("canonical Claim tenant repair could not retire its old row")
        upsert_catalog(refreshed)
        self._refresh_claim_vector(refreshed, proof=proof)


def _revision_bound_projection_proof(self: CanonicalMemoryProjector, existing: CatalogRecord) -> ProjectionRecord:
    """Load the exact attempt named by a serving row and verify publication.

    ``ProjectionRecordStore.load(claim, revision)`` intentionally chooses a
    preferred attempt and may therefore select a later failed/stale retry.
    A historical refresh must instead remain bound to the attempt that was
    actually published for this Catalog row.
    """

    metadata = dict(existing.metadata)
    claim_uri = str(existing.canonical_claim_uri or metadata.get("claim_uri") or existing.uri)
    source_revision = int(existing.source_revision or existing.canonical_revision or 0)
    attempt_id = str(metadata.get("projection_attempt_id") or "")
    if not claim_uri or source_revision < 1 or not attempt_id:
        raise ProjectionIntegrityError("canonical Claim refresh proof identity is incomplete")
    proof = self.record_store.load(
        claim_uri,
        source_revision,
        projection_attempt_id=attempt_id,
    )
    if proof is None:
        raise ProjectionIntegrityError("canonical Claim refresh has no exact projection attempt")
    expected_attempt = {
        "source_revision": source_revision,
        "projection_revision": int(metadata.get("projection_revision", 0) or 0),
        "projection_attempt_id": attempt_id,
        "input_effect_hash": str(metadata.get("projection_input_effect_hash") or ""),
        "publish_token": str(metadata.get("projection_publish_token") or ""),
        "projected_content_digest": str(metadata.get("projection_content_digest") or ""),
        "projected_relation_digest": str(metadata.get("projection_relation_digest") or ""),
    }
    actual_attempt = {
        "source_revision": proof.source_revision,
        "projection_revision": proof.projection_revision,
        "projection_attempt_id": proof.projection_attempt_id,
        "input_effect_hash": proof.input_effect_hash,
        "publish_token": proof.publish_token,
        "projected_content_digest": proof.projected_content_digest,
        "projected_relation_digest": proof.projected_relation_digest,
    }
    if expected_attempt != actual_attempt:
        raise ProjectionIntegrityError("canonical Claim refresh attempt differs from Catalog proof")
    transaction_id = str(metadata.get("current_transaction_id") or "")
    if not transaction_id:
        raise ProjectionIntegrityError("canonical Claim refresh has no immutable transaction identity")
    proof_store = ProjectionProofStore(self.root)
    publication = proof_store.load_publication(transaction_id)
    if publication is None:
        # Compatibility for a pre-publication projection.  With no
        # immutable publication to recover from, only a fully completed
        # exact attempt is admissible.
        if proof.status not in {ProjectionStatus.COMPLETED.value, ProjectionStatus.STALE.value}:
            raise ProjectionIntegrityError("canonical Claim refresh attempt was never successfully published")
        terminal_steps = {ProjectionStepStatus.COMPLETED.value, ProjectionStepStatus.SKIPPED.value}
        if any(
            status not in terminal_steps
            for status in (
                proof.index_status,
                proof.vector_status,
                proof.relation_status,
                proof.scope_status,
                proof.taxonomy_status,
            )
        ):
            raise ProjectionIntegrityError("canonical Claim refresh attempt has incomplete component state")
        return proof

    receipt = self._verified_publication_receipt(publication)
    matches = [
        item
        for item in publication.get("claims", ())
        if isinstance(item, dict)
        and str(item.get("claim_uri") or "") == claim_uri
        and int(item.get("source_revision", 0) or 0) == source_revision
    ]
    if len(matches) != 1:
        raise ProjectionIntegrityError("canonical Claim refresh publication binding is not unique")
    published = dict(matches[0])
    published_domain = dict(published.get("domain_identity", {}) or {})
    immutable_domain_fields = (
        "claim_uri",
        "canonical_kind",
        "claim_state",
        "canonical_head_digest",
        "current_transaction_id",
        "current_receipt_digest",
        "current_claim_revision",
    )
    if str(publication.get("receipt_digest") or "") != str(metadata.get("current_receipt_digest") or "") or any(
        metadata.get(field) != published_domain.get(field) for field in immutable_domain_fields
    ):
        raise ProjectionIntegrityError("canonical Claim refresh differs from immutable publication")
    published_proof = self._verified_projection_record_from_publication(
        published,
        receipt=receipt,
    )
    if (
        proof.source_revision != published_proof.source_revision
        or proof.projection_revision != published_proof.projection_revision
        or proof.input_effect_hash != published_proof.input_effect_hash
        or proof.claim_uri != published_proof.claim_uri
        or proof.slot_uri != published_proof.slot_uri
    ):
        raise ProjectionIntegrityError("canonical Claim retry differs from immutable published effect")
    return published_proof


def _claim_catalog_record_key(metadata: Any, source_revision: int) -> str:
    values = dict(metadata) if isinstance(metadata, dict) else {}
    claim_id = str(values.get("claim_id") or "")
    if not claim_id or source_revision < 1:
        raise ProjectionIntegrityError("canonical Claim Catalog identity is incomplete")
    return f"claim:{claim_id}:revision:{source_revision}"


def _canonical_tree_paths(self: CanonicalMemoryProjector, metadata: dict[str, Any]) -> tuple[str, ...]:
    """Derive bounded schema-owned paths without affecting Canonical Identity."""

    identity = dict(metadata.get("identity_fields", {}) or {})
    memory_type = str(metadata.get("memory_type") or "state")
    category = {
        "preference": "preferences",
        "profile": "profiles",
        "project_rule": "rules",
        "project_decision": "decisions",
        "agent_experience": "experiences",
        "entity": "entities",
        "event": "events",
    }.get(memory_type, "state")
    dynamic: tuple[Any, ...]
    if memory_type == "preference":
        dynamic = (identity.get("subject") or "general", identity.get("dimension") or "general")
    elif memory_type == "profile":
        dynamic = (identity.get("attribute_key") or "general",)
    elif memory_type == "project_rule":
        dynamic = (identity.get("rule_topic") or "general",)
    elif memory_type == "project_decision":
        dynamic = (identity.get("decision_topic") or "general",)
    elif memory_type == "agent_experience":
        dynamic = (identity.get("task_pattern") or "general",)
    elif memory_type == "entity":
        dynamic = (identity.get("canonical_entity_id") or "general",)
    elif memory_type == "event":
        dynamic = (identity.get("event_type") or identity.get("subject") or "general",)
    else:
        dynamic = (memory_type, identity.get("dimension") or identity.get("subject") or "general")
    paths = ["/".join(("memories", category, *(self._canonical_path_segment(value) for value in dynamic)))]
    raw_scope = metadata.get("scope")
    if not isinstance(raw_scope, dict):
        raise ProjectionIntegrityError("canonical Claim scope is missing from projection")
    try:
        scope = MemoryScope.from_dict(raw_scope)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionIntegrityError("canonical Claim scope is invalid for tree projection") from exc
    for scope_ref in scope.applicability.all_of:
        if scope_ref.kind == "workspace":
            paths.append(f"projects/{self._canonical_path_segment(scope_ref.id)}")
            break
    return validate_tree_paths(tuple(paths))


def _canonical_path_segment(self: CanonicalMemoryProjector, value: Any) -> str:
    text = canonical_json(value) if isinstance(value, dict | list | tuple) else str(value or "general")
    return self._segment(text)
