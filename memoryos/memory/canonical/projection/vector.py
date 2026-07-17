"""Vector responsibilities for canonical projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memoryos.contextdb.catalog import (
    CatalogRecord,
    CatalogRecordKind,
    catalog_vector_metadata,
)
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.vector import vector_row_id
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
)

if TYPE_CHECKING:
    from .service import CanonicalMemoryProjector


def _refresh_claim_vector(
    self: CanonicalMemoryProjector,
    record: CatalogRecord,
    *,
    proof: ProjectionRecord | None = None,
) -> None:
    if self.vector_store is None:
        return
    proof = proof or self.record_store.load(record.canonical_claim_uri, record.source_revision)
    if proof is None:
        raise ProjectionIntegrityError("canonical refresh has no revision-bound projection proof")
    embedding = self.embedding_provider.embed("\n".join((record.l0_text, record.l1_text)))
    metadata = dict(record.metadata)
    self.vector_store.upsert_vector(
        vector_row_id(record.tenant_id, record.record_key),
        embedding,
        metadata={
            **catalog_vector_metadata(record, sanitizer=self.sanitizer),
            "public_uri": record.uri,
            "claim_uri": record.canonical_claim_uri,
            "claim_id": record.canonical_claim_id,
            "slot_id": record.canonical_slot_id,
            "canonical_kind": "claim",
            "claim_state": record.canonical_state,
            "current_transaction_id": metadata.get("current_transaction_id"),
            "current_receipt_digest": metadata.get("current_receipt_digest"),
            "current_claim_revision": metadata.get("current_claim_revision"),
            "source_revision": proof.source_revision,
            "projection_revision": proof.projection_revision,
            "projection_attempt_id": proof.projection_attempt_id,
            "input_effect_hash": proof.input_effect_hash,
            "publish_token": proof.publish_token,
            "projected_content_digest": proof.projected_content_digest,
            "projected_relation_digest": proof.projected_relation_digest,
            "embedding_model": self.embedding_provider.model_name,
            "schema_version": "canonical_vector_projection_v5",
        },
    )


def _publish_catalog_vector(
    self: CanonicalMemoryProjector,
    catalog_record: CatalogRecord,
    embedding: list[float],
    record: ProjectionRecord,
) -> None:
    assert self.vector_store is not None
    metadata = dict(catalog_record.metadata)
    self.vector_store.upsert_vector(
        vector_row_id(catalog_record.tenant_id, catalog_record.record_key),
        embedding,
        metadata={
            **catalog_vector_metadata(catalog_record, sanitizer=self.sanitizer),
            "public_uri": catalog_record.uri,
            "claim_uri": catalog_record.canonical_claim_uri,
            "claim_id": catalog_record.canonical_claim_id,
            "slot_id": catalog_record.canonical_slot_id,
            "canonical_kind": "claim",
            "claim_state": metadata.get("claim_state"),
            "current_transaction_id": metadata.get("current_transaction_id"),
            "current_receipt_digest": metadata.get("current_receipt_digest"),
            "current_claim_revision": metadata.get("current_claim_revision"),
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
            "embedding_model": self.embedding_provider.model_name,
            "schema_version": "canonical_vector_projection_v5",
        },
    )


def _publish_vector(
    self: CanonicalMemoryProjector,
    obj: ContextObject,
    embedding: list[float],
    record: ProjectionRecord,
) -> None:
    assert self.vector_store is not None
    catalog_record = CatalogRecord.from_context_object(
        obj,
        record_key=self._claim_catalog_record_key(obj.metadata, record.source_revision),
        record_kind=CatalogRecordKind.CLAIM_REVISION.value,
        tree_paths=tuple(obj.metadata.get("tree_paths", ()) or ()),
    )
    self.vector_store.upsert_vector(
        vector_row_id(catalog_record.tenant_id, catalog_record.record_key),
        embedding,
        metadata={
            **catalog_vector_metadata(catalog_record, sanitizer=self.sanitizer),
            "public_uri": obj.uri,
            "claim_uri": obj.uri,
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
            "schema_version": "canonical_vector_projection_v5",
        },
    )
