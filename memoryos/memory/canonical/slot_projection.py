"""Current Slot serving projection layered beside Claim revision projection.

The canonical Source remains authoritative.  This module reads one exact Slot
through :class:`CanonicalMemoryRepository`, selects its one ACTIVE Claim, and
publishes one rebuildable ``current_slot`` catalog row.  It never enumerates
all Slots/Claims and never replaces the existing Claim revision projection.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    catalog_vector_metadata,
    normalize_timestamp,
    validate_tree_paths,
)
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.store.vector import VectorStore, vector_row_id
from memoryos.core.integrity import canonical_digest, canonicalize
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.state import ClaimState, MemoryClaim, MemorySlot, profile_for
from memoryos.memory.canonical.visibility import CommittedCanonicalRead, read_committed_canonical
from memoryos.security.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)

_PATH_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]+")


class CurrentSlotProjectionIntegrityError(RuntimeError):
    """Current Slot serving state cannot be proved from canonical Source."""


class CurrentSlotCatalogStore(Protocol):
    """Minimum write surface shared with the Session catalog projector."""

    def upsert_catalog(self, record: CatalogRecord) -> None: ...


class CanonicalSlotRepository(Protocol):
    source_store: SourceStore | None
    relation_store: RelationStore | None

    def load_uri(self, slot_uri: str) -> tuple[MemorySlot, tuple[MemoryClaim, ...]]: ...


@dataclass(frozen=True)
class CurrentSlotProjectionProof:
    tenant_id: str
    owner_user_id: str
    transaction_id: str
    slot_head_digest: str
    slot_receipt_digest: str
    claim_head_digest: str
    claim_receipt_digest: str
    projection_effect_hash: str
    projection_source_revision: int

    def __post_init__(self) -> None:
        required = (
            self.tenant_id,
            self.transaction_id,
            self.slot_head_digest,
            self.slot_receipt_digest,
            self.claim_head_digest,
            self.claim_receipt_digest,
            self.projection_effect_hash,
        )
        if any(not str(value or "") for value in required):
            raise CurrentSlotProjectionIntegrityError("current Slot projection proof is incomplete")
        if self.projection_source_revision < 1:
            raise CurrentSlotProjectionIntegrityError("current Slot projection source revision is invalid")


@dataclass(frozen=True)
class CurrentSlotProjectionResult:
    slot_id: str
    record_key: str
    status: str
    active_claim_id: str = ""
    source_revision: int = 0
    record: CatalogRecord | None = None
    evidence_digest: str = ""


class CurrentSlotProjection:
    """Project one exact canonical Slot into one CURRENT serving record."""

    def __init__(
        self,
        repository: CanonicalSlotRepository,
        catalog_store: CurrentSlotCatalogStore,
        *,
        sanitizer: ContextProjectionSanitizer | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.repository = repository
        self.catalog_store = catalog_store
        self.sanitizer = sanitizer or ContextProjectionSanitizer()
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def project(
        self,
        slot_uri: str,
        *,
        tree_paths: Sequence[str] = (),
    ) -> CurrentSlotProjectionResult:
        """Project one exact Slot; no ACTIVE Claim durably retires its row."""

        slot, claims = self.repository.load_uri(slot_uri)
        if slot.uri != slot_uri:
            raise CurrentSlotProjectionIntegrityError("repository returned a different Slot URI")
        slot.validate_claims(claims)
        slot_committed = self._read_committed(slot.uri)
        self._validate_slot_source(slot, slot_committed)
        record_key = self.record_key(slot.slot_id)

        if slot.active_claim_id is None:
            status = self._retire(
                record_key,
                tenant_id=str(slot_committed.object.tenant_id or "default"),
                uri=self.serving_uri(slot.uri),
                source_revision=slot.revision,
                reason=self._no_active_claim_reason(claims),
            )
            return CurrentSlotProjectionResult(
                slot_id=slot.slot_id,
                record_key=record_key,
                status=status,
                source_revision=slot.revision,
            )

        active_claim = next((claim for claim in claims if claim.claim_id == slot.active_claim_id), None)
        if active_claim is None:
            raise CurrentSlotProjectionIntegrityError("Slot active_claim_id has no exact Claim")
        if active_claim.current.state != ClaimState.ACTIVE.value:
            raise CurrentSlotProjectionIntegrityError("Slot active Claim is not ACTIVE")
        claim_committed = self._read_committed(active_claim.uri)
        self._validate_claim_source(slot, active_claim, slot_committed, claim_committed)
        proof = self._proof(slot, active_claim, slot_committed, claim_committed)
        record = self.build_record(
            slot,
            active_claim,
            proof=proof,
            workspace_id=self._workspace_id(slot_committed.object.metadata),
            canonical_metadata=slot_committed.object.metadata,
            tree_paths=tree_paths,
        )
        self.validate_record(
            record,
            expected_slot=slot,
            expected_claim=active_claim,
            expected_head_digest=proof.slot_head_digest,
            expected_receipt_digest=proof.slot_receipt_digest,
            expected_effect_hash=proof.projection_effect_hash,
            expected_claim_head_digest=proof.claim_head_digest,
            expected_claim_receipt_digest=proof.claim_receipt_digest,
            expected_tenant_id=proof.tenant_id,
            expected_owner_user_id=proof.owner_user_id,
            expected_transaction_id=proof.transaction_id,
        )
        vector_row = self._prepare_vector(record)
        self.catalog_store.upsert_catalog(record)
        if vector_row is not None:
            embedding, vector_metadata = vector_row
            try:
                assert self.vector_store is not None
                self.vector_store.upsert_vector(
                    vector_row_id(record.tenant_id, record.record_key),
                    embedding,
                    {**vector_metadata, "public_uri": record.uri},
                )
            except Exception:
                # The canonical state is still valid and FTS-readable, but the
                # outbox must retry before acknowledging the vector projection.
                self.catalog_store.upsert_catalog(
                    replace(record, projection_status=CatalogProjectionStatus.DEGRADED.value)
                )
                raise
        return CurrentSlotProjectionResult(
            slot_id=slot.slot_id,
            record_key=record_key,
            status="projected",
            active_claim_id=active_claim.claim_id,
            source_revision=slot.revision,
            record=record,
            evidence_digest=proof.slot_receipt_digest,
        )

    def expected_projection(
        self,
        slot_uri: str,
        *,
        tree_paths: Sequence[str] = (),
    ) -> CurrentSlotProjectionResult:
        """Build the receipt-proved expected row without mutating derivatives."""

        slot, claims = self.repository.load_uri(slot_uri)
        if slot.uri != slot_uri:
            raise CurrentSlotProjectionIntegrityError("repository returned a different Slot URI")
        slot.validate_claims(claims)
        slot_committed = self._read_committed(slot.uri)
        self._validate_slot_source(slot, slot_committed)
        receipt = slot_committed.receipt
        evidence_digest = str(receipt.get("receipt_digest") or "") if isinstance(receipt, Mapping) else ""
        if not evidence_digest:
            raise CurrentSlotProjectionIntegrityError("current Slot expected projection has no receipt digest")
        record_key = self.record_key(slot.slot_id)
        if slot.active_claim_id is None:
            return CurrentSlotProjectionResult(
                slot_id=slot.slot_id,
                record_key=record_key,
                status="expected_absent",
                source_revision=slot.revision,
                evidence_digest=evidence_digest,
            )
        active_claim = next((claim for claim in claims if claim.claim_id == slot.active_claim_id), None)
        if active_claim is None:
            raise CurrentSlotProjectionIntegrityError("Slot active_claim_id has no exact Claim")
        if active_claim.current.state != ClaimState.ACTIVE.value:
            raise CurrentSlotProjectionIntegrityError("Slot active Claim is not ACTIVE")
        claim_committed = self._read_committed(active_claim.uri)
        self._validate_claim_source(slot, active_claim, slot_committed, claim_committed)
        proof = self._proof(slot, active_claim, slot_committed, claim_committed)
        record = self.build_record(
            slot,
            active_claim,
            proof=proof,
            workspace_id=self._workspace_id(slot_committed.object.metadata),
            canonical_metadata=slot_committed.object.metadata,
            tree_paths=tree_paths,
        )
        self.validate_record(
            record,
            expected_slot=slot,
            expected_claim=active_claim,
            expected_head_digest=proof.slot_head_digest,
            expected_receipt_digest=proof.slot_receipt_digest,
            expected_effect_hash=proof.projection_effect_hash,
            expected_claim_head_digest=proof.claim_head_digest,
            expected_claim_receipt_digest=proof.claim_receipt_digest,
            expected_tenant_id=proof.tenant_id,
            expected_owner_user_id=proof.owner_user_id,
            expected_transaction_id=proof.transaction_id,
        )
        return CurrentSlotProjectionResult(
            slot_id=slot.slot_id,
            record_key=record_key,
            status="expected_present",
            active_claim_id=active_claim.claim_id,
            source_revision=slot.revision,
            record=record,
            evidence_digest=proof.slot_receipt_digest,
        )

    def tombstone_active_claim_switch(
        self,
        *,
        slot_id: str,
        slot_uri: str,
        tenant_id: str,
        previous_active_claim_id: str,
        active_claim_id: str,
        previous_source_revision: int,
        replacement_source_revision: int,
    ) -> str:
        """Durably retire the previous CURRENT row before publishing its replacement.

        The tombstone deliberately carries the previous Slot revision.  Catalog
        stores can therefore reject a stale delete while still accepting the
        strictly newer replacement row during the same outbox replay.
        """

        required = (slot_id, slot_uri, tenant_id, previous_active_claim_id, active_claim_id)
        if any(not isinstance(value, str) or not value for value in required):
            raise CurrentSlotProjectionIntegrityError("active Claim switch identity is incomplete")
        if previous_active_claim_id == active_claim_id:
            raise CurrentSlotProjectionIntegrityError("active Claim switch must change the Claim identity")
        if (
            isinstance(previous_source_revision, bool)
            or isinstance(replacement_source_revision, bool)
            or previous_source_revision < 1
            or replacement_source_revision <= previous_source_revision
        ):
            raise CurrentSlotProjectionIntegrityError("active Claim switch Slot revisions are invalid")
        return self._retire(
            self.record_key(slot_id),
            tenant_id=tenant_id,
            uri=self.serving_uri(slot_uri),
            source_revision=previous_source_revision,
            reason="canonical_active_claim_switched",
            payload={
                "record_kind": CatalogRecordKind.CURRENT_SLOT.value,
                "canonical_slot_id": slot_id,
                "previous_active_claim_id": previous_active_claim_id,
                "active_claim_id": active_claim_id,
                "replacement_source_revision": replacement_source_revision,
            },
        )

    def build_record(
        self,
        slot: MemorySlot,
        active_claim: MemoryClaim,
        *,
        proof: CurrentSlotProjectionProof,
        workspace_id: str = "",
        canonical_metadata: Mapping[str, Any] | None = None,
        tree_paths: Sequence[str] = (),
    ) -> CatalogRecord:
        """Build a sanitized row from an already receipt-proved exact state."""

        self._validate_active_pair(slot, active_claim)
        if proof.projection_source_revision != slot.revision:
            raise CurrentSlotProjectionIntegrityError("projection proof is for another Slot revision")
        current = active_claim.current
        raw_canonical_value = current.value_fields.get("canonical_value", active_claim.canonical_value)
        canonical_value = (
            active_claim.canonical_value if isinstance(raw_canonical_value, str) else canonicalize(raw_canonical_value)
        )
        # Serving metadata keeps the normalized canonical value, while the
        # human-readable layers preserve the source revision's display case.
        display_value = self._display_value(raw_canonical_value)
        identity_text = self._identity_display(slot.identity_fields)
        l0_text = f"{identity_text}: {display_value} [{ClaimState.ACTIVE.value}]"
        l1_text = "\n".join(
            (
                f"# {identity_text}: {display_value}",
                f"- type: {slot.memory_type}",
                f"- identity: {identity_text}",
                f"- state: {current.state}",
                f"- slot: {slot.slot_id}",
                f"- active claim: {active_claim.claim_id}",
                f"- active claim revision: {current.revision}",
            )
        )
        resolved_paths = validate_tree_paths(
            tuple(tree_paths) if tree_paths else self._default_tree_paths(slot),
        )
        transaction_time = current.transaction_time or current.created_at
        metadata: dict[str, Any] = {
            "canonical_kind": "current_slot_projection",
            "record_kind": CatalogRecordKind.CURRENT_SLOT.value,
            "memory_type": slot.memory_type,
            "identity_algorithm_version": slot.identity_algorithm_version,
            "identity_fields": canonicalize(slot.identity_fields),
            "canonical_subject": slot.canonical_subject_key,
            "scope_keys": list(slot.scope_keys),
            "canonical_value": canonical_value,
            "canonical_state": current.state,
            "slot_id": slot.slot_id,
            "slot_uri": slot.uri,
            "slot_revision": slot.revision,
            "active_claim_id": active_claim.claim_id,
            "active_claim_uri": active_claim.uri,
            "active_claim_revision": current.revision,
            "claim_latest_revision": active_claim.latest_revision.revision,
            "transition_profile": active_claim.profile.value,
            "transaction_id": proof.transaction_id,
            "current_head_digest": proof.slot_head_digest,
            "receipt_digest": proof.slot_receipt_digest,
            "claim_head_digest": proof.claim_head_digest,
            "claim_receipt_digest": proof.claim_receipt_digest,
            "projection_source": "canonical_slot_overlay",
            "projection_source_revision": proof.projection_source_revision,
            "projection_effect_hash": proof.projection_effect_hash,
        }
        source_metadata = dict(canonical_metadata or {})
        if isinstance(source_metadata.get("scope"), Mapping):
            metadata["scope"] = dict(source_metadata["scope"])
        for field_name in ("asserted_by", "asserted_by_service"):
            if isinstance(source_metadata.get(field_name), str) and source_metadata[field_name]:
                metadata[field_name] = source_metadata[field_name]
        record = CatalogRecord(
            record_key=self.record_key(slot.slot_id),
            uri=self.serving_uri(slot.uri),
            tenant_id=proof.tenant_id,
            owner_user_id=proof.owner_user_id,
            workspace_id=workspace_id,
            context_type="memory",
            source_kind="canonical_current_slot",
            record_kind=CatalogRecordKind.CURRENT_SLOT.value,
            lifecycle_state="active",
            parent_uri=slot.uri,
            primary_tree_path=resolved_paths[0] if resolved_paths else "",
            tree_paths=resolved_paths,
            created_at=active_claim.revisions[0].created_at,
            updated_at=transaction_time,
            event_time=current.valid_from,
            ingested_at=transaction_time,
            transaction_time=transaction_time,
            valid_from=current.valid_from,
            valid_to=current.valid_to or "",
            title=f"{slot.memory_type}: {identity_text}: {display_value}",
            l0_text=l0_text,
            l1_text=l1_text,
            l2_uri=active_claim.uri,
            source_uri=slot.uri,
            source_digest=proof.projection_effect_hash,
            source_revision=proof.projection_source_revision,
            canonical_slot_id=slot.slot_id,
            canonical_slot_uri=slot.uri,
            canonical_claim_id=active_claim.claim_id,
            canonical_claim_uri=active_claim.uri,
            canonical_revision=current.revision,
            canonical_state=current.state,
            canonical_head_digest=proof.slot_head_digest,
            receipt_digest=proof.slot_receipt_digest,
            projection_effect_hash=proof.projection_effect_hash,
            serving_tier=ServingTier.HOT.value,
            projection_status=CatalogProjectionStatus.PROJECTED.value,
            metadata=metadata,
        ).with_sanitized_projection(self.sanitizer)
        payload_digest = self._record_payload_digest(record)
        return replace(record, metadata={**dict(record.metadata), "projection_payload_digest": payload_digest})

    def validate_record(
        self,
        record: CatalogRecord,
        *,
        expected_slot: MemorySlot,
        expected_claim: MemoryClaim,
        expected_head_digest: str,
        expected_receipt_digest: str,
        expected_effect_hash: str,
        expected_claim_head_digest: str | None = None,
        expected_claim_receipt_digest: str | None = None,
        expected_tenant_id: str | None = None,
        expected_owner_user_id: str | None = None,
        expected_transaction_id: str | None = None,
    ) -> None:
        """Fail closed unless the row is the exact expected current state."""

        self._validate_active_pair(expected_slot, expected_claim)
        if not expected_head_digest or not expected_receipt_digest or not expected_effect_hash:
            raise CurrentSlotProjectionIntegrityError("expected current Slot proof is incomplete")
        current = expected_claim.current
        checks = {
            "record kind": record.record_kind == CatalogRecordKind.CURRENT_SLOT.value,
            "record key": record.record_key == self.record_key(expected_slot.slot_id),
            "serving URI": record.uri == self.serving_uri(expected_slot.uri),
            "Slot ID": record.canonical_slot_id == expected_slot.slot_id,
            "Slot URI": record.canonical_slot_uri == expected_slot.uri,
            "Claim ID": record.canonical_claim_id == expected_claim.claim_id,
            "Claim URI": record.canonical_claim_uri == expected_claim.uri,
            "Claim state": record.canonical_state == ClaimState.ACTIVE.value,
            "Claim revision": record.canonical_revision == current.revision,
            "source revision": record.source_revision == expected_slot.revision,
            "head digest": record.canonical_head_digest == expected_head_digest,
            "receipt digest": record.receipt_digest == expected_receipt_digest,
            "effect hash": record.projection_effect_hash == expected_effect_hash,
            "source digest": record.source_digest == expected_effect_hash,
            "source URI": record.source_uri == expected_slot.uri,
            "L2 URI": record.l2_uri == expected_claim.uri,
            "valid from": record.valid_from == normalize_timestamp(current.valid_from, "valid_from"),
            "valid to": record.valid_to
            == (normalize_timestamp(current.valid_to, "valid_to") if current.valid_to else ""),
            "projection status": record.projection_status == CatalogProjectionStatus.PROJECTED.value,
        }
        if expected_tenant_id is not None:
            checks["tenant"] = record.tenant_id == expected_tenant_id
        if expected_owner_user_id is not None:
            checks["owner"] = record.owner_user_id == expected_owner_user_id
        metadata = dict(record.metadata)
        checks.update(
            {
                "metadata Slot revision": int(metadata.get("slot_revision", -1)) == expected_slot.revision,
                "metadata active Claim": str(metadata.get("active_claim_id") or "") == expected_claim.claim_id,
                "metadata active revision": int(metadata.get("active_claim_revision", -1)) == current.revision,
                "metadata latest revision": int(metadata.get("claim_latest_revision", -1))
                == expected_claim.latest_revision.revision,
                "metadata head": str(metadata.get("current_head_digest") or "") == expected_head_digest,
                "metadata receipt": str(metadata.get("receipt_digest") or "") == expected_receipt_digest,
                "metadata effect": str(metadata.get("projection_effect_hash") or "") == expected_effect_hash,
                "metadata source revision": int(metadata.get("projection_source_revision", -1))
                == expected_slot.revision,
                "metadata payload digest": str(metadata.get("projection_payload_digest") or "")
                == self._record_payload_digest(record),
            }
        )
        if expected_claim_head_digest is not None:
            checks["Claim head digest"] = metadata.get("claim_head_digest") == expected_claim_head_digest
        if expected_claim_receipt_digest is not None:
            checks["Claim receipt digest"] = metadata.get("claim_receipt_digest") == expected_claim_receipt_digest
        if expected_transaction_id is not None:
            checks["transaction ID"] = metadata.get("transaction_id") == expected_transaction_id

        current_value = current.value_fields.get("canonical_value", expected_claim.canonical_value)
        raw_value = expected_claim.canonical_value if isinstance(current_value, str) else canonicalize(current_value)
        safe_expected = self.sanitizer.sanitize(
            title="",
            metadata={
                "canonical_value": raw_value,
                "identity_fields": canonicalize(expected_slot.identity_fields),
            },
            source_kind="canonical_current_slot",
        ).metadata
        checks["canonical value"] = metadata.get("canonical_value") == safe_expected.get("canonical_value")
        checks["identity fields"] = metadata.get("identity_fields") == safe_expected.get("identity_fields")
        failures = tuple(label for label, valid in checks.items() if not valid)
        if failures:
            raise CurrentSlotProjectionIntegrityError(
                f"current Slot projection validation failed: {', '.join(failures)}"
            )

    @staticmethod
    def record_key(slot_id: str) -> str:
        if not str(slot_id or ""):
            raise ValueError("slot_id is required")
        return f"slot:{slot_id}:current"

    @staticmethod
    def serving_uri(slot_uri: str) -> str:
        return f"{str(slot_uri).rstrip('/')}/serving/current"

    def _read_committed(self, uri: str) -> CommittedCanonicalRead:
        source_store = self.repository.source_store
        if source_store is None:
            raise CurrentSlotProjectionIntegrityError("Current Slot projection requires canonical SourceStore")
        return read_committed_canonical(source_store, uri, self.repository.relation_store)

    def _validate_slot_source(self, slot: MemorySlot, committed: CommittedCanonicalRead) -> None:
        obj = committed.object
        metadata = dict(obj.metadata or {})
        head, receipt = self._complete_proof(committed, kind="slot", uri=slot.uri)
        checks = (
            metadata.get("canonical_kind") == "slot",
            metadata.get("slot_id") == slot.slot_id,
            int(metadata.get("revision", -1)) == slot.revision,
            (str(metadata.get("active_claim_id")) if metadata.get("active_claim_id") else None) == slot.active_claim_id,
            int(head.get("current_revision", -1)) == slot.revision,
            head.get("receipt_digest") == receipt.get("receipt_digest"),
        )
        if slot.revision < 1 or not all(checks):
            raise CurrentSlotProjectionIntegrityError("canonical Slot changed while projecting")

    def _validate_claim_source(
        self,
        slot: MemorySlot,
        claim: MemoryClaim,
        slot_committed: CommittedCanonicalRead,
        claim_committed: CommittedCanonicalRead,
    ) -> None:
        obj = claim_committed.object
        metadata = dict(obj.metadata or {})
        head, receipt = self._complete_proof(claim_committed, kind="claim", uri=claim.uri)
        checks = (
            metadata.get("canonical_kind") == "claim",
            metadata.get("slot_id") == slot.slot_id,
            metadata.get("claim_id") == claim.claim_id,
            int(metadata.get("revision", -1)) == claim.latest_revision.revision,
            int(metadata.get("current_revision", -1)) == claim.current.revision,
            metadata.get("state") == claim.current.state,
            int(head.get("current_revision", -1)) == claim.latest_revision.revision,
            head.get("receipt_digest") == receipt.get("receipt_digest"),
            str(obj.tenant_id or "default") == str(slot_committed.object.tenant_id or "default"),
            str(obj.owner_user_id or "") == str(slot_committed.object.owner_user_id or ""),
        )
        if not all(checks):
            raise CurrentSlotProjectionIntegrityError("canonical active Claim changed while projecting")

    def _proof(
        self,
        slot: MemorySlot,
        claim: MemoryClaim,
        slot_committed: CommittedCanonicalRead,
        claim_committed: CommittedCanonicalRead,
    ) -> CurrentSlotProjectionProof:
        slot_head, slot_receipt = self._complete_proof(slot_committed, kind="slot", uri=slot.uri)
        claim_head, claim_receipt = self._complete_proof(claim_committed, kind="claim", uri=claim.uri)
        effect_hash = canonical_digest(
            {
                "slot": slot_committed.object.to_dict(),
                "active_claim": claim_committed.object.to_dict(),
                "active_revision": claim.current.to_dict(),
                "slot_head_digest": slot_head["head_digest"],
                "slot_receipt_digest": slot_head["receipt_digest"],
                "claim_head_digest": claim_head["head_digest"],
                "claim_receipt_digest": claim_head["receipt_digest"],
            }
        )
        return CurrentSlotProjectionProof(
            tenant_id=str(slot_committed.object.tenant_id or "default"),
            owner_user_id=str(slot_committed.object.owner_user_id or ""),
            transaction_id=str(slot_head["current_transaction_id"]),
            slot_head_digest=str(slot_head["head_digest"]),
            slot_receipt_digest=str(slot_receipt["receipt_digest"]),
            claim_head_digest=str(claim_head["head_digest"]),
            claim_receipt_digest=str(claim_receipt["receipt_digest"]),
            projection_effect_hash=effect_hash,
            projection_source_revision=slot.revision,
        )

    @staticmethod
    def _complete_proof(
        committed: CommittedCanonicalRead,
        *,
        kind: str,
        uri: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        head = dict(committed.head or {})
        receipt = dict(committed.receipt or {})
        if (
            head.get("uri") != uri
            or head.get("canonical_kind") != kind
            or not str(head.get("head_digest") or "")
            or not str(head.get("current_transaction_id") or "")
            or not str(head.get("receipt_digest") or "")
            or head.get("receipt_digest") != receipt.get("receipt_digest")
        ):
            raise CurrentSlotProjectionIntegrityError(f"canonical {kind} has no complete current proof")
        return head, receipt

    @staticmethod
    def _validate_active_pair(slot: MemorySlot, claim: MemoryClaim) -> None:
        if (
            slot.active_claim_id != claim.claim_id
            or claim.slot_id != slot.slot_id
            or claim.current.state != ClaimState.ACTIVE.value
            or claim.profile != profile_for(slot.memory_type)
        ):
            raise CurrentSlotProjectionIntegrityError("Current Slot projection is not the exact ACTIVE Claim")

    def _retire(
        self,
        record_key: str,
        *,
        tenant_id: str,
        uri: str,
        source_revision: int,
        reason: str,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        if not reason:
            raise CurrentSlotProjectionIntegrityError("Current Slot tombstone reason is required")
        tombstone_payload = {
            "record_kind": CatalogRecordKind.CURRENT_SLOT.value,
            **dict(payload or {}),
        }
        enqueue = getattr(self.catalog_store, "enqueue_tombstone", None)
        begin = getattr(self.catalog_store, "begin_tombstone_cleanup", None)
        finish = getattr(self.catalog_store, "finish_tombstone_cleanup", None)
        if callable(enqueue) and callable(begin) and callable(finish):
            queued = enqueue(
                tenant_id=tenant_id,
                record_key=record_key,
                reason=reason,
                uri=uri,
                source_revision=source_revision,
                payload=tombstone_payload,
            )
            if not isinstance(queued, Mapping):
                raise CurrentSlotProjectionIntegrityError("Current Slot tombstone enqueue returned invalid state")
            tombstone_id = str(queued.get("tombstone_id") or "")
            begun = begin(tombstone_id)
            if not isinstance(begun, Mapping):
                raise CurrentSlotProjectionIntegrityError("Current Slot tombstone begin returned invalid state")
            status = str(begun.get("status") or "")
            if status == "STALE":
                return "stale"
            if status == "APPLIED":
                return "tombstoned"
            if status != "CLEANING":
                raise CurrentSlotProjectionIntegrityError(
                    f"Current Slot tombstone entered unexpected state: {status or 'missing'}"
                )
            try:
                self._delete_vector(
                    uri,
                    expected_tenant_id=tenant_id,
                    expected_record_key=record_key,
                    max_source_revision=source_revision,
                )
                applied = finish(tombstone_id)
            except Exception as exc:
                marker = getattr(self.catalog_store, "mark_tombstone_cleanup_failed", None)
                if callable(marker):
                    marker(tombstone_id, f"{type(exc).__name__}: {exc}")
                raise
            if not isinstance(applied, Mapping) or str(applied.get("status") or "") != "APPLIED":
                raise CurrentSlotProjectionIntegrityError("Current Slot tombstone did not become APPLIED")
            return "tombstoned"
        apply_tombstone = getattr(self.catalog_store, "apply_tombstone", None)
        if callable(apply_tombstone):
            applied = apply_tombstone(
                tenant_id=tenant_id,
                record_key=record_key,
                reason=reason,
                uri=uri,
                source_revision=source_revision,
                payload=tombstone_payload,
            )
            if not isinstance(applied, Mapping):
                raise CurrentSlotProjectionIntegrityError("Current Slot tombstone apply returned invalid state")
            if str(applied.get("status") or "") == "STALE":
                return "stale"
            self._delete_vector(
                uri,
                expected_tenant_id=tenant_id,
                expected_record_key=record_key,
                max_source_revision=source_revision,
            )
            return "tombstoned"
        tombstone = getattr(self.catalog_store, "tombstone_catalog", None)
        if callable(tombstone):
            tombstone(
                record_key,
                tenant_id=tenant_id,
                reason=reason,
                source_revision=source_revision,
            )
            self._delete_vector(
                uri,
                expected_tenant_id=tenant_id,
                expected_record_key=record_key,
                max_source_revision=source_revision,
            )
            return "tombstoned"
        delete = getattr(self.catalog_store, "delete_catalog", None)
        if callable(delete):
            delete(record_key, tenant_id=tenant_id)
            self._delete_vector(
                uri,
                expected_tenant_id=tenant_id,
                expected_record_key=record_key,
                max_source_revision=source_revision,
            )
            return "deleted"
        raise CurrentSlotProjectionIntegrityError("catalog store cannot tombstone or delete a stale Current Slot")

    def _prepare_vector(self, record: CatalogRecord) -> tuple[list[float], dict[str, Any]] | None:
        if self.vector_store is None or self.embedding_provider is None:
            return None
        text = "\n".join(part for part in (record.title, record.l0_text, record.l1_text) if part)
        if not text:
            raise CurrentSlotProjectionIntegrityError("Current Slot vector projection has no sanitized text")
        embedding = [float(value) for value in self.embedding_provider.embed(text)]
        if not embedding or any(not math.isfinite(value) for value in embedding):
            raise CurrentSlotProjectionIntegrityError("Current Slot embedding is invalid")
        return embedding, {
            **catalog_vector_metadata(record, sanitizer=self.sanitizer),
            "embedding_model": str(getattr(self.embedding_provider, "model_name", "")),
            "schema_version": "canonical_current_slot_vector_v1",
        }

    def _delete_vector(
        self,
        uri: str,
        *,
        expected_tenant_id: str = "",
        expected_record_key: str = "",
        max_source_revision: int = 0,
    ) -> None:
        if self.vector_store is None:
            return
        if not expected_tenant_id or not expected_record_key:
            raise CurrentSlotProjectionIntegrityError("Current Slot vector delete identity is incomplete")
        row_id = vector_row_id(expected_tenant_id, expected_record_key)
        getter = getattr(self.vector_store, "get_vector_metadata", None)
        if callable(getter):
            metadata_value = getter(row_id)
            if metadata_value is None:
                return
            if not isinstance(metadata_value, Mapping):
                return
            metadata = metadata_value
            actual_tenant_id = str(metadata.get("tenant_id") or "")
            if actual_tenant_id and expected_tenant_id and actual_tenant_id != expected_tenant_id:
                return
            actual_record_key = str(metadata.get("catalog_record_key") or "")
            if actual_record_key and expected_record_key and actual_record_key != expected_record_key:
                return
            try:
                actual_revision = int(metadata.get("source_revision") or 0)
            except (TypeError, ValueError):
                return
            if max_source_revision and actual_revision > max_source_revision:
                return
        self.vector_store.delete_vector(row_id)

    @staticmethod
    def _no_active_claim_reason(claims: Sequence[MemoryClaim]) -> str:
        states = {claim.current.state for claim in claims}
        if ClaimState.RETRACTED.value in states:
            return "canonical_claim_retracted"
        if ClaimState.SUPERSEDED.value in states:
            return "canonical_claim_superseded"
        return "canonical_slot_has_no_active_claim"

    def _default_tree_paths(self, slot: MemorySlot) -> tuple[str, ...]:
        identity = dict(slot.identity_fields)
        category_fields = {
            "preference": ("preferences", ("subject", "dimension")),
            "profile": ("profiles", ("attribute_key",)),
            "project_rule": ("rules", ("rule_topic",)),
            "project_decision": ("decisions", ("decision_topic",)),
            "agent_experience": ("experiences", ("task_pattern", "environment_signature")),
            "entity": ("entities", ("entity_type", "entity_key", "name")),
            "event": ("events", ("event_type", "event_id")),
        }
        category, fields = category_fields.get(slot.memory_type, ("state", ()))
        segments = [self._path_segment(identity[name]) for name in fields if identity.get(name) not in {None, ""}]
        if not segments:
            segments.append(self._path_segment(slot.slot_id))
        return ("/".join(("memories", category, *segments)),)

    def _path_segment(self, value: object) -> str:
        rendered = str(value or "").strip()
        try:
            return self.sanitizer.sanitize_tree_segment(rendered)
        except ContextProjectionSanitizationError:
            # Human labels may contain spaces or non-Latin characters.  They
            # remain derived taxonomy only; credentials and paths have already
            # been pseudonymized by the direct attempt above.
            safe = _PATH_UNSAFE.sub("_", rendered).strip("._-")[:120]
            if not safe:
                safe = f"id-{canonical_digest(rendered)[:12]}"
            return self.sanitizer.sanitize_tree_segment(safe)

    @staticmethod
    def _display_value(value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _identity_display(cls, identity_fields: Mapping[str, Any]) -> str:
        """Render stable identity as sanitized serving text, never as identity input."""

        parts: list[str] = []
        for key in sorted(identity_fields):
            raw_value = identity_fields[key]
            if raw_value is None or raw_value == "":
                continue
            label = str(key).replace("_", " ").replace("-", " ")
            value = cls._display_value(raw_value)
            if isinstance(raw_value, str):
                value = value.replace("_", " ").replace("-", " ")
            parts.append(f"{label}: {value}")
        return "; ".join(parts) if parts else "state"

    @staticmethod
    def _workspace_id(metadata: Mapping[str, Any]) -> str:
        scope = metadata.get("scope")
        if not isinstance(scope, Mapping):
            return str(metadata.get("workspace_id") or metadata.get("project_id") or "")
        if scope.get("project_id"):
            return str(scope["project_id"])
        applicability = scope.get("applicability")
        if not isinstance(applicability, Mapping):
            return ""
        for item in applicability.get("all_of", ()) or ():
            if isinstance(item, Mapping) and item.get("kind") == "workspace" and item.get("id"):
                return str(item["id"])
        return ""

    @staticmethod
    def _record_payload_digest(record: CatalogRecord) -> str:
        payload = record.to_dict()
        metadata = dict(payload.get("metadata", {}) or {})
        metadata.pop("projection_payload_digest", None)
        payload["metadata"] = metadata
        return canonical_digest(payload)


def current_slot_projection(
    source_store: SourceStore,
    catalog_store: CurrentSlotCatalogStore,
    slot_uri: str,
    *,
    relation_store: RelationStore | None = None,
    tree_paths: Sequence[str] = (),
) -> CurrentSlotProjectionResult:
    """Small convenience entry point for one exact, bounded projection."""

    repository = CanonicalMemoryRepository(source_store, relation_store)
    return CurrentSlotProjection(repository, catalog_store).project(slot_uri, tree_paths=tree_paths)


__all__ = [
    "CanonicalSlotRepository",
    "CurrentSlotCatalogStore",
    "CurrentSlotProjection",
    "CurrentSlotProjectionIntegrityError",
    "CurrentSlotProjectionProof",
    "CurrentSlotProjectionResult",
    "current_slot_projection",
]
