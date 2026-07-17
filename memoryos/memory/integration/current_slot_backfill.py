"""Memory-owned CurrentSlot migration backfill."""

from __future__ import annotations

import hashlib
from typing import Any

from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.projection_equivalence import (
    ProjectionEquivalenceProof,
    build_projection_equivalence_proof,
)
from memoryos.contextdb.unified_migration import CanonicalBackfillBatchResult


class CurrentSlotMigrationBackfill:
    """Offline bounded rebuild of CurrentSlot rows from committed Slot heads."""

    def __init__(self, source_store: Any, projector: Any) -> None:
        self.source_store = source_store
        self.projector = projector

    def __call__(self, after_slot_uri: str, limit: int) -> CanonicalBackfillBatchResult:
        selected, has_more = self._select_slot_uris(after_slot_uri, limit)
        projected = 0
        proofs: list[ProjectionEquivalenceProof] = []
        checkpoint = after_slot_uri
        for slot_uri in selected:
            result = self.projector.project(slot_uri)
            checkpoint = slot_uri
            if str(getattr(result, "status", "")) == "projected":
                projected += 1
            record = getattr(result, "record", None)
            catalog_store = getattr(self.projector, "catalog_store", None)
            getter = getattr(catalog_store, "get_catalog", None)
            if record is not None and callable(getter):
                actual = getter(result.record_key, tenant_id=record.tenant_id)
                if actual is not None and not isinstance(actual, CatalogRecord):
                    raise TypeError("CurrentSlot proof lookup returned an invalid Catalog record")
                proofs.append(
                    build_projection_equivalence_proof(
                        plane="canonical_current_slot",
                        source_identity=slot_uri,
                        evidence_digest=record.receipt_digest,
                        expected_records=(record,),
                        actual_records=((actual,) if actual is not None else ()),
                    )
                )
        return CanonicalBackfillBatchResult(
            processed_slots=len(selected),
            projected_records=projected,
            checkpoint=checkpoint,
            complete=not has_more,
            equivalence_proofs=tuple(proofs),
        )

    def prove(self, after_slot_uri: str, limit: int) -> CanonicalBackfillBatchResult:
        """Compare Source-derived CurrentSlot rows without repairing Catalog."""

        selected, has_more = self._select_slot_uris(after_slot_uri, limit)
        expected = getattr(self.projector, "expected_projection", None)
        catalog_store = getattr(self.projector, "catalog_store", None)
        getter = getattr(catalog_store, "get_catalog", None)
        if not callable(expected) or not callable(getter):
            raise RuntimeError("CurrentSlot shadow proof requires non-mutating Source and exact Catalog reads")
        proofs: list[ProjectionEquivalenceProof] = []
        checkpoint = after_slot_uri
        for slot_uri in selected:
            result: Any = expected(slot_uri)
            checkpoint = slot_uri
            tenant_id = str(
                getattr(result.record, "tenant_id", "")
                or getattr(self.source_store, "tenant_id", "default")
                or "default"
            )
            actual = getter(result.record_key, tenant_id=tenant_id)
            if actual is not None and not isinstance(actual, CatalogRecord):
                raise TypeError("CurrentSlot proof lookup returned an invalid Catalog record")
            expected_records = (result.record,) if isinstance(result.record, CatalogRecord) else ()
            proofs.append(
                build_projection_equivalence_proof(
                    plane="canonical_current_slot",
                    source_identity=slot_uri,
                    evidence_digest=str(result.evidence_digest),
                    expected_records=expected_records,
                    actual_records=((actual,) if actual is not None else ()),
                )
            )
        return CanonicalBackfillBatchResult(
            processed_slots=len(selected),
            projected_records=0,
            checkpoint=checkpoint,
            complete=not has_more,
            equivalence_proofs=tuple(proofs),
        )

    def _select_slot_uris(self, after_slot_uri: str, limit: int) -> tuple[list[str], bool]:
        if not 1 <= int(limit) <= 1_000:
            raise ValueError("CurrentSlot backfill limit must be between 1 and 1000")
        from memoryos.memory.canonical.current_head import artifact_root_for, iter_current_head_uris

        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is None:
            return [], False
        slot_uris = (
            uri for uri in sorted(iter_current_head_uris(artifact_root, kinds=("slot",))) if uri > after_slot_uri
        )
        selected: list[str] = []
        has_more = False
        for slot_uri in slot_uris:
            if len(selected) >= int(limit):
                has_more = True
                break
            selected.append(slot_uri)
        return selected, has_more

    def source_snapshot(self) -> tuple[str, int]:
        """Hash the receipt-proved CurrentSlot source set for cutover fencing."""

        from memoryos.memory.canonical.current_head import artifact_root_for, iter_current_head_uris

        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is None:
            return hashlib.sha256(b"").hexdigest(), 0
        expected = getattr(self.projector, "expected_projection", None)
        if not callable(expected):
            raise RuntimeError("CurrentSlot cutover snapshot requires receipt-proved expected projections")
        digest = hashlib.sha256()
        count = 0
        for slot_uri in sorted(iter_current_head_uris(artifact_root, kinds=("slot",))):
            result: Any = expected(slot_uri)
            for value in (slot_uri, str(result.evidence_digest), str(result.record_key)):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            count += 1
        return digest.hexdigest(), count



__all__ = ["CurrentSlotMigrationBackfill"]
