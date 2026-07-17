"""Canonical-memory endpoint policy for generic ordinary relations."""

from __future__ import annotations

from collections.abc import Callable

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.ordinary_relations import OrdinaryRelationEligibility
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.integration.classification import (
    is_canonical_memory_object,
    is_canonical_memory_uri,
)


class CanonicalMemoryRelationPolicy:
    def owns_uri(self, uri: str) -> bool:
        return is_canonical_memory_uri(uri)

    def owns_object(self, obj: ContextObject) -> bool:
        return is_canonical_memory_object(obj)

    def validate_target(
        self,
        obj: ContextObject,
        *,
        role: str,
        source_store: SourceStore,
        tenant_id: str,
        domain_reader: Callable[[str], ContextObject] | None,
    ) -> OrdinaryRelationEligibility:
        if role == "source":
            return OrdinaryRelationEligibility(
                False,
                "canonical Source requires an immutable receipt",
            )
        metadata = dict(obj.metadata or {})
        kind = str(metadata.get("canonical_kind") or "")
        if kind == "claim":
            try:
                current_revision = int(
                    metadata.get("current_revision", metadata.get("revision", 0))
                    or 0
                )
            except (TypeError, ValueError):
                return OrdinaryRelationEligibility(
                    False,
                    "canonical Claim current revision is invalid",
                )
            revisions = [
                dict(item)
                for item in metadata.get("revisions", []) or []
                if isinstance(item, dict)
                and int(item.get("revision", 0) or 0) == current_revision
            ]
            if (
                len(revisions) != 1
                or str(revisions[0].get("state") or "").upper() != "ACTIVE"
            ):
                return OrdinaryRelationEligibility(
                    False,
                    "canonical Claim is not ACTIVE",
                )
            if str(metadata.get("state") or "").upper() != "ACTIVE":
                return OrdinaryRelationEligibility(
                    False,
                    "canonical Claim materialized state is inconsistent",
                )
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            try:
                slot = (
                    domain_reader(slot_uri)
                    if domain_reader is not None
                    else source_store.read_object(slot_uri)
                )
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return OrdinaryRelationEligibility(
                    False,
                    "canonical Claim Slot is missing",
                )
            slot_metadata = dict(slot.metadata or {})
            if (
                str(slot.tenant_id or "default") != tenant_id
                or str(slot_metadata.get("canonical_kind") or "") != "slot"
                or str(slot_metadata.get("active_claim_id") or "")
                != str(metadata.get("claim_id") or "")
            ):
                return OrdinaryRelationEligibility(
                    False,
                    "canonical Claim is not its Slot current state",
                )
            return OrdinaryRelationEligibility(True)
        if kind == "slot":
            if not str(metadata.get("active_claim_id") or ""):
                return OrdinaryRelationEligibility(
                    False,
                    "canonical Slot has no ACTIVE Claim",
                )
            return OrdinaryRelationEligibility(True)
        return OrdinaryRelationEligibility(
            False,
            "canonical endpoint is not a serving Slot or Claim",
        )


__all__ = ["CanonicalMemoryRelationPolicy"]
