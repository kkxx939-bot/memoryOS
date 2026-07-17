"""Canonical-memory authoritative-state validation for serving rebuilds."""

from __future__ import annotations

from typing import Any

from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.memory.canonical.projection_state import ProjectionIntegrityError
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.visibility import (
    capture_committed_canonical_snapshot,
    committed_content,
    committed_relations,
)


def validate_canonical_authoritative_state(
    source_store: SourceStore,
    relation_store: RelationStore | None,
    projection_store: Any | None,
) -> dict[str, int]:
    """Validate non-rebuildable canonical truth before derived mutation."""

    snapshot = capture_committed_canonical_snapshot(source_store, relation_store)
    claims = {
        uri: committed
        for uri, committed in snapshot.records.items()
        if str(dict(committed.object.metadata or {}).get("canonical_kind") or "")
        == "claim"
    }
    if projection_store is None:
        if claims:
            raise ProjectionIntegrityError(
                "committed canonical Claims have no projection record store"
            )
        return {
            "canonical_objects": len(snapshot.records),
            "canonical_claims": 0,
            "projection_records": 0,
        }

    current_records = {
        record.claim_uri: record for record in projection_store.iter_current()
    }
    if set(current_records) != set(claims):
        dangling = sorted(set(current_records) - set(claims))
        missing = sorted(set(claims) - set(current_records))
        raise ProjectionIntegrityError(
            f"projection current/head closure mismatch; dangling={dangling}; missing={missing}"
        )

    for claim_uri, committed in claims.items():
        metadata = dict(committed.object.metadata or {})
        source_revision = int(metadata.get("revision", 0) or 0)
        try:
            current_revision = materialized_current_revision_payload(metadata)
        except CanonicalMemoryInvariantError as exc:
            raise ProjectionIntegrityError(
                f"committed Claim has an invalid materialized revision: {claim_uri}"
            ) from exc
        expected_effect_hash = canonical_digest(
            {
                "claim_uri": claim_uri,
                "source_revision": source_revision,
                "object": committed.object.to_dict(),
                "content": committed_content(committed),
                "relations": sorted(
                    (
                        relation.to_dict()
                        for relation in committed_relations(committed)
                    ),
                    key=canonical_json,
                ),
            }
        )
        record = current_records[claim_uri]
        if (
            record.slot_uri != claim_uri.rsplit("/claims/", 1)[0]
            or record.source_revision != source_revision
            or record.projection_revision != source_revision
            or record.current_claim_revision != int(current_revision["revision"])
            or record.input_effect_hash != expected_effect_hash
            or not record.current
            or not record.usable
        ):
            raise ProjectionIntegrityError(
                "projection current record is detached from committed Claim state: "
                f"{claim_uri}"
            )
    return {
        "canonical_objects": len(snapshot.records),
        "canonical_claims": len(claims),
        "projection_records": len(current_records),
    }


__all__ = ["validate_canonical_authoritative_state"]
