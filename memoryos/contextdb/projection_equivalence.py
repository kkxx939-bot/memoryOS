"""Bounded proof that one immutable source produced its exact Catalog rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from memoryos.contextdb.catalog import CatalogRecord
from memoryos.security.context_projection import ContextProjectionSanitizer

MAX_EQUIVALENCE_RECORDS = 1_000


@dataclass(frozen=True)
class ProjectionEquivalenceProof:
    """Payload-free expected-versus-actual projection proof.

    Expected identities are derived from immutable evidence.  Actual identities
    are read back by exact Catalog projection identity, never by running the
    online retrieval query against the same Catalog a second time.
    """

    plane: str
    source_identity_digest: str
    evidence_digest: str
    expected_count: int
    actual_count: int
    expected_digest: str
    actual_digest: str
    matched: bool
    overflow: bool = False

    def to_journal_entry(self) -> dict[str, object]:
        return {
            "plane": self.plane,
            "source_identity_digest": self.source_identity_digest,
            "evidence_digest": self.evidence_digest,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "matched": self.matched,
            "overflow": self.overflow,
        }


def build_projection_equivalence_proof(
    *,
    plane: str,
    source_identity: str,
    evidence_digest: str,
    expected_records: Sequence[CatalogRecord],
    actual_records: Sequence[CatalogRecord],
    sanitizer: ContextProjectionSanitizer | None = None,
    max_records: int = MAX_EQUIVALENCE_RECORDS,
) -> ProjectionEquivalenceProof:
    """Compare sanitized Catalog identities with a hard proof-size bound."""

    policy = sanitizer or ContextProjectionSanitizer()
    if not plane or not source_identity or not evidence_digest:
        raise ValueError("projection equivalence proof source identity is incomplete")
    if not 1 <= int(max_records) <= MAX_EQUIVALENCE_RECORDS:
        raise ValueError("projection equivalence proof bound must be between 1 and 1000")
    overflow = len(expected_records) > max_records or len(actual_records) > max_records
    expected_identities = tuple(sorted(_catalog_identity(record, policy) for record in expected_records[:max_records]))
    actual_identities = tuple(sorted(_catalog_identity(record, policy) for record in actual_records[:max_records]))
    expected_digest = policy.digest(expected_identities)
    actual_digest = policy.digest(actual_identities)
    return ProjectionEquivalenceProof(
        plane=str(plane),
        source_identity_digest=policy.digest(str(source_identity)),
        evidence_digest=str(evidence_digest),
        expected_count=len(expected_records),
        actual_count=len(actual_records),
        expected_digest=expected_digest,
        actual_digest=actual_digest,
        matched=not overflow and expected_digest == actual_digest,
        overflow=overflow,
    )


def _catalog_identity(record: CatalogRecord, sanitizer: ContextProjectionSanitizer) -> str:
    """Hash the full sanitized serving record, excluding no mutable payload."""

    safe = record.with_sanitized_projection(sanitizer)
    payload = safe.to_dict()
    paths = list(safe.tree_paths)
    payload["tree_paths"] = [*paths[:1], *sorted(paths[1:])]
    metadata = dict(payload.get("metadata") or {})
    if isinstance(metadata.get("tree_paths"), list):
        metadata_paths = [str(path) for path in metadata["tree_paths"]]
        metadata["tree_paths"] = [*metadata_paths[:1], *sorted(metadata_paths[1:])]
    payload["metadata"] = metadata
    return sanitizer.digest(payload)


__all__ = [
    "MAX_EQUIVALENCE_RECORDS",
    "ProjectionEquivalenceProof",
    "build_projection_equivalence_proof",
]
