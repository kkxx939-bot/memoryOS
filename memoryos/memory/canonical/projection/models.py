"""Data contracts shared by projection services and workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionResult:
    claim_uri: str
    source_revision: int
    status: str
    record_path: str = ""
    projection_attempt_id: str = ""
    input_effect_hash: str = ""


@dataclass(frozen=True)
class _CurrentSlotProjectionTarget:
    slot_uri: str
    slot_id: str
    tenant_id: str
    source_revision: int
    active_claim_id: str | None
    previous_source_revision: int | None = None
    previous_active_claim_id: str | None = None


class ProjectionOutboxIntegrityError(RuntimeError):
    """A projection outbox control file is corrupt or missing."""


_MAX_CLAIM_REVISION_REFRESH = 10_000
_PROJECTION_DOMAIN_IDENTITY_FIELDS = (
    "claim_uri",
    "tenant_id",
    "owner_user_id",
    "canonical_kind",
    "claim_state",
    "canonical_head_digest",
    "current_transaction_id",
    "current_receipt_digest",
    "current_claim_revision",
)
_PROJECTION_ATTEMPT_IDENTITY_FIELDS = (
    "projection_revision",
    "projection_attempt_id",
    "projection_input_effect_hash",
    "projection_publish_token",
    "projection_content_digest",
    "projection_relation_digest",
    "projection_manifest_uri",
)
