"""Canonical memory state and structural invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.clock import utc_now
from memoryos.core.integrity import canonicalize
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2
from memoryos.memory.canonical.scope import ScopeRef


class CanonicalMemoryInvariantError(ValueError):
    """Base class for corrupted canonical state."""


class RevisionSequenceError(CanonicalMemoryInvariantError):
    def __init__(self, claim_id: str, revisions: tuple[int, ...]) -> None:
        self.claim_id = claim_id
        self.revisions = revisions
        super().__init__(f"claim {claim_id} has a non-contiguous revision sequence: {revisions}")


class MissingClaimInvariantError(CanonicalMemoryInvariantError):
    def __init__(self, slot_id: str, missing_claim_ids: tuple[str, ...]) -> None:
        self.slot_id = slot_id
        self.missing_claim_ids = missing_claim_ids
        super().__init__(f"slot {slot_id} is missing claims: {','.join(missing_claim_ids)}")


class ActiveClaimInvariantError(CanonicalMemoryInvariantError):
    def __init__(
        self,
        slot_id: str,
        active_claim_ids: tuple[str, ...],
        declared_active_claim_id: str | None,
    ) -> None:
        self.slot_id = slot_id
        self.active_claim_ids = active_claim_ids
        self.declared_active_claim_id = declared_active_claim_id
        super().__init__(
            f"slot {slot_id} ACTIVE invariant violated: active_claim_ids={active_claim_ids}, "
            f"active_claim_id={declared_active_claim_id}"
        )


class TransitionProfile(str, Enum):
    AUTHORITATIVE_STATE = "AUTHORITATIVE_STATE"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPERIENCE = "EXPERIENCE"


class ClaimState(str, Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONFLICTED = "CONFLICTED"
    RETRACTED = "RETRACTED"


CLAIM_STATES = frozenset(state.value for state in ClaimState)


def materialized_current_revision_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one revision selected by the canonical current pointer.

    ``revision`` is the immutable history tail while ``current_revision`` is
    the effective state pointer. They intentionally differ for a late
    historical assertion, so callers must never infer current state from list
    order.
    """

    try:
        current_revision = int(metadata.get("current_revision", metadata.get("revision", 0)) or 0)
    except (TypeError, ValueError) as exc:
        raise CanonicalMemoryInvariantError("canonical current revision pointer is invalid") from exc
    if current_revision < 1:
        raise CanonicalMemoryInvariantError("canonical current revision pointer is missing")
    matches: list[dict[str, Any]] = []
    for item in metadata.get("revisions", []) or []:
        if not isinstance(item, Mapping):
            continue
        try:
            revision = int(item.get("revision", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise CanonicalMemoryInvariantError("canonical revision number is invalid") from exc
        if revision == current_revision:
            matches.append(dict(item))
    if len(matches) != 1:
        raise CanonicalMemoryInvariantError("canonical current revision pointer must select exactly one revision")
    qualifiers = dict(matches[0].get("qualifiers", {}) or {})
    if qualifiers.get("non_current_historical", False):
        raise CanonicalMemoryInvariantError("canonical current revision pointer selects a historical-only revision")
    return matches[0]


def revision_payload_with_effective_validity(
    revisions: Sequence[Mapping[str, Any]],
    revision_number: int,
) -> dict[str, Any]:
    """Return one immutable revision with its derived half-open validity end.

    Canonical Source revisions are append-only, so advancing a Claim must not
    rewrite ``valid_to`` in an older serialized revision.  Serving projections
    and bounded AS_OF validation nevertheless need a closed interval.  When an
    explicit end is absent, the earliest later *non-historical* revision start
    deterministically supplies that end.  A late historical assertion can
    therefore end at the already-known effective revision without changing
    either Source payload.
    """

    if revision_number < 1:
        raise CanonicalMemoryInvariantError("canonical revision number must be positive")
    rows: list[tuple[int, dict[str, Any], datetime]] = []
    seen_numbers: set[int] = set()
    for raw in revisions:
        if not isinstance(raw, Mapping):
            raise CanonicalMemoryInvariantError("canonical revision payload is invalid")
        try:
            number = int(raw.get("revision", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise CanonicalMemoryInvariantError("canonical revision number is invalid") from exc
        if number < 1:
            raise CanonicalMemoryInvariantError("canonical revision number must be positive")
        if number in seen_numbers:
            raise CanonicalMemoryInvariantError("canonical revision number is duplicated")
        seen_numbers.add(number)
        row = dict(raw)
        start = _parse_timestamp(str(row.get("valid_from") or ""), "valid_from")
        rows.append((number, row, start))
    matches = [item for item in rows if item[0] == revision_number]
    if len(matches) != 1:
        raise CanonicalMemoryInvariantError("canonical revision must resolve exactly once")
    _number, selected, selected_start = matches[0]
    explicit_end = selected.get("valid_to")
    if explicit_end:
        end = _parse_timestamp(str(explicit_end), "valid_to")
        if end <= selected_start:
            raise CanonicalMemoryInvariantError("canonical revision valid_to must be later than valid_from")
        return selected
    next_effective = min(
        (
            (start, number, row)
            for number, row, start in rows
            if start > selected_start
            and not bool(dict(row.get("qualifiers", {}) or {}).get("non_current_historical", False))
        ),
        default=None,
        key=lambda item: (item[0], item[1]),
    )
    if next_effective is not None:
        selected["valid_to"] = str(next_effective[2].get("valid_from") or "")
    return selected


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"memory revision {label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"memory revision {label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def states_for(profile: TransitionProfile) -> frozenset[str]:  # noqa: ARG001
    return CLAIM_STATES


@dataclass(frozen=True)
class MemoryRevision:
    """An immutable transaction revision with separate effective time."""

    revision: int
    state: str
    value_fields: Mapping[str, Any]
    evidence_refs: tuple[EvidenceRef, ...]
    proposal_id: str
    relation: str
    epistemic_status: str
    field_evidence_refs: Mapping[str, tuple[EvidenceRef, ...]] = field(default_factory=dict)
    proposal_fingerprint: str = ""
    extractor_version: str = ""
    model_id: str | None = None
    prompt_version: str = ""
    policy_version: str = ""
    schema_version: str = ""
    qualifiers: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    previous_revision: int | None = None
    valid_from: str = ""
    valid_to: str | None = None
    transaction_time: str = ""

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("memory revision must be positive")
        if self.state not in CLAIM_STATES:
            raise ValueError(f"invalid claim state: {self.state}")
        object.__setattr__(self, "value_fields", MappingProxyType(dict(self.value_fields)))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(
            self,
            "field_evidence_refs",
            MappingProxyType(
                {str(field_name): tuple(refs) for field_name, refs in sorted(dict(self.field_evidence_refs).items())}
            ),
        )
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if not self.transaction_time:
            object.__setattr__(self, "transaction_time", self.created_at)
        if not self.valid_from:
            object.__setattr__(self, "valid_from", self.created_at)
        valid_from = _parse_timestamp(self.valid_from, "valid_from")
        if self.valid_to is not None:
            valid_to = _parse_timestamp(self.valid_to, "valid_to")
            if valid_to <= valid_from:
                raise ValueError("memory revision valid_to must be later than valid_from")
        _parse_timestamp(self.transaction_time, "transaction_time")

    @property
    def historical_only(self) -> bool:
        return bool(self.qualifiers.get("non_current_historical", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "state": self.state,
            "value_fields": canonicalize(self.value_fields),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "field_evidence_refs": {
                field_name: [ref.to_dict() for ref in refs] for field_name, refs in self.field_evidence_refs.items()
            },
            "proposal_id": self.proposal_id,
            "relation": self.relation,
            "epistemic_status": self.epistemic_status,
            "proposal_fingerprint": self.proposal_fingerprint,
            "extractor_version": self.extractor_version,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "qualifiers": canonicalize(self.qualifiers),
            "created_at": self.created_at,
            "transaction_time": self.transaction_time,
            "previous_revision": self.previous_revision,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryRevision:
        return cls(
            revision=int(payload["revision"]),
            state=str(payload["state"]),
            value_fields=dict(payload.get("value_fields", {}) or {}),
            evidence_refs=tuple(EvidenceRef(**dict(ref)) for ref in payload.get("evidence_refs", []) or []),
            proposal_id=str(payload.get("proposal_id", "")),
            relation=str(payload.get("relation", "UNRELATED")),
            epistemic_status=str(payload.get("epistemic_status", "INFERRED")),
            field_evidence_refs={
                str(field_name): tuple(EvidenceRef(**dict(ref)) for ref in refs)
                for field_name, refs in dict(payload.get("field_evidence_refs", {}) or {}).items()
            },
            proposal_fingerprint=str(payload.get("proposal_fingerprint", "")),
            extractor_version=str(payload.get("extractor_version", "")),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            prompt_version=str(payload.get("prompt_version", "")),
            policy_version=str(payload.get("policy_version", "")),
            schema_version=str(payload.get("schema_version", "")),
            qualifiers=dict(payload.get("qualifiers", {}) or {}),
            created_at=str(payload.get("created_at", "")),
            transaction_time=str(payload.get("transaction_time", "")),
            previous_revision=int(payload["previous_revision"])
            if payload.get("previous_revision") is not None
            else None,
            valid_from=str(payload.get("valid_from", "")),
            valid_to=str(payload["valid_to"]) if payload.get("valid_to") else None,
        )


@dataclass(frozen=True)
class MemoryClaim:
    claim_id: str
    uri: str
    slot_id: str
    canonical_value: str
    profile: TransitionProfile
    revisions: tuple[MemoryRevision, ...]
    identity_algorithm_version: str = IDENTITY_ALGORITHM_V2
    canonical_subject_key: str = ""

    def __post_init__(self) -> None:
        if self.identity_algorithm_version != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("canonical Claim must use Identity V2")
        object.__setattr__(self, "revisions", tuple(self.revisions))
        if not self.revisions:
            raise ValueError("memory claim must have at least one revision")
        revision_numbers = tuple(revision.revision for revision in self.revisions)
        expected = tuple(range(1, len(self.revisions) + 1))
        if revision_numbers != expected:
            raise RevisionSequenceError(self.claim_id, revision_numbers)
        for index, revision in enumerate(self.revisions):
            if revision.state not in states_for(self.profile):
                raise ValueError(f"invalid {self.profile.value} state: {revision.state}")
            expected_previous = index if index else None
            if revision.previous_revision is not None and revision.previous_revision != expected_previous:
                raise RevisionSequenceError(self.claim_id, revision_numbers)
            if index:
                previous = self.revisions[index - 1]
                if previous.valid_to is not None and previous.valid_to != revision.valid_from:
                    raise ValueError("memory revision validity intervals must be contiguous")

    @property
    def latest_revision(self) -> MemoryRevision:
        return self.revisions[-1]

    @property
    def current(self) -> MemoryRevision:
        return next(
            (revision for revision in reversed(self.revisions) if not revision.historical_only), self.latest_revision
        )

    def with_revision(self, revision: MemoryRevision) -> MemoryClaim:
        if revision.revision != self.latest_revision.revision + 1:
            raise ValueError("claim revision must increase by exactly one")
        if revision.state not in states_for(self.profile):
            raise ValueError(f"invalid {self.profile.value} state: {revision.state}")
        revisions = list(self.revisions)
        if not revision.historical_only:
            current_start = _parse_timestamp(self.current.valid_from, "valid_from")
            incoming_start = _parse_timestamp(revision.valid_from, "valid_from")
            if incoming_start <= current_start:
                transaction_time = _parse_timestamp(revision.transaction_time, "transaction_time")
                next_start = max(transaction_time, current_start + timedelta(microseconds=1))
                revision = replace(revision, valid_from=next_start.isoformat())
            # Historical revisions are immutable.  Their effective end is
            # derived from the next non-historical revision rather than
            # retroactively rewriting ``valid_to`` in the stored prefix.
        revisions.append(revision)
        return MemoryClaim(
            self.claim_id,
            self.uri,
            self.slot_id,
            self.canonical_value,
            self.profile,
            tuple(revisions),
            self.identity_algorithm_version,
            self.canonical_subject_key,
        )

    def to_context_object(
        self, *, tenant_id: str, owner_user_id: str, memory_type: str, scope: dict[str, Any]
    ) -> ContextObject:
        current_qualifiers = dict(self.current.to_dict().get("qualifiers", {}) or {})
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=f"{memory_type}: {self.canonical_value}",
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={
                "canonical_kind": "claim",
                "memory_type": memory_type,
                "slot_id": self.slot_id,
                "claim_id": self.claim_id,
                "identity_algorithm_version": self.identity_algorithm_version,
                "canonical_subject": self.canonical_subject_key,
                "asserted_by": owner_user_id,
                "shared_authority": bool(
                    self.canonical_subject_key and ":principal:" not in self.canonical_subject_key
                ),
                "canonical_value": self.canonical_value,
                "transition_profile": self.profile.value,
                "state": self.current.state,
                "epistemic_status": self.current.epistemic_status,
                "semantic_relation": self.current.relation,
                # Compatibility mirrors derive only from the materialized
                # current revision; transition callers cannot maintain a
                # second, independently mutable display state.
                "display_fields": canonicalize(current_qualifiers.get("display_fields", {}) or {}),
                "display_field_evidence_refs": canonicalize(
                    current_qualifiers.get("display_field_evidence_refs", {}) or {}
                ),
                "revision": self.latest_revision.revision,
                "current_revision": self.current.revision,
                "revisions": [revision.to_dict() for revision in self.revisions],
                "scope": scope,
                "projection_pending": True,
            },
            created_at=self.revisions[0].created_at,
            updated_at=self.latest_revision.transaction_time,
            schema_version="canonical_memory_v2",
        )


@dataclass(frozen=True)
class MemorySlot:
    slot_id: str
    uri: str
    memory_type: str
    identity_fields: Mapping[str, Any]
    scope_keys: tuple[str, ...]
    claim_ids: tuple[str, ...] = ()
    active_claim_id: str | None = None
    revision: int = 0
    identity_algorithm_version: str = IDENTITY_ALGORITHM_V2
    canonical_subject_key: str = ""
    canonical_subject: ScopeRef | None = None

    def __post_init__(self) -> None:
        if self.identity_algorithm_version != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("canonical Slot must use Identity V2")
        object.__setattr__(self, "identity_fields", MappingProxyType(dict(self.identity_fields)))
        object.__setattr__(self, "scope_keys", tuple(sorted(dict.fromkeys(self.scope_keys))))
        claim_ids = tuple(self.claim_ids)
        if len(claim_ids) != len(set(claim_ids)):
            raise CanonicalMemoryInvariantError(f"slot {self.slot_id} contains duplicate claim IDs")
        object.__setattr__(self, "claim_ids", claim_ids)
        if self.revision < 0:
            raise ValueError("slot revision cannot be negative")
        if (
            self.canonical_subject is not None
            and self.canonical_subject_key
            and self.canonical_subject.key != self.canonical_subject_key
        ):
            raise ValueError("slot canonical subject payload does not match its identity key")

    def validate_claims(self, claims: tuple[MemoryClaim, ...]) -> None:
        by_id = {claim.claim_id: claim for claim in claims}
        missing = tuple(claim_id for claim_id in self.claim_ids if claim_id not in by_id)
        if missing:
            raise MissingClaimInvariantError(self.slot_id, missing)
        if any(claim.slot_id != self.slot_id for claim in claims):
            raise CanonicalMemoryInvariantError(f"slot {self.slot_id} contains a claim from another slot")
        active = tuple(claim.claim_id for claim in claims if claim.current.state == ClaimState.ACTIVE.value)
        if len(active) > 1 or (active and self.active_claim_id != active[0]) or (not active and self.active_claim_id):
            raise ActiveClaimInvariantError(self.slot_id, active, self.active_claim_id)

    def to_context_object(self, *, tenant_id: str, owner_user_id: str, scope: dict[str, Any]) -> ContextObject:
        scope_payload = dict(scope)
        if self.canonical_subject is not None:
            scope_payload["canonical_subject"] = self.canonical_subject.to_dict()
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=f"slot: {self.memory_type}",
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={
                "canonical_kind": "slot",
                "memory_type": self.memory_type,
                "slot_id": self.slot_id,
                "identity_algorithm_version": self.identity_algorithm_version,
                "canonical_subject": self.canonical_subject_key,
                "asserted_by": owner_user_id,
                "shared_authority": bool(
                    self.canonical_subject_key and ":principal:" not in self.canonical_subject_key
                ),
                "identity_fields": dict(self.identity_fields),
                "scope_keys": list(self.scope_keys),
                "claim_ids": list(self.claim_ids),
                "active_claim_id": self.active_claim_id,
                "revision": self.revision,
                "scope": scope_payload,
                "projection_pending": False,
            },
            schema_version="canonical_memory_v2",
        )


def profile_for(memory_type: str) -> TransitionProfile:
    if memory_type in {"profile", "preference", "project_rule", "project_decision"}:
        return TransitionProfile.AUTHORITATIVE_STATE
    if memory_type == "agent_experience":
        return TransitionProfile.EXPERIENCE
    return TransitionProfile.OBSERVATIONAL
