"""Canonical-memory commit semantics owned by the memory domain.

OperationCommitter calls this explicit handler while retaining its stable fault-
injection surface.
"""

from __future__ import annotations

import json
from typing import Protocol, cast

from memoryos.contextdb.extensions import ContextDomainClassifier
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.errors import RevisionConflictError
from memoryos.core.integrity import canonical_json
from memoryos.memory.canonical.event import resolve_content_path
from memoryos.memory.canonical.evidence import evidence_hash
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2
from memoryos.memory.canonical.proposal import (
    PendingMemoryProposal,
)
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.canonical.visibility import (
    read_committed_canonical,
)
from memoryos.memory.integration.context_overlay import CanonicalMemoryContextOverlay
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class _CanonicalCommitDomainClassifier:
    """Compose canonical classification with a caller-supplied classifier."""

    def __init__(self, fallback: object) -> None:
        self.canonical = CanonicalMemoryContextOverlay()
        self.fallback = fallback

    def owns_uri(self, uri: str) -> bool:
        fallback = getattr(self.fallback, "owns_uri", None)
        return self.canonical.owns_uri(uri) or (callable(fallback) and bool(fallback(uri)))

    def owns_object(self, obj: ContextObject) -> bool:
        fallback = getattr(self.fallback, "owns_object", None)
        return self.canonical.owns_object(obj) or (callable(fallback) and bool(fallback(obj)))


class _DomainAwareStore(Protocol):
    domain_classifier: ContextDomainClassifier


def bind_canonical_commit_domain_classifier(*stores: object) -> CanonicalMemoryContextOverlay:
    """Bind canonical classification for direct OperationCommitter construction.

    Runtime composition injects the same overlay explicitly. This registration
    keeps the historical direct-construction path correct without teaching the
    generic coordinator any Memory types.
    """

    overlay = CanonicalMemoryContextOverlay()
    for store in stores:
        if store is None or not hasattr(store, "domain_classifier"):
            continue
        domain_store = cast(_DomainAwareStore, store)
        current = domain_store.domain_classifier
        if isinstance(current, CanonicalMemoryContextOverlay | _CanonicalCommitDomainClassifier):
            continue
        domain_store.domain_classifier = _CanonicalCommitDomainClassifier(current)
    return overlay


class CanonicalMemoryCommitHandler:
    """Validate and apply canonical-memory commit semantics."""

    @staticmethod
    def _validate_regular_canonical_boundary(
        committer,
        operation: ContextOperation,
        current: ContextObject | None,
        desired: ContextObject | None,
        *,
        allow_existing_add: bool,
    ) -> None:
        """Keep canonical Slot/Claim and pending objects on their formal paths."""

        def canonical_slot_or_claim(obj: ContextObject | None) -> bool:
            if obj is None:
                return False
            kind = str(dict(obj.metadata or {}).get("canonical_kind") or "")
            return (
                kind in {"slot", "claim"}
                or obj.schema_version == "canonical_memory_v2"
                or "/memories/canonical/" in obj.uri
            )

        def pending_proposal(obj: ContextObject | None) -> bool:
            if obj is None:
                return False
            metadata = dict(obj.metadata or {})
            return (
                metadata.get("canonical_kind") == "pending_proposal"
                or obj.schema_version == PendingMemoryProposal.SCHEMA_VERSION
                or "/memories/pending/" in obj.uri
            )

        if canonical_slot_or_claim(current) or canonical_slot_or_claim(desired):
            raise ValueError("canonical Slot and Claim mutations require a canonical transaction")

        current_pending = pending_proposal(current)
        desired_pending = pending_proposal(desired)
        lifecycle_transition = operation.payload.get("pending_lifecycle_transition") is True
        declares_pending = (
            operation.payload.get("canonical_pending_proposal") is True or lifecycle_transition or desired_pending
        )
        if operation.action == OperationAction.ADD:
            if lifecycle_transition:
                raise ValueError("pending proposal creation cannot declare a lifecycle transition")
            if declares_pending:
                if current is not None and not allow_existing_add:
                    raise ValueError("pending proposal ADD cannot overwrite an existing object")
                if desired is None or not desired_pending:
                    raise ValueError("pending proposal ADD requires a canonical pending object")
                pending = PendingMemoryProposal.from_context_object(desired)
                if (
                    pending.lifecycle_state != LifecycleState.PENDING
                    or pending.lifecycle_revision != 1
                    or pending.retry_count != 0
                    or pending.lifecycle_history
                ):
                    raise ValueError("pending proposal ADD must create the initial PENDING lifecycle revision")
                expected = PendingMemoryProposal.create(
                    pending.proposal,
                    pending.scope,
                    tenant_id=str(desired.tenant_id or "default"),
                    owner_user_id=str(desired.owner_user_id or ""),
                    source_role=pending.source_role,
                    pending_reason_code=pending.pending_reason_code,
                    request_identity=pending.request_identity,
                    related_existing_memory_ids=pending.related_existing_memory_ids,
                    retrieval_views=pending.retrieval_views,
                    created_at=pending.created_at,
                )
                expected_obj = pending.to_context_object(
                    tenant_id=str(desired.tenant_id or "default"),
                    owner_user_id=str(desired.owner_user_id or ""),
                )
                if (
                    operation.payload.get("canonical_pending_proposal") is not True
                    or desired.owner_user_id != operation.user_id
                    or operation.payload.get("tenant_id") != str(desired.tenant_id or "default")
                    or operation.payload.get("memory_type") != pending.proposal.memory_type
                    or pending.uri != expected.uri
                    or operation.target_uri != pending.uri
                    or operation.payload.get("content") != pending.content()
                    or operation.payload.get("pending_proposal_id") != pending.proposal_id
                    or operation.payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or canonical_json(desired.to_dict()) != canonical_json(expected_obj.to_dict())
                ):
                    raise ValueError("pending proposal ADD identity or content is invalid")
            return
        if current_pending:
            if operation.action != OperationAction.UPDATE or not lifecycle_transition or not desired_pending:
                raise ValueError("pending proposal mutations require a legal lifecycle UPDATE")
            return
        if declares_pending:
            raise ValueError("pending lifecycle flags cannot target a non-pending object")

    @staticmethod
    def _preflight_canonical_revisions(
        committer,
        operations: list[ContextOperation],
        *,
        check_revisions: bool = True,
    ) -> None:
        tenants: set[str] = set()
        owners: set[str] = set()
        slot_ids: set[str] = set()
        scope_payloads: set[str] = set()
        for operation in operations:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict) or not object_payload.get("uri"):
                raise ValueError("canonical operation requires a context_object URI")
            uri = str(object_payload["uri"])
            metadata = dict(object_payload.get("metadata", {}) or {})
            if committer._canonical_pending_effect(operation):
                if (
                    object_payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or operation.payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or metadata.get("canonical_kind") != "pending_proposal"
                ):
                    raise ValueError("canonical pending lifecycle effect requires a pending proposal object")
                object_tenant = str(object_payload.get("tenant_id") or "default")
                operation_tenant = str(operation.payload.get("tenant_id") or "default")
                object_owner = str(object_payload.get("owner_user_id") or operation.user_id)
                if object_tenant != operation_tenant or object_owner != operation.user_id:
                    raise ValueError("canonical pending lifecycle tenant or owner mismatch")
                scope = dict(metadata.get("scope", {}) or {})
                subject_payload = scope.get("canonical_subject")
                if not isinstance(subject_payload, dict):
                    raise ValueError("canonical pending lifecycle requires an explicit subject")
                tenants.add(object_tenant)
                owners.add(object_owner)
                slot_ids.add(str(operation.payload.get("slot_id") or ""))
                if operation.payload.get("canonical_pending_resolution") is True:
                    scope_payloads.add(json.dumps(scope, ensure_ascii=False, sort_keys=True))
                if not operation.evidence or any(
                    not item.get("event_id") or not item.get("content_hash") for item in operation.evidence
                ):
                    raise ValueError("canonical pending lifecycle effect requires durable evidence references")
                committer._validate_canonical_evidence(operation)
                if check_revisions:
                    committer._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
                continue
            if object_payload.get("schema_version") != "canonical_memory_v2":
                raise ValueError("canonical operation requires canonical_memory_v2 object schema")
            if operation.payload.get("schema_version") != "canonical_memory_v2":
                raise ValueError("canonical operation requires canonical_memory_v2 transaction schema")
            if (
                metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
                or operation.payload.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
            ):
                raise ValueError("canonical operation requires Identity V2")
            if "identity_alias_operations" in operation.payload:
                raise ValueError("Identity V2 canonical transactions cannot contain redirects")
            scope = dict(metadata.get("scope", {}) or {})
            subject_payload = scope.get("canonical_subject")
            subject_key = str(metadata.get("canonical_subject") or "")
            if not isinstance(subject_payload, dict) or not subject_key:
                raise ValueError("canonical operation requires an explicit canonical subject")
            if ScopeRef.from_dict(subject_payload).key != subject_key:
                raise ValueError("canonical operation subject payload does not match Identity V2")
            authority = dict(scope.get("authority", {}) or {})
            if not authority or bool(authority.get("inferred", False)):
                raise ValueError("canonical operation requires non-inferred assertion authority")
            object_tenant = str(object_payload.get("tenant_id") or "default")
            operation_tenant = str(operation.payload.get("tenant_id") or "default")
            object_owner = str(object_payload.get("owner_user_id") or operation.user_id)
            asserted_by = str(metadata.get("asserted_by") or operation.user_id)
            if (
                object_tenant != operation_tenant
                or object_owner != operation.user_id
                or asserted_by != operation.user_id
            ):
                raise ValueError("canonical operation tenant or owner does not match its transaction envelope")
            tenants.add(object_tenant)
            owners.add(object_owner)
            slot_ids.add(str(metadata.get("slot_id") or operation.payload.get("slot_id") or ""))
            scope_payloads.add(json.dumps(metadata.get("scope", {}), ensure_ascii=False, sort_keys=True))
            if not operation.evidence or any(
                not item.get("event_id") or not item.get("content_hash") for item in operation.evidence
            ):
                raise ValueError("canonical operation requires durable evidence references")
            committer._validate_canonical_evidence(operation)
            if check_revisions:
                expected = int(operation.payload.get("expected_revision", 0))
                try:
                    current = read_committed_canonical(
                        committer.source_store,
                        uri,
                        committer.relation_store,
                    ).object
                    actual = int(dict(current.metadata or {}).get("revision", 0))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    actual = 0
                if actual != expected:
                    raise RevisionConflictError(f"revision conflict for {uri}: expected {expected}, actual {actual}")
        if len(tenants) != 1 or len(slot_ids - {""}) != 1 or len(scope_payloads) != 1:
            raise ValueError("canonical transaction must preserve tenant, slot, and scope boundaries")
        committer._validate_pending_resolution_batch(operations)
        committer._validate_pending_correction_batch(operations)

    @staticmethod
    def _validate_canonical_evidence(committer, operation: ContextOperation) -> None:
        store = committer._session_evidence_reader(
            str(operation.payload.get("tenant_id") or "default")
        )
        verified_sources: set[str] = set()
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            metadata = dict(object_payload.get("metadata", {}) or {})
            for revision in metadata.get("revisions", []) or []:
                if not isinstance(revision, dict):
                    raise ValueError("canonical revision evidence payload must be an object")
                if int(revision.get("revision", 0)) != int(metadata.get("revision", 0)):
                    continue
                field_refs = dict(revision.get("field_evidence_refs", {}) or {})
                for field_name, refs in field_refs.items():
                    if not refs:
                        raise ValueError(f"canonical revision has no field evidence for {field_name}")
                # The final-state validator distinguishes changed fields
                # (which require this transaction's evidence) from unchanged
                # fields (which must retain prior provenance).  Requiring all
                # materialized field refs here would make immutable provenance
                # impossible across revisions.
        for payload in operation.evidence:
            source_uri = str(payload.get("source_uri") or "")
            if not source_uri:
                raise ValueError("canonical evidence requires a durable source_uri")
            if source_uri not in verified_sources:
                store.current_manifest(
                    source_uri,
                    tenant_id=str(operation.payload.get("tenant_id") or "default"),
                )
                verified_sources.add(source_uri)
            event_digest = str(payload.get("event_digest") or "")
            required = {
                "event_id",
                "event_digest",
                "event_schema_version",
                "tenant_id",
                "episode_id",
                "actor_id",
                "actor_kind",
                "actor_role",
                "actor_id_inferred",
                "actor_role_inferred",
                "subject_refs",
                "content_path",
                "occurred_at",
                "ingested_at",
                "sequence",
                "evidence_strength",
                "content_hash",
            }
            if any(name not in payload or payload[name] is None or payload[name] == "" for name in required):
                raise ValueError("canonical evidence reference is incomplete")
            event = store.read_event(
                source_uri,
                event_digest,
                tenant_id=str(operation.payload.get("tenant_id") or "default"),
            )
            if str(event.get("event_id")) != str(payload["event_id"]):
                raise ValueError("canonical evidence event ID does not match its immutable digest")
            if str(event.get("episode_id")) != str(payload["episode_id"]) or str(payload["episode_id"]) != str(
                operation.source_episode_id
            ):
                raise ValueError("canonical evidence event is not part of the source episode")
            if str(event.get("schema_version")) != str(payload["event_schema_version"]):
                raise ValueError("canonical evidence schema version mismatch")
            tenant_id = str(operation.payload.get("tenant_id") or "default")
            if str(event.get("tenant_id")) != str(payload["tenant_id"]) or str(payload["tenant_id"]) != tenant_id:
                raise ValueError("canonical evidence tenant mismatch")
            actor = dict(event.get("actor", {}) or {})
            for field_name, evidence_name in (
                ("id", "actor_id"),
                ("kind", "actor_kind"),
                ("role", "actor_role"),
                ("id_inferred", "actor_id_inferred"),
                ("role_inferred", "actor_role_inferred"),
            ):
                if actor.get(field_name) != payload[evidence_name]:
                    raise ValueError(f"canonical evidence actor mismatch: {evidence_name}")
            expected_subjects = tuple(canonical_json(item) for item in event.get("subjects", []) or [])
            if tuple(str(item) for item in payload.get("subject_refs", []) or []) != expected_subjects:
                raise ValueError("canonical evidence subject mismatch")
            content_path = str(payload["content_path"])
            if content_path != str(event.get("content_path") or ""):
                raise ValueError("canonical evidence content path mismatch")
            content = resolve_content_path(event.get("content"), content_path)
            text = content if isinstance(content, str) else canonical_json(content)
            if evidence_hash(text) != str(payload["content_hash"]):
                raise ValueError("canonical evidence content hash no longer matches the archive")
            if not committer._same_evidence_time(event.get("occurred_at"), payload["occurred_at"]):
                raise ValueError("canonical evidence occurred_at mismatch")
            if not committer._same_evidence_time(event.get("ingested_at"), payload["ingested_at"]):
                raise ValueError("canonical evidence ingested_at mismatch")
            if int(event.get("sequence", 0)) != int(payload["sequence"]):
                raise ValueError("canonical evidence sequence mismatch")
            inference = dict(event.get("inference", {}) or {})
            expected_strength = "INFERRED" if any(bool(value) for value in inference.values()) else "EXPLICIT"
            if str(payload["evidence_strength"]) != expected_strength:
                raise ValueError("canonical evidence strength mismatch")
            span_start = payload.get("span_start")
            span_end = payload.get("span_end")
            if (span_start is None) != (span_end is None):
                raise ValueError("canonical evidence span is incomplete")
            if span_start is None or span_end is None:
                continue
            start, end = int(span_start), int(span_end)
            if start < 0 or end <= start or end > len(text):
                raise ValueError("canonical evidence span is invalid")
            quoted_hash = payload.get("quoted_text_hash")
            quoted_text = text[start:end]
            if not quoted_hash or evidence_hash(quoted_text) != str(quoted_hash):
                raise ValueError("canonical evidence quote hash no longer matches the archive")
            if payload.get("quoted_text") != quoted_text:
                raise ValueError("canonical evidence quoted text no longer matches the archive")

    @staticmethod
    def _same_evidence_time(committer, left: object, right: object) -> bool:
        from datetime import datetime, timezone

        try:
            left_time = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
            right_time = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
        except ValueError:
            return False
        if left_time.tzinfo is None:
            left_time = left_time.replace(tzinfo=timezone.utc)
        if right_time.tzinfo is None:
            right_time = right_time.replace(tzinfo=timezone.utc)
        return left_time.astimezone(timezone.utc) == right_time.astimezone(timezone.utc)

    @staticmethod
    def _validate_authoritative_batch(committer, operations: list[ContextOperation]) -> None:
        slot_active: dict[str, str | None] = {}
        active_by_slot: dict[str, list[str]] = {}
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "slot":
                committer._validate_existing_slot_invariant(str(payload.get("uri", "")))
                slot_active[str(metadata.get("slot_id", ""))] = (
                    str(metadata["active_claim_id"]) if metadata.get("active_claim_id") else None
                )
            elif (
                metadata.get("canonical_kind") == "claim"
                and metadata.get("transition_profile") == "AUTHORITATIVE_STATE"
                and metadata.get("state") == "ACTIVE"
            ):
                active_by_slot.setdefault(str(metadata.get("slot_id", "")), []).append(
                    str(metadata.get("claim_id", ""))
                )
        for slot_id, active_claims in active_by_slot.items():
            if len(active_claims) > 1:
                raise ValueError("authoritative slot cannot commit more than one ACTIVE claim")
            declared = slot_active.get(slot_id)
            if declared and active_claims and declared != active_claims[0]:
                raise ValueError("slot active_claim_id does not match active claim revision")

    @staticmethod
    def _validate_existing_slot_invariant(committer, slot_uri: str) -> None:
        if not slot_uri:
            return
        try:
            slot = read_committed_canonical(
                committer.source_store,
                slot_uri,
                committer.relation_store,
            ).object
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return
        metadata = dict(slot.metadata or {})
        claim_ids = [str(item) for item in metadata.get("claim_ids", []) or []]
        active: list[str] = []
        for claim_id in claim_ids:
            try:
                claim = read_committed_canonical(
                    committer.source_store,
                    f"{slot_uri}/claims/{claim_id}",
                    committer.relation_store,
                ).object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            claim_metadata = dict(claim.metadata or {})
            if str(claim_metadata.get("state", "")) == "ACTIVE":
                active.append(str(claim_metadata.get("claim_id", claim_id)))
        if len(active) > 1:
            raise ValueError(f"canonical slot invariant violation: multiple ACTIVE claims for {slot_uri}")
        pointer = str(metadata.get("active_claim_id") or "")
        if pointer and active and pointer != active[0]:
            raise ValueError(f"canonical slot invariant violation: active_claim_id mismatch for {slot_uri}")

    @staticmethod
    def _apply_canonical_source(committer, operation: ContextOperation) -> None:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical operation requires context_object")
        obj = ContextObject.from_dict(payload)
        committer.source_store.write_object(obj, content=str(operation.payload.get("content", "")))
