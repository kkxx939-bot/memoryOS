"""Memory-owned exact validation for canonical retrieval candidates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.retrieval.fusion import RetrievalCandidate
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.memory.canonical.current_head import artifact_root_for, head_from_receipt_snapshot
from memoryos.memory.canonical.projection_proof import (
    ProjectionProofStore,
    projection_publication_record_digest,
)
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import (
    ClaimState,
    materialized_current_revision_payload,
    revision_payload_with_effective_validity,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalRead,
    committed_content,
    committed_relations,
    read_committed_canonical,
)
from memoryos.operations.commit.outbox_envelope import prepared_intent_digest, validate_outbox
from memoryos.operations.commit.receipt import load_transaction_receipt, receipt_snapshot
from memoryos.security.context_projection import ContextProjectionSanitizer

# A Current Slot requires one exact Slot read and one exact active-Claim read.
# The fixed allowance lets candidate_limit=1 validate one Current result while
# keeping the public online invariant ``source_reads <= candidate_limit + C``.
SOURCE_READ_BOUND_ALLOWANCE = 2


# Catalog metadata is rebuildable and must never become an authority for
# Canonical business payloads.  These fields are the bounded serving/proof/ACL
# envelope that may survive candidate generation.  Slot/Claim identity,
# current value, revision payload, evidence, and scope are overlaid below from
# receipt-proved Source reads.
_SAFE_CANDIDATE_METADATA_FIELDS = frozenset(
    {
        "adapter_id",
        "canonical_head_digest",
        "catalog_record_key",
        "claim_head_digest",
        "claim_receipt_digest",
        "context_type",
        "current_claim_revision",
        "current_receipt_digest",
        "current_transaction_id",
        "degraded_mode",
        "event_time",
        "hotness",
        "ingested_at",
        "owner_user_id",
        "primary_tree_path",
        "project_id",
        "projection_attempt_id",
        "projection_content_digest",
        "projection_effect_hash",
        "projection_input_effect_hash",
        "projection_lag",
        "projection_manifest_uri",
        "projection_publish_token",
        "projection_relation_digest",
        "projection_revision",
        "projection_source",
        "projection_source_revision",
        "projection_status",
        "receipt_digest",
        "record_kind",
        "retrieval_scores",
        "semantic_hotness",
        "serving_tier",
        "session_id",
        "source_digest",
        "source_kind",
        "source_revision",
        "tenant_id",
        "transaction_id",
        "transaction_time",
        "tree_paths",
        "updated_at",
        "valid_from",
        "valid_to",
        "workspace_id",
    }
)

_CURRENT_TAIL_METADATA_FIELDS = frozenset(
    {
        "active_claim_id",
        "active_claim_revision",
        "active_claim_uri",
        "canonical_value",
        "claim_latest_revision",
        "current_revision",
        "display_field_evidence_refs",
        "display_fields",
        "epistemic_status",
        "evidence_refs",
        "field_evidence",
        "field_evidence_refs",
        "identity_algorithm_version",
        "identity_fields",
        "memory_type",
        "previous_revision",
        "proposal_fingerprint",
        "proposal_id",
        "qualifiers",
        "relation",
        "revision",
        "revisions",
        "schema_version",
        "semantic_relation",
        "slot_id",
        "slot_revision",
        "slot_uri",
        "state",
        "transition_profile",
        "value_fields",
    }
)


@dataclass(frozen=True)
class CanonicalResolutionResult:
    candidates: tuple[RetrievalCandidate, ...]
    dropped: tuple[dict[str, Any], ...]
    canonical_candidates: int
    canonical_validated: int
    source_reads: int


class BoundedCanonicalResolver:
    """Validate only the post-fusion candidate set; never capture a snapshot."""

    def __init__(
        self,
        source_store: SourceStore,
        relation_store: RelationStore | None = None,
        projection_store: ProjectionRecordStore | None = None,
    ) -> None:
        self.source_store = source_store
        self.relation_store = relation_store
        self.projection_store = projection_store
        self.sanitizer = ContextProjectionSanitizer()

    def resolve(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
    ) -> CanonicalResolutionResult:
        if len(candidates) > plan.candidate_limit:
            raise ValueError("canonical validation input exceeds candidate_limit")
        selected: list[RetrievalCandidate] = []
        dropped: list[dict[str, Any]] = []
        canonical_candidates = 0
        validated = 0
        source_reads = 0
        source_read_budget = plan.candidate_limit + SOURCE_READ_BOUND_ALLOWANCE
        for candidate in candidates:
            if not candidate.canonical_slot_id and not candidate.canonical_claim_id:
                context_type = str(getattr(candidate.context_type, "value", candidate.context_type))
                if context_type == "memory" and (
                    not plan.owner_user_id or str(candidate.metadata.get("owner_user_id") or "") != plan.owner_user_id
                ):
                    dropped.append(
                        {
                            "record_key": candidate.record_key,
                            "uri": candidate.uri,
                            "drop_reason": "noncanonical_memory_owner_mismatch",
                            "canonical_validation_status": "not_canonical_owner_denied",
                        }
                    )
                    continue
                selected.append(candidate)
                continue
            canonical_candidates += 1
            expected_reads = (
                2
                if candidate.record_kind == "current_slot"
                or (plan.query_intent == RetrievalQueryIntent.CURRENT and candidate.canonical_slot_id)
                else 1
            )
            if source_reads + expected_reads > source_read_budget:
                dropped.append(
                    {
                        "record_key": candidate.record_key,
                        "uri": candidate.uri,
                        "drop_reason": "canonical_source_read_bound",
                        "canonical_validation_status": "not_validated_bound",
                    }
                )
                continue
            # Reserve the complete exact-read budget before touching Source.
            # A missing/corrupt Claim after a successful Slot read must not
            # make failed attempts invisible or let later candidates exceed
            # the hard online bound.
            source_reads += expected_reads
            try:
                if candidate.record_kind == "current_slot" or (
                    plan.query_intent == RetrievalQueryIntent.CURRENT and candidate.canonical_slot_id
                ):
                    resolved, _reads = self._validate_current(candidate, plan)
                else:
                    resolved, _reads = self._validate_revision(candidate, plan)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, KeyError, TypeError, ValueError) as exc:
                dropped.append(
                    {
                        "record_key": candidate.record_key,
                        "uri": candidate.uri,
                        "drop_reason": "canonical_unavailable",
                        "canonical_validation_status": "unavailable",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            if resolved is None:
                dropped.append(
                    {
                        "record_key": candidate.record_key,
                        "uri": candidate.uri,
                        "drop_reason": "stale_canonical_projection",
                        "canonical_validation_status": "stale",
                    }
                )
                continue
            selected.append(resolved)
            validated += 1
        return CanonicalResolutionResult(
            tuple(selected),
            tuple(dropped),
            canonical_candidates,
            validated,
            source_reads,
        )

    def _validate_current(
        self,
        candidate: RetrievalCandidate,
        plan: RetrievalQueryPlan,
    ) -> tuple[RetrievalCandidate | None, int]:
        slot_uri = str(candidate.metadata.get("canonical_slot_uri") or candidate.uri)
        if "/claims/" in slot_uri:
            slot_uri = slot_uri.rsplit("/claims/", 1)[0]
        slot_read = read_committed_canonical(self.source_store, slot_uri, self.relation_store)
        slot_obj = slot_read.object
        slot_metadata = dict(slot_obj.metadata or {})
        if slot_metadata.get("canonical_kind") != "slot":
            return None, 1
        if not self._scope_allowed(slot_obj, slot_metadata, plan):
            return None, 1
        slot_id = str(slot_metadata.get("slot_id") or "")
        active_claim_id = str(slot_metadata.get("active_claim_id") or "")
        if slot_id != candidate.canonical_slot_id or not active_claim_id:
            return None, 1
        if candidate.canonical_claim_id and candidate.canonical_claim_id != active_claim_id:
            return None, 1
        claim_uri = f"{slot_uri}/claims/{active_claim_id}"
        claim_read = read_committed_canonical(self.source_store, claim_uri, self.relation_store)
        claim_obj = claim_read.object
        claim_metadata = dict(claim_obj.metadata or {})
        current = materialized_current_revision_payload(claim_metadata)
        if (
            claim_metadata.get("canonical_kind") != "claim"
            or str(claim_metadata.get("slot_id") or "") != slot_id
            or str(claim_metadata.get("claim_id") or "") != active_claim_id
            or str(current.get("state") or "") != ClaimState.ACTIVE.value
        ):
            return None, 2
        if not self._scope_allowed(claim_obj, claim_metadata, plan):
            return None, 2
        expected_revision = int(candidate.canonical_revision or candidate.metadata.get("active_claim_revision") or 0)
        actual_revision = int(current.get("revision") or 0)
        if expected_revision and expected_revision != actual_revision:
            return None, 2
        if candidate.record_kind == "current_slot":
            if not self._current_slot_proof_matches(
                candidate,
                slot_read=slot_read,
                claim_read=claim_read,
                current_revision=current,
            ):
                return None, 2
        elif not self._proof_matches(candidate, claim_read, actual_revision):
            # Compatibility read for pre-cutover Claim projections.  It is
            # still exact and receipt-bound, but is visibly degraded until a
            # Current Slot row is backfilled.
            return None, 2
        current_values = dict(current.get("value_fields", {}) or {})
        current_value = current_values.get(
            "canonical_value",
            current_values.get("value", claim_metadata.get("canonical_value")),
        )
        canonical_fields = {
            "canonical_validation_status": "validated",
            "canonical_slot_id": slot_id,
            "canonical_claim_id": active_claim_id,
            "slot_id": slot_id,
            "claim_id": active_claim_id,
            "canonical_slot_uri": slot_uri,
            "canonical_claim_uri": claim_uri,
            "canonical_revision": actual_revision,
            "canonical_state": ClaimState.ACTIVE.value,
            "active_claim_revision": actual_revision,
            "projection_lag": 0,
            "canonical_value": current_value,
        }
        metadata = self._safe_candidate_metadata(candidate.metadata)
        metadata.update(
            self._revision_business_metadata(
                current,
                canonical_value=current_value,
            )
        )
        metadata.update(
            {
                "tenant_id": str(claim_obj.tenant_id or "default"),
                "owner_user_id": str(claim_obj.owner_user_id or ""),
                "canonical_kind": "current_slot_projection",
                "memory_type": str(claim_metadata.get("memory_type") or slot_metadata.get("memory_type") or ""),
                "identity_algorithm_version": str(
                    slot_metadata.get("identity_algorithm_version")
                    or claim_metadata.get("identity_algorithm_version")
                    or ""
                ),
                "identity_fields": dict(slot_metadata.get("identity_fields", {}) or {}),
                "canonical_subject": str(
                    claim_metadata.get("canonical_subject") or slot_metadata.get("canonical_subject") or ""
                ),
                "scope_keys": list(slot_metadata.get("scope_keys", ()) or ()),
                "scope": dict(claim_metadata.get("scope", {}) or {}),
                # Compatibility filters are public serving fields, but the
                # disposable candidate cannot authorize them. Rebuild them
                # from the receipt-proved Claim alongside scope/authority.
                "connect": dict(claim_metadata.get("connect", {}) or {}),
                "retrieval_views": list(claim_metadata.get("retrieval_views", ()) or ()),
                "asserted_by": str(claim_metadata.get("asserted_by") or ""),
                "asserted_by_service": str(claim_metadata.get("asserted_by_service") or ""),
                "shared_authority": bool(claim_metadata.get("shared_authority", False)),
                "transition_profile": str(claim_metadata.get("transition_profile") or ""),
                "slot_id": slot_id,
                "slot_uri": slot_uri,
                "slot_revision": int(slot_metadata.get("revision") or 0),
                "active_claim_id": active_claim_id,
                "active_claim_uri": claim_uri,
                "active_claim_revision": actual_revision,
                "claim_id": active_claim_id,
                "claim_uri": claim_uri,
                "claim_latest_revision": int(claim_metadata.get("revision") or actual_revision),
            }
        )
        metadata.update(canonical_fields)
        if candidate.record_kind != "current_slot":
            metadata["degraded_mode"] = "legacy_claim_current_projection"
        if self.projection_store is not None:
            projection = self.projection_store.load_current(claim_uri)
            if (
                projection is not None
                and projection.usable
                and projection.current
                and projection.current_claim_revision == actual_revision
            ):
                # ``projection_record`` is part of the historical SDK result
                # contract.  Load it only after the candidate has passed the
                # bounded Slot/Claim/receipt validation above; this is a
                # single exact control-record read, never a Source scan.
                metadata["projection_record"] = projection.to_dict()
        # The canonical Source bundle contains the full serialized Claim.  A
        # CURRENT compatibility result exposes the receipt-proved current
        # value, not that internal JSON envelope or a mutable projection layer.
        committed_text = current_value if isinstance(current_value, str) else canonical_json(current_value)
        if str(claim_metadata.get("memory_type") or "") == "project_rule":
            identity = dict(
                claim_metadata.get("identity_fields")
                or claim_metadata.get("proposal_identity_fields")
                or slot_metadata.get("identity_fields")
                or {}
            )
            topic = str(identity.get("rule_topic") or "").strip()
            if topic:
                committed_text = f"{topic}: {committed_text}"
        safe = self._sanitize_canonical_egress(
            title=committed_text,
            l1_text=committed_text,
            metadata=metadata,
            source_kind="canonical_claim",
            required_fields={field: value for field, value in canonical_fields.items() if field != "canonical_value"},
        )
        return replace(
            candidate,
            uri=claim_uri,
            title=safe.title,
            text=safe.l1_text,
            l0_text=safe.l1_text,
            l1_text=safe.l1_text,
            source_uri=claim_uri,
            metadata=safe.metadata,
        ), 2

    def _validate_revision(
        self,
        candidate: RetrievalCandidate,
        plan: RetrievalQueryPlan,
    ) -> tuple[RetrievalCandidate | None, int]:
        claim_uri = str(candidate.metadata.get("canonical_claim_uri") or candidate.uri)
        committed = read_committed_canonical(self.source_store, claim_uri, self.relation_store)
        obj = committed.object
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "claim" or not self._scope_allowed(obj, metadata, plan):
            return None, 1
        if candidate.canonical_claim_id and str(metadata.get("claim_id") or "") != candidate.canonical_claim_id:
            return None, 1
        revisions = [item for item in metadata.get("revisions", []) or [] if isinstance(item, dict)]
        requested = int(candidate.canonical_revision or candidate.metadata.get("source_revision") or 0)
        raw_revision = next((item for item in revisions if int(item.get("revision") or 0) == requested), None)
        if raw_revision is None:
            return None, 1
        proved_revision = self._validate_historical_publication(
            candidate,
            claim_uri=claim_uri,
            requested=requested,
            plan=plan,
        )
        if proved_revision is None or canonical_digest(proved_revision) != canonical_digest(raw_revision):
            return None, 1
        revision = revision_payload_with_effective_validity(revisions, requested)
        if plan.query_intent == RetrievalQueryIntent.AS_OF:
            valid_at = str(plan.valid_at or "")
            if str(revision.get("state") or "") != ClaimState.ACTIVE.value or not self._valid_at(revision, valid_at):
                return None, 1
        revision_values = dict(revision.get("value_fields", {}) or {})
        revision_value = revision_values.get("canonical_value", revision_values.get("value"))
        canonical_fields = {
            "canonical_validation_status": "validated_history",
            "canonical_claim_id": str(metadata.get("claim_id") or candidate.canonical_claim_id),
            "canonical_slot_id": str(metadata.get("slot_id") or candidate.canonical_slot_id),
            "canonical_claim_uri": claim_uri,
            "canonical_slot_uri": claim_uri.rsplit("/claims/", 1)[0],
            "canonical_revision": requested,
            "source_revision": requested,
            "projection_lag": 0,
            # Historical serving payloads are revision snapshots.  Binding
            # metadata to the Claim-level aggregate would leak the effective
            # current value into HISTORY/AS_OF/CONFLICTS/OPTIONS results when
            # a late-arriving non-current revision carries a different value.
            "canonical_value": revision_value,
            "canonical_state": str(revision.get("state") or ""),
            "valid_from": str(revision.get("valid_from") or ""),
            "valid_to": str(revision.get("valid_to") or ""),
        }
        resolved_metadata = self._safe_candidate_metadata(candidate.metadata)
        resolved_metadata.update(
            self._revision_business_metadata(
                revision,
                canonical_value=revision_value,
            )
        )
        resolved_metadata.update(
            {
                "tenant_id": str(obj.tenant_id or "default"),
                "owner_user_id": str(obj.owner_user_id or ""),
                "canonical_kind": "claim",
                "memory_type": str(metadata.get("memory_type") or ""),
                "identity_algorithm_version": str(metadata.get("identity_algorithm_version") or ""),
                "canonical_subject": str(metadata.get("canonical_subject") or ""),
                "scope": dict(metadata.get("scope", {}) or {}),
                "connect": dict(metadata.get("connect", {}) or {}),
                "retrieval_views": list(metadata.get("retrieval_views", ()) or ()),
                "asserted_by": str(metadata.get("asserted_by") or ""),
                "asserted_by_service": str(metadata.get("asserted_by_service") or ""),
                "shared_authority": bool(metadata.get("shared_authority", False)),
                "transition_profile": str(metadata.get("transition_profile") or ""),
                "slot_id": str(metadata.get("slot_id") or candidate.canonical_slot_id),
                "slot_uri": claim_uri.rsplit("/claims/", 1)[0],
                "claim_id": str(metadata.get("claim_id") or candidate.canonical_claim_id),
                "claim_uri": claim_uri,
            }
        )
        resolved_metadata.update(canonical_fields)
        content = revision_value if isinstance(revision_value, str) else canonical_json(revision_value)
        safe = self._sanitize_canonical_egress(
            title=content,
            l1_text=content,
            metadata=resolved_metadata,
            source_kind="canonical_claim",
            required_fields={field: value for field, value in canonical_fields.items() if field != "canonical_value"},
        )
        return replace(
            candidate,
            title=safe.title,
            text=safe.l1_text,
            l0_text=safe.l1_text,
            l1_text=safe.l1_text,
            source_uri=claim_uri,
            metadata=safe.metadata,
        ), 1

    def _validate_historical_publication(
        self,
        candidate: RetrievalCandidate,
        *,
        claim_uri: str,
        requested: int,
        plan: RetrievalQueryPlan,
    ) -> dict[str, Any] | None:
        """Bind one history row to its immutable receipt and publication.

        This is a constant number of exact control-file reads for the final
        bounded candidate.  It never scans receipts, publications, Claims, or
        projection records.
        """

        metadata = dict(candidate.metadata)
        transaction_id = str(metadata.get("current_transaction_id") or metadata.get("transaction_id") or "")
        expected_head = str(metadata.get("canonical_head_digest") or "")
        expected_receipt = str(metadata.get("receipt_digest") or metadata.get("current_receipt_digest") or "")
        expected_effect = str(
            metadata.get("projection_effect_hash") or metadata.get("projection_input_effect_hash") or ""
        )
        root = artifact_root_for(self.source_store)
        if not transaction_id or not expected_head or not expected_receipt or not expected_effect or root is None:
            return None
        try:
            outbox_path = root / "system" / "outbox" / f"{transaction_id}.json"
            resolved_root = root.resolve()
            expected_outbox = resolved_root / "system" / "outbox" / f"{transaction_id}.json"
            if outbox_path.is_symlink() or outbox_path.resolve(strict=True) != expected_outbox:
                return None
            outbox = validate_outbox(
                json.loads(outbox_path.read_text(encoding="utf-8")),
                transaction_id=transaction_id,
                tenant_id=str(plan.tenant_id or "default"),
                allowed_statuses={"committed"},
            )
            receipt_relative = Path(str(outbox.get("receipt_path") or ""))
            receipt_path = root / receipt_relative
            resolved_receipt = receipt_path.resolve(strict=True)
            if (
                receipt_relative.is_absolute()
                or receipt_path.is_symlink()
                or resolved_receipt == resolved_root
                or resolved_root not in resolved_receipt.parents
            ):
                return None
            receipt = load_transaction_receipt(resolved_receipt)
            if (
                str(receipt.get("transaction_id") or "") != transaction_id
                or str(receipt.get("receipt_digest") or "") != expected_receipt
                or str(outbox.get("receipt_digest") or "") != expected_receipt
                or str(receipt.get("commit_group_id") or "") != str(outbox.get("commit_group_id") or "")
                or str(receipt.get("prepared_intent_digest") or "") != prepared_intent_digest(outbox)
            ):
                return None
            proof_store = ProjectionProofStore(root)
            publication = proof_store.load_publication(transaction_id)
            completion = proof_store.load_completion(transaction_id)
            if publication is None:
                return None
            if completion is not None:
                for key in (
                    "commit_group_id",
                    "transaction_id",
                    "job_id",
                    "tenant_id",
                    "user_id",
                    "queue_identity_digest",
                    "outbox_digest",
                    "receipt_digest",
                    "prepared_intent_digest",
                    "operation_ids",
                    "claim_revisions",
                    "claims",
                    "publication_digest",
                ):
                    if completion.get(key) != publication.get(key):
                        return None
            if (
                publication.get("receipt_digest") != expected_receipt
                or publication.get("outbox_digest") != outbox.get("outbox_digest")
                or publication.get("prepared_intent_digest") != receipt.get("prepared_intent_digest")
            ):
                return None
            claim_proofs = publication.get("claims")
            if not isinstance(claim_proofs, list):
                return None
            matches = [
                item
                for item in claim_proofs
                if isinstance(item, dict)
                and str(item.get("claim_uri") or "") == claim_uri
                and int(item.get("source_revision") or 0) == requested
            ]
            if len(matches) != 1:
                return None
            claim_proof = matches[0]
            domain_identity = claim_proof.get("domain_identity")
            if (
                not isinstance(domain_identity, dict)
                or str(domain_identity.get("canonical_head_digest") or "") != expected_head
                or str(domain_identity.get("current_receipt_digest") or "") != expected_receipt
                or str(domain_identity.get("current_transaction_id") or "") != transaction_id
                or str(claim_proof.get("input_effect_hash") or "") != expected_effect
            ):
                return None
            if self.projection_store is None:
                return None
            projection_attempt_id = str(claim_proof.get("projection_attempt_id") or "")
            projection_record = self.projection_store.load(
                claim_uri,
                requested,
                projection_attempt_id=projection_attempt_id,
            )
            if (
                not projection_attempt_id
                or projection_record is None
                or projection_publication_record_digest(projection_record)
                != str(claim_proof.get("publication_record_digest") or "")
            ):
                return None
            snapshot = receipt_snapshot(receipt, claim_uri)
            snapshot_obj = ContextObject.from_dict(dict(snapshot["object"]))
            if (
                str(snapshot.get("canonical_kind") or dict(snapshot_obj.metadata or {}).get("canonical_kind") or "")
                != "claim"
                or int(snapshot.get("after_revision") or 0) != requested
                or str(head_from_receipt_snapshot(snapshot, receipt).get("head_digest") or "") != expected_head
            ):
                return None
            historical = CommittedCanonicalRead(snapshot_obj, receipt=receipt)
            actual_effect = canonical_digest(
                {
                    "claim_uri": snapshot_obj.uri,
                    "source_revision": requested,
                    "object": snapshot_obj.to_dict(),
                    "content": committed_content(historical),
                    "relations": sorted(
                        (relation.to_dict() for relation in committed_relations(historical)),
                        key=canonical_json,
                    ),
                }
            )
            snapshot_metadata = dict(snapshot_obj.metadata or {})
            snapshot_revisions = [
                item for item in snapshot_metadata.get("revisions", []) or [] if isinstance(item, dict)
            ]
            proved_revision = next(
                (item for item in snapshot_revisions if int(item.get("revision") or 0) == requested),
                None,
            )
            return proved_revision if actual_effect == expected_effect else None
        except (OSError, KeyError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _current_slot_proof_matches(
        candidate: RetrievalCandidate,
        *,
        slot_read: Any,
        claim_read: Any,
        current_revision: dict[str, Any],
    ) -> bool:
        slot_head = dict(slot_read.head or {})
        claim_head = dict(claim_read.head or {})
        metadata = dict(candidate.metadata)
        if (
            str(metadata.get("canonical_head_digest") or "") != str(slot_head.get("head_digest") or "")
            or str(metadata.get("receipt_digest") or "") != str(slot_head.get("receipt_digest") or "")
            or str(metadata.get("claim_head_digest") or "") != str(claim_head.get("head_digest") or "")
            or str(metadata.get("claim_receipt_digest") or "") != str(claim_head.get("receipt_digest") or "")
        ):
            return False
        expected_effect = str(metadata.get("projection_effect_hash") or "")
        if not expected_effect:
            return False
        actual_effect = canonical_digest(
            {
                "slot": slot_read.object.to_dict(),
                "active_claim": claim_read.object.to_dict(),
                "active_revision": current_revision,
                "slot_head_digest": slot_head.get("head_digest"),
                "slot_receipt_digest": slot_head.get("receipt_digest"),
                "claim_head_digest": claim_head.get("head_digest"),
                "claim_receipt_digest": claim_head.get("receipt_digest"),
            }
        )
        return actual_effect == expected_effect

    def _proof_matches(self, candidate: RetrievalCandidate, committed: Any, revision: int) -> bool:
        head = dict(committed.head or {})
        expected_head = str(candidate.metadata.get("canonical_head_digest") or "")
        expected_receipt = str(
            candidate.metadata.get("receipt_digest") or candidate.metadata.get("current_receipt_digest") or ""
        )
        if expected_head and expected_head != str(head.get("head_digest") or ""):
            return False
        if expected_receipt and expected_receipt != str(head.get("receipt_digest") or ""):
            return False
        if int(head.get("current_revision") or 0) != int(revision):
            return False
        expected_effect = str(
            candidate.metadata.get("projection_effect_hash")
            or candidate.metadata.get("projection_input_effect_hash")
            or ""
        )
        if expected_effect:
            obj = committed.object
            effect = canonical_digest(
                {
                    "claim_uri": obj.uri,
                    "source_revision": revision,
                    "object": obj.to_dict(),
                    "content": committed_content(committed),
                    "relations": sorted(
                        (relation.to_dict() for relation in committed_relations(committed)),
                        key=canonical_json,
                    ),
                }
            )
            if effect != expected_effect:
                return False
        return True

    @staticmethod
    def _safe_candidate_metadata(metadata: Any) -> dict[str, Any]:
        """Keep only non-authoritative serving/proof fields from Catalog.

        The Catalog, FTS, and Vector layers are disposable.  They may select a
        bounded candidate and carry immutable proof identifiers, but their
        arbitrary metadata must not be reflected through the public API after
        Canonical resolution.
        """

        raw = dict(metadata or {})
        safe = {field: raw[field] for field in _SAFE_CANDIDATE_METADATA_FIELDS if field in raw}
        for field in _CURRENT_TAIL_METADATA_FIELDS:
            safe.pop(field, None)
        return safe

    @staticmethod
    def _revision_business_metadata(
        revision: Any,
        *,
        canonical_value: Any,
    ) -> dict[str, Any]:
        """Build the complete public revision mirror from one proved payload."""

        raw = dict(revision or {})
        values = dict(raw.get("value_fields", {}) or {})
        evidence_refs = [dict(item) for item in raw.get("evidence_refs", ()) or () if isinstance(item, dict)]
        field_evidence_refs = {
            str(field): [dict(item) for item in refs or () if isinstance(item, dict)]
            for field, refs in dict(raw.get("field_evidence_refs", {}) or {}).items()
        }
        qualifiers = dict(raw.get("qualifiers", {}) or {})
        display_fields = dict(qualifiers.get("display_fields", {}) or {})
        display_evidence = dict(qualifiers.get("display_field_evidence_refs", {}) or {})
        revision_number = int(raw.get("revision") or 0)
        created_at = str(raw.get("created_at") or "")
        transaction_time = str(raw.get("transaction_time") or created_at)
        valid_from = str(raw.get("valid_from") or created_at)
        valid_to = str(raw.get("valid_to") or "")
        bound_revision = {
            "revision": revision_number,
            "state": str(raw.get("state") or ""),
            "value_fields": values,
            "evidence_refs": evidence_refs,
            "field_evidence_refs": field_evidence_refs,
            "proposal_id": str(raw.get("proposal_id") or ""),
            "relation": str(raw.get("relation") or ""),
            "epistemic_status": str(raw.get("epistemic_status") or ""),
            "proposal_fingerprint": str(raw.get("proposal_fingerprint") or ""),
            "extractor_version": str(raw.get("extractor_version") or ""),
            "model_id": raw.get("model_id"),
            "prompt_version": str(raw.get("prompt_version") or ""),
            "policy_version": str(raw.get("policy_version") or ""),
            "schema_version": str(raw.get("schema_version") or ""),
            "qualifiers": qualifiers,
            "created_at": created_at,
            "transaction_time": transaction_time,
            "previous_revision": raw.get("previous_revision"),
            "valid_from": valid_from,
            "valid_to": valid_to or None,
        }
        return {
            **bound_revision,
            "revisions": [bound_revision],
            "current_revision": revision_number,
            "canonical_value": canonical_value,
            "canonical_state": str(raw.get("state") or ""),
            "semantic_relation": str(raw.get("relation") or ""),
            "display_fields": display_fields,
            "display_field_evidence_refs": display_evidence,
            "event_time": str(raw.get("event_time") or raw.get("occurred_at") or valid_from),
            "ingested_at": created_at,
            "updated_at": transaction_time,
        }

    @staticmethod
    def _scope_allowed(obj: Any, metadata: dict[str, Any], plan: RetrievalQueryPlan) -> bool:
        if str(obj.tenant_id or "default") != str(plan.tenant_id or "default"):
            return False
        try:
            scope = MemoryScope.from_dict(metadata["scope"])
        except (KeyError, TypeError, ValueError):
            return False
        if scope.canonical_subject is None or scope.authority.inferred:
            return False
        workspaces = {item.id for item in scope.applicability.all_of if item.kind == "workspace"}
        visibility_allowed = scope.visibility.permits(
            tenant_id=str(plan.tenant_id or "default"),
            principal_id=plan.owner_user_id,
            service_id=plan.service_id,
        )
        if not visibility_allowed:
            return False
        asserted_by = str(metadata.get("asserted_by") or "")
        asserted_service = str(metadata.get("asserted_by_service") or "")
        if (scope.authority.principal_ids or scope.authority.service_ids) and not (
            asserted_by in set(scope.authority.principal_ids) or asserted_service in set(scope.authority.service_ids)
        ):
            return False
        if plan.workspace_ids:
            if plan.workspace_ids == ("__memoryos_principal_only__",):
                if workspaces:
                    return False
            elif workspaces and not workspaces.intersection(plan.workspace_ids):
                return False
        if "applicability_scope_keys" in plan.metadata_filters:
            available_scopes = {str(value) for value in plan.metadata_filters.get("applicability_scope_keys", ()) or ()}
        else:
            # Directly constructed plans predate QueryPlanner scope binding.
            # Keep that internal compatibility without widening a planner-bound
            # explicit narrowing filter.
            available_scopes = set()
            if plan.owner_user_id:
                available_scopes.add(f"memoryos:principal:{plan.owner_user_id}")
            available_scopes.update(
                f"memoryos:workspace:{workspace_id}"
                for workspace_id in plan.workspace_ids
                if workspace_id != "__memoryos_principal_only__"
            )
        required_scopes = {item.key for item in scope.applicability.all_of}
        if required_scopes and not required_scopes.issubset(available_scopes):
            return False
        return True

    @staticmethod
    def _valid_at(revision: dict[str, Any], valid_at: str) -> bool:
        start = str(revision.get("valid_from") or "")
        end = str(revision.get("valid_to") or "")
        if not start or not valid_at:
            return False
        point = datetime.fromisoformat(valid_at.replace("Z", "+00:00"))
        start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if point.tzinfo is None or start_time.tzinfo is None:
            raise ValueError("canonical AS_OF timestamps must include a timezone")
        point = point.astimezone(timezone.utc)
        start_time = start_time.astimezone(timezone.utc)
        if point < start_time:
            return False
        if not end:
            return True
        end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if end_time.tzinfo is None:
            raise ValueError("canonical AS_OF timestamps must include a timezone")
        return point < end_time.astimezone(timezone.utc)

    def _sanitize_canonical_egress(
        self,
        *,
        title: object,
        l1_text: object,
        metadata: dict[str, Any],
        source_kind: str,
        required_fields: dict[str, Any],
    ) -> Any:
        """Sanitize the complete public result and retain its exact proof IDs."""

        safe = self.sanitizer.sanitize(
            title=title,
            l1_text=l1_text,
            metadata=metadata,
            source_kind=source_kind,
        )
        for field, expected in required_fields.items():
            if safe.metadata.get(field) != expected:
                raise ValueError(f"canonical egress sanitization did not preserve {field}")
        for field in (
            "canonical_head_digest",
            "receipt_digest",
            "current_receipt_digest",
            "projection_effect_hash",
            "projection_input_effect_hash",
            "transaction_id",
            "current_transaction_id",
        ):
            expected = metadata.get(field)
            if expected not in (None, "") and safe.metadata.get(field) != expected:
                raise ValueError(f"canonical egress sanitization did not preserve {field}")
        return safe


__all__ = ["BoundedCanonicalResolver", "CanonicalResolutionResult"]
