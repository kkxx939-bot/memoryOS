"""Untrusted-operation validation at the canonical commit boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.integrity import canonical_json
from memoryos.memory.canonical.identity import (
    IDENTITY_ALGORITHM_V2,
    AliasRegistry,
    StableMemoryIdentityResolver,
)
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    PendingMemoryProposal,
    SemanticAssessment,
    SpeechAct,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import (
    MemoryClaim,
    MemoryRevision,
    MemorySlot,
    TransitionProfile,
    profile_for,
)
from memoryos.memory.canonical.transition import (
    MemoryTransitionPolicy,
    PendingSemanticReconciliation,
)
from memoryos.memory.schema import FieldMergeMode, MemoryTypeRegistry
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class FinalStateValidationError(ValueError):
    """Base error for a canonical transaction whose final state is unsafe."""


class RevisionPrefixError(FinalStateValidationError):
    pass


class IdentityValidationError(FinalStateValidationError):
    pass


class OperationCompletenessError(FinalStateValidationError):
    pass


class RevisionEvidenceError(FinalStateValidationError):
    pass


def _payload(value: object) -> object:
    serializer = getattr(value, "to_dict", None)
    return serializer() if callable(serializer) else value


@dataclass(frozen=True)
class ValidatedFinalState:
    slot: MemorySlot
    claims: tuple[MemoryClaim, ...]
    changed_uris: tuple[str, ...]


class CanonicalFinalStateValidator:
    """Rebuild and validate the complete final domain state before writes."""

    def __init__(
        self,
        source_store: SourceStore,
        relation_store: RelationStore | None = None,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self.source_store = source_store
        self.relation_store = relation_store
        self.repository = CanonicalMemoryRepository(source_store, relation_store)
        self.identity = StableMemoryIdentityResolver(alias_registry)
        self.registry = MemoryTypeRegistry()

    def validate(
        self,
        operations: list[ContextOperation],
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> ValidatedFinalState | None:
        domain_operations = [
            operation
            for operation in operations
            if operation.payload.get("canonical_pending_resolution") is not True
            and operation.payload.get("canonical_pending_correction") is not True
        ]
        if not domain_operations:
            return None
        if len({operation.operation_id for operation in domain_operations}) != len(domain_operations):
            raise OperationCompletenessError("canonical transaction contains duplicate operation ids")
        proposal_proofs = self._proposal_proofs(domain_operations)
        payloads: dict[str, tuple[ContextOperation, ContextObject]] = {}
        for operation in domain_operations:
            raw = operation.payload.get("context_object")
            if not isinstance(raw, dict):
                raise FinalStateValidationError("canonical final state requires context_object payloads")
            obj = ContextObject.from_dict(raw)
            if obj.uri in payloads:
                raise OperationCompletenessError("canonical transaction contains duplicate object operations")
            payloads[obj.uri] = (operation, obj)
        slot_rows = [row for row in payloads.values() if dict(row[1].metadata or {}).get("canonical_kind") == "slot"]
        if len(slot_rows) != 1:
            raise OperationCompletenessError("state-changing canonical transaction requires exactly one Slot operation")
        slot_operation, slot_obj = slot_rows[0]
        desired_slot = self._slot(slot_obj)
        if str(slot_obj.tenant_id or "default") != tenant_id or slot_obj.owner_user_id != owner_user_id:
            raise FinalStateValidationError("canonical final Slot crosses tenant or owner boundary")
        self._validate_domain_mirrors(
            slot_operation,
            slot_obj,
            desired_slot,
            slot_scope=slot_obj,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        self._validate_proposal_slot_identity(
            proposal_proofs,
            desired_slot,
            slot_obj,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        try:
            current_slot, current_claims = self.repository.load_uri(desired_slot.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            current_slot, current_claims = None, ()

        current_by_id = {claim.claim_id: claim for claim in current_claims}
        transition_relations = self._recompute_transition_relations(
            proposal_proofs,
            current_slot,
            current_claims,
            slot_obj,
            operations=operations,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        desired_claim_operations: dict[str, tuple[ContextOperation, MemoryClaim, ContextObject]] = {}
        for operation, obj in payloads.values():
            metadata = dict(obj.metadata or {})
            if metadata.get("canonical_kind") == "slot":
                continue
            if metadata.get("canonical_kind") != "claim":
                raise OperationCompletenessError("canonical transaction contains an unrelated object operation")
            claim = self._claim(obj, desired_slot)
            self._validate_domain_mirrors(
                operation,
                obj,
                desired_slot,
                slot_scope=slot_obj,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )
            if claim.claim_id in desired_claim_operations:
                raise OperationCompletenessError("canonical transaction contains conflicting Claim operations")
            desired_claim_operations[claim.claim_id] = (operation, claim, obj)

        extra_claims = set(desired_claim_operations) - set(desired_slot.claim_ids)
        if extra_claims:
            raise OperationCompletenessError(
                "canonical transaction contains Claims outside final Slot membership: " + ",".join(sorted(extra_claims))
            )
        if current_slot is not None:
            removed_claims = set(current_slot.claim_ids) - set(desired_slot.claim_ids)
            if removed_claims:
                # Claim membership is canonical history.  Retraction and
                # replacement append a terminal revision; they never make a
                # previously committed Claim disappear from the Slot or its
                # current head-set.
                raise OperationCompletenessError(
                    "canonical transaction removes historical Slot Claims: " + ",".join(sorted(removed_claims))
                )
        final_by_id = dict(current_by_id)
        final_by_id.update({claim_id: row[1] for claim_id, row in desired_claim_operations.items()})
        missing = set(desired_slot.claim_ids) - set(final_by_id)
        if missing:
            raise OperationCompletenessError(
                "canonical transaction omits final Slot Claims: " + ",".join(sorted(missing))
            )
        final_claims = tuple(final_by_id[claim_id] for claim_id in desired_slot.claim_ids)

        self._validate_slot_revision(current_slot, desired_slot, slot_operation, desired_claim_operations)
        self._validate_expected_revision_map(
            domain_operations,
            current_slot,
            current_claims,
            desired_claim_operations,
        )
        changed_uris: set[str] = {desired_slot.uri}
        for claim_id, (operation, desired, _obj) in desired_claim_operations.items():
            current = current_by_id.get(claim_id)
            self._validate_claim_revision(current, desired, operation)
            self._validate_revision_evidence(
                current,
                desired,
                operation,
                memory_type=desired_slot.memory_type,
                proposal_proofs=proposal_proofs,
                transition_relations=transition_relations,
            )
            changed_uris.add(desired.uri)

        # Slot identity is part of the final state even when a malicious
        # transaction tries to submit only a Slot operation. Recompute every
        # final Claim against the final Slot instead of trusting identity
        # fields that arrived in any operation payload.
        for claim in final_claims:
            self._validate_identity(
                desired_slot,
                claim,
                slot_obj,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )

        expected_operation_uris = {
            desired_slot.uri,
            *(claim.uri for claim in final_claims if claim.claim_id in desired_claim_operations),
        }
        actual_operation_uris = set(payloads)
        if actual_operation_uris != expected_operation_uris:
            raise OperationCompletenessError("canonical operation set does not equal the materialized domain delta")

        desired_slot.validate_claims(final_claims)
        self._validate_recomputed_transition(
            proposal_proofs,
            current_slot,
            current_claims,
            desired_slot,
            final_claims,
            desired_claim_operations,
            slot_obj,
            operations=operations,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        self._validate_relation_membership(
            desired_slot,
            final_by_id,
            tenant_id=tenant_id,
        )
        # Validate derived object/content mirrors only after the semantic
        # boundary checks above.  This preserves precise, typed failures for
        # forged identity, revision-prefix, no-op, and domain-invariant
        # violations while still rejecting any stale materialized bundle
        # before the first source write.
        self._validate_slot_materialization(slot_operation, slot_obj, desired_slot)
        for operation, claim, obj in desired_claim_operations.values():
            self._validate_claim_materialization(operation, obj, claim)
        return ValidatedFinalState(desired_slot, final_claims, tuple(sorted(changed_uris)))

    def _proposal_proofs(
        self,
        operations: list[ContextOperation],
    ) -> dict[str, MemorySemanticProposal]:
        proof_sets: set[str] = set()
        parsed: dict[str, MemorySemanticProposal] = {}
        for operation in operations:
            raw_proofs = operation.payload.get("proposal_proofs")
            if (
                not isinstance(raw_proofs, list)
                or not raw_proofs
                or any(not isinstance(item, dict) for item in raw_proofs)
            ):
                raise RevisionEvidenceError("canonical transaction has no complete proposal proof")
            proof_sets.add(canonical_json(raw_proofs))
            operation_proofs: dict[str, MemorySemanticProposal] = {}
            for raw in raw_proofs:
                try:
                    proposal = MemorySemanticProposal.from_dict(raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RevisionEvidenceError("canonical transaction proposal proof is invalid") from exc
                self._validate_proposal_schema(proposal)
                fingerprint = proposal.fingerprint
                if fingerprint in operation_proofs:
                    raise RevisionEvidenceError("canonical transaction contains duplicate proposal proofs")
                operation_proofs[fingerprint] = proposal
            declared = {str(item) for item in operation.payload.get("proposal_fingerprints", []) or []}
            declared_ids = {str(item) for item in operation.payload.get("proposal_ids", []) or []}
            if declared != set(operation_proofs) or declared_ids != {
                proposal.proposal_id for proposal in operation_proofs.values()
            }:
                raise RevisionEvidenceError("canonical transaction proposal proof disagrees with its declared identity")
            parsed = operation_proofs
        if len(proof_sets) != 1:
            raise RevisionEvidenceError("canonical transaction operations disagree on proposal proof")
        operation_evidence = {canonical_json(item) for operation in operations for item in operation.evidence}
        proof_evidence = {
            canonical_json(ref.to_dict()) for proposal in parsed.values() for ref in proposal.evidence_refs
        }
        if operation_evidence != proof_evidence:
            raise RevisionEvidenceError("canonical transaction evidence is detached from its proposal proof")
        return parsed

    def _validate_proposal_schema(self, proposal: MemorySemanticProposal) -> None:
        """Apply the model-facing MemoryType contract at the final trust boundary.

        The built-in LLM parser already rejects unknown fields, but canonical
        operations may also be constructed by SDKs, adapters, internal scripts,
        or tests.  Identity V2 intentionally hashes only declared identity
        fields and FieldMerger otherwise defaults an undeclared value field to
        REPLACE, so accepting either here would silently turn an untrusted
        extension into committed domain state.
        """

        try:
            schema = self.registry.get(proposal.memory_type)
        except ValueError as exc:
            raise RevisionEvidenceError("canonical proposal proof uses an unsupported memory schema") from exc
        expected_identity = set(schema.slot_identity_fields)
        actual_identity = set(proposal.identity_fields)
        if actual_identity != expected_identity:
            raise RevisionEvidenceError("canonical proposal proof identity fields violate its memory schema")
        if set(proposal.value_fields) - set(schema.allowed_value_fields()):
            raise RevisionEvidenceError("canonical proposal proof value fields violate its memory schema")

    def _validate_proposal_slot_identity(
        self,
        proposals: dict[str, MemorySemanticProposal],
        slot: MemorySlot,
        slot_obj: ContextObject,
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> None:
        scope = MemoryScope.from_dict(dict(dict(slot_obj.metadata or {}).get("scope", {}) or {}))
        for proposal in proposals.values():
            try:
                resolved = self.identity.resolve(
                    proposal,
                    scope,
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                )
            except (TypeError, ValueError) as exc:
                raise RevisionEvidenceError("canonical proposal proof has invalid Identity V2 input") from exc
            if resolved.slot_id != slot.slot_id or resolved.slot_uri != slot.uri:
                raise RevisionEvidenceError("canonical proposal proof is detached from the final Slot identity")

    def _recompute_transition_relations(
        self,
        proposals: dict[str, MemorySemanticProposal],
        current_slot: MemorySlot | None,
        current_claims: tuple[MemoryClaim, ...],
        slot_obj: ContextObject,
        *,
        operations: list[ContextOperation],
        tenant_id: str,
        owner_user_id: str,
    ) -> dict[str, str]:
        scope = MemoryScope.from_dict(dict(dict(slot_obj.metadata or {}).get("scope", {}) or {}))
        reviewed = any(operation.payload.get("canonical_pending_resolution") is True for operation in operations)
        reconciler = MemorySemanticReconciler()
        results: dict[str, str] = {}
        for fingerprint, proposal in proposals.items():
            try:
                identity = self.identity.resolve(
                    proposal,
                    scope,
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                )
                reconciliation = reconciler.reconcile(
                    proposal,
                    identity,
                    slot=current_slot,
                    claims=current_claims,
                )
            except (TypeError, ValueError) as exc:
                raise RevisionEvidenceError(
                    "canonical transition relation cannot be recomputed from committed state"
                ) from exc
            relation = reconciliation.relation.value
            semantic = proposal.semantic
            if (
                reviewed
                and relation == "AMBIGUOUS"
                and isinstance(semantic, NormalizedSemanticAssessment)
                and semantic.relation_to_existing.value == "SUPPLEMENTS"
                and reconciliation.claim is not None
                and reconciliation.active_claim is not None
                and reconciliation.claim.claim_id == reconciliation.active_claim.claim_id
            ):
                relation = "SUPPLEMENTS"
            results[fingerprint] = relation
        return results

    def _validate_recomputed_transition(
        self,
        proposals: dict[str, MemorySemanticProposal],
        current_slot: MemorySlot | None,
        current_claims: tuple[MemoryClaim, ...],
        desired_slot: MemorySlot,
        final_claims: tuple[MemoryClaim, ...],
        claim_operations: dict[str, tuple[ContextOperation, MemoryClaim, ContextObject]],
        slot_obj: ContextObject,
        *,
        operations: list[ContextOperation],
        tenant_id: str,
        owner_user_id: str,
    ) -> None:
        if len(proposals) != 1:
            raise OperationCompletenessError(
                "one canonical transaction must be derived from exactly one proposal proof"
            )
        proposal = next(iter(proposals.values()))
        scope = MemoryScope.from_dict(dict(dict(slot_obj.metadata or {}).get("scope", {}) or {}))
        try:
            identity = self.identity.resolve(
                proposal,
                scope,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )
            reconciliation = MemorySemanticReconciler().reconcile(
                proposal,
                identity,
                slot=current_slot,
                claims=current_claims,
            )
            policy = MemoryTransitionPolicy(self.registry)
            resolution_operations = [
                operation for operation in operations if operation.payload.get("canonical_pending_resolution") is True
            ]
            if resolution_operations:
                if len(resolution_operations) != 1 or not resolution_operations[0].target_uri:
                    raise RevisionEvidenceError("canonical reviewed transition has an ambiguous pending authorization")
                pending = self.repository.load_pending(
                    str(resolution_operations[0].target_uri),
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                )
                expected = policy._apply_confirmed_pending_review(
                    self._review_transition_pending(pending),
                    proposal,
                    identity,
                    reconciliation,
                    authorization_id=str(
                        resolution_operations[0].payload.get("confirmation_operation_id")
                        or resolution_operations[0].operation_id
                    ),
                    owner_user_id=owner_user_id,
                    tenant_id=tenant_id,
                )
            elif (
                isinstance(proposal.semantic, NormalizedSemanticAssessment)
                and proposal.semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION}
                and proposal.metadata.get("effect_authority") == "structured_explicit_command"
            ):
                expected = policy._apply_structured_retraction(
                    proposal,
                    identity,
                    reconciliation,
                    authorization_id=proposal.proposal_id,
                    owner_user_id=owner_user_id,
                    tenant_id=tenant_id,
                )
            else:
                expected = policy.apply(proposal, identity, reconciliation)
        except RevisionEvidenceError:
            raise
        except (FileNotFoundError, KeyError, TypeError, ValueError, PendingSemanticReconciliation) as exc:
            raise RevisionEvidenceError(
                "canonical final state is not reproducible by the formal transition policy"
            ) from exc

        changed = set(claim_operations)
        if changed != set(expected.changed_claim_ids):
            raise OperationCompletenessError(
                "canonical Claim operation set differs from the recomputed transition delta"
            )
        if self._slot_domain_payload(desired_slot) != self._slot_domain_payload(expected.slot):
            raise FinalStateValidationError("canonical final Slot differs from the recomputed transition result")
        actual_claims = {claim.claim_id: self._claim_domain_payload(claim) for claim in final_claims}
        expected_claims = {claim.claim_id: self._claim_domain_payload(claim) for claim in expected.claims}
        if actual_claims != expected_claims:
            raise FinalStateValidationError("canonical final Claims differ from the recomputed transition result")

    def _review_transition_pending(
        self,
        pending: PendingMemoryProposal,
    ) -> PendingMemoryProposal:
        display_names = {
            "title",
            "display_name",
            "display_text",
            "source_text",
            "source_wording",
            "summary",
            "details",
            "rationale",
            "reason",
            "decision",
            "rule",
        }
        proposal = pending.proposal
        display = {key: value for key, value in proposal.value_fields.items() if key in display_names}
        if not display:
            return pending
        remaining = {key: value for key, value in proposal.value_fields.items() if key not in display_names}
        display_evidence = {
            field_name: [ref.to_dict() for ref in refs]
            for field_name, refs in proposal.field_evidence_refs.items()
            if field_name.startswith("value.") and field_name.split(".", 1)[1] in display_names
        }
        remaining_evidence = {
            field_name: refs
            for field_name, refs in proposal.field_evidence_refs.items()
            if field_name not in display_evidence
        }
        return replace(
            pending,
            proposal=replace(
                proposal,
                value_fields=remaining,
                field_evidence_refs=remaining_evidence,
                metadata={
                    **dict(proposal.metadata),
                    "display_fields": display,
                    "display_field_evidence_refs": display_evidence,
                },
            ),
        )

    def _slot_domain_payload(self, slot: MemorySlot) -> str:
        return canonical_json(
            {
                "slot_id": slot.slot_id,
                "uri": slot.uri,
                "memory_type": slot.memory_type,
                "identity_fields": dict(slot.identity_fields),
                "scope_keys": list(slot.scope_keys),
                "claim_ids": list(slot.claim_ids),
                "active_claim_id": slot.active_claim_id,
                "revision": slot.revision,
                "identity_algorithm_version": slot.identity_algorithm_version,
                "canonical_subject_key": slot.canonical_subject_key,
                "canonical_subject": (slot.canonical_subject.to_dict() if slot.canonical_subject is not None else None),
            }
        )

    def _claim_domain_payload(self, claim: MemoryClaim) -> str:
        return canonical_json(
            {
                "claim_id": claim.claim_id,
                "uri": claim.uri,
                "slot_id": claim.slot_id,
                "canonical_value": claim.canonical_value,
                "profile": claim.profile.value,
                "revisions": [revision.to_dict() for revision in claim.revisions],
                "identity_algorithm_version": claim.identity_algorithm_version,
                "canonical_subject_key": claim.canonical_subject_key,
            }
        )

    def _validate_domain_mirrors(
        self,
        operation: ContextOperation,
        obj: ContextObject,
        slot: MemorySlot,
        *,
        slot_scope: ContextObject,
        tenant_id: str,
        owner_user_id: str,
    ) -> None:
        metadata = dict(obj.metadata or {})
        slot_metadata = dict(slot_scope.metadata or {})
        expected = {
            "canonical_transaction_id": str(operation.payload.get("transaction_id") or ""),
            "canonical_idempotency_key": str(operation.payload.get("idempotency_key") or ""),
            "commit_group_id": str(operation.payload.get("commit_group_id") or ""),
            "identity_algorithm_version": str(operation.payload.get("identity_algorithm_version") or ""),
            "canonical_subject": str(operation.payload.get("canonical_subject") or ""),
            "slot_id": str(operation.payload.get("slot_id") or ""),
            "memory_type": str(operation.payload.get("memory_type") or ""),
        }
        if any(not value or str(metadata.get(field) or "") != value for field, value in expected.items()):
            raise FinalStateValidationError("canonical object transaction and identity mirrors are inconsistent")
        if (
            operation.user_id != owner_user_id
            or obj.owner_user_id != owner_user_id
            or str(obj.tenant_id or "default") != tenant_id
            or str(operation.payload.get("tenant_id") or "default") != tenant_id
            or str(metadata.get("memory_type") or "") != slot.memory_type
            or str(operation.payload.get("memory_type") or "") != slot.memory_type
            or str(metadata.get("asserted_by") or "") != owner_user_id
            or canonical_json(metadata.get("scope", {})) != canonical_json(slot_metadata.get("scope", {}))
        ):
            raise FinalStateValidationError("canonical object domain mirrors cross the final Slot boundary")

    def _slot(self, obj: ContextObject) -> MemorySlot:
        metadata = dict(obj.metadata or {})
        raw_scope = metadata.get("scope")
        if not isinstance(raw_scope, dict):
            raise FinalStateValidationError("canonical Slot scope is missing")
        scope = MemoryScope.from_dict(raw_scope)
        if scope.canonical_subject is None:
            raise FinalStateValidationError("canonical Slot subject is missing")
        return MemorySlot(
            slot_id=str(metadata.get("slot_id") or ""),
            uri=obj.uri,
            memory_type=str(metadata.get("memory_type") or ""),
            identity_fields=dict(metadata.get("identity_fields", {}) or {}),
            scope_keys=tuple(str(item) for item in metadata.get("scope_keys", []) or []),
            claim_ids=tuple(str(item) for item in metadata.get("claim_ids", []) or []),
            active_claim_id=str(metadata["active_claim_id"]) if metadata.get("active_claim_id") else None,
            revision=int(metadata.get("revision", 0)),
            identity_algorithm_version=str(metadata.get("identity_algorithm_version") or ""),
            canonical_subject_key=str(metadata.get("canonical_subject") or ""),
            canonical_subject=scope.canonical_subject,
        )

    def _claim(self, obj: ContextObject, slot: MemorySlot) -> MemoryClaim:
        metadata = dict(obj.metadata or {})
        revisions = tuple(MemoryRevision.from_dict(item) for item in metadata.get("revisions", []) or [])
        claim = MemoryClaim(
            claim_id=str(metadata.get("claim_id") or ""),
            uri=obj.uri,
            slot_id=str(metadata.get("slot_id") or ""),
            canonical_value=str(metadata.get("canonical_value") or ""),
            profile=TransitionProfile(str(metadata.get("transition_profile") or "")),
            revisions=revisions,
            identity_algorithm_version=str(metadata.get("identity_algorithm_version") or ""),
            canonical_subject_key=str(metadata.get("canonical_subject") or ""),
        )
        if claim.slot_id != slot.slot_id or claim.uri != f"{slot.uri}/claims/{claim.claim_id}":
            raise IdentityValidationError("canonical Claim path does not match its final Slot")
        if claim.profile != profile_for(slot.memory_type):
            raise FinalStateValidationError("canonical Claim transition profile disagrees with its memory type")
        if int(metadata.get("revision", 0)) != claim.latest_revision.revision:
            raise RevisionPrefixError("canonical Claim revision pointer does not match its history")
        if int(metadata.get("current_revision", claim.current.revision)) != claim.current.revision:
            raise RevisionPrefixError("canonical Claim current revision pointer is inconsistent")
        return claim

    def _validate_slot_materialization(
        self,
        operation: ContextOperation,
        obj: ContextObject,
        slot: MemorySlot,
    ) -> None:
        metadata = dict(obj.metadata or {})
        if operation.target_uri != obj.uri:
            raise OperationCompletenessError("canonical Slot operation target does not match its object")
        expected_content = json.dumps(
            {
                "slot_id": slot.slot_id,
                "identity_algorithm_version": slot.identity_algorithm_version,
                "canonical_subject": slot.canonical_subject_key,
                "identity_fields": dict(slot.identity_fields),
                "claim_ids": list(slot.claim_ids),
                "active_claim_id": slot.active_claim_id,
                "revision": slot.revision,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if operation.payload.get("content") != expected_content:
            raise FinalStateValidationError("canonical Slot content is not its materialized final state")
        if (
            metadata.get("projection_pending") is not False
            or obj.schema_version != "canonical_memory_v2"
            or obj.lifecycle_state.value != "active"
        ):
            raise FinalStateValidationError("canonical Slot object mirror is inconsistent")

    def _validate_claim_materialization(
        self,
        operation: ContextOperation,
        obj: ContextObject,
        claim: MemoryClaim,
    ) -> None:
        metadata = dict(obj.metadata or {})
        current = claim.current
        expected_display = dict(current.qualifiers.get("display_fields", {}) or {})
        expected_display_evidence = {
            str(key): [_payload(ref) for ref in refs]
            for key, refs in dict(current.qualifiers.get("display_field_evidence_refs", {}) or {}).items()
        }
        if operation.target_uri != obj.uri:
            raise OperationCompletenessError("canonical Claim operation target does not match its object")
        if (
            metadata.get("state") != current.state
            or metadata.get("epistemic_status") != current.epistemic_status
            or metadata.get("semantic_relation") != current.relation
            or canonical_json(metadata.get("display_fields", {})) != canonical_json(expected_display)
            or canonical_json(metadata.get("display_field_evidence_refs", {}))
            != canonical_json(expected_display_evidence)
            or metadata.get("projection_pending") is not True
            or obj.schema_version != "canonical_memory_v2"
            or obj.lifecycle_state.value != "active"
        ):
            raise FinalStateValidationError("canonical Claim object mirror is inconsistent")
        expected_content = json.dumps(
            {
                "slot_id": claim.slot_id,
                "claim_id": claim.claim_id,
                "canonical_value": claim.canonical_value,
                "current": current.to_dict(),
                "latest_revision": claim.latest_revision.revision,
                "revisions": [revision.to_dict() for revision in claim.revisions],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if operation.payload.get("content") != expected_content:
            raise FinalStateValidationError("canonical Claim content is not its materialized final state")

    def _validate_slot_revision(
        self,
        current: MemorySlot | None,
        desired: MemorySlot,
        operation: ContextOperation,
        claim_operations: dict[str, tuple[ContextOperation, MemoryClaim, ContextObject]],
    ) -> None:
        expected = int(operation.payload.get("expected_revision", 0))
        actual = current.revision if current is not None else 0
        if expected != actual:
            raise RevisionPrefixError(f"Slot expected revision {expected} does not match committed {actual}")
        required_action = OperationAction.ADD if current is None else OperationAction.UPDATE
        if operation.action != required_action:
            raise RevisionPrefixError(
                f"{'new' if current is None else 'existing'} canonical Slot requires {required_action.value.upper()}"
            )
        domain_changed = (
            bool(claim_operations)
            or current is None
            or (
                current.claim_ids != desired.claim_ids
                or current.active_claim_id != desired.active_claim_id
                or dict(current.identity_fields) != dict(desired.identity_fields)
                or current.scope_keys != desired.scope_keys
            )
        )
        required = actual + (1 if domain_changed else 0)
        if desired.revision != required:
            raise RevisionPrefixError(
                f"Slot revision must {'increase by exactly one' if domain_changed else 'remain unchanged'}"
            )
        if not domain_changed:
            raise OperationCompletenessError("canonical no-op must not create a state-changing transaction")

    def _validate_expected_revision_map(
        self,
        operations: list[ContextOperation],
        current_slot: MemorySlot | None,
        current_claims: tuple[MemoryClaim, ...],
        claim_operations: dict[str, tuple[ContextOperation, MemoryClaim, ContextObject]],
    ) -> None:
        expected = {claim.uri: claim.latest_revision.revision for claim in current_claims}
        if current_slot is not None:
            expected[current_slot.uri] = current_slot.revision
        for _operation, claim, _obj in claim_operations.values():
            expected.setdefault(claim.uri, 0)
        if current_slot is None:
            slot_uris = {
                str(operation.target_uri or "")
                for operation in operations
                if isinstance(operation.payload.get("context_object"), dict)
                and dict(operation.payload["context_object"].get("metadata", {}) or {}).get("canonical_kind") == "slot"
            }
            if len(slot_uris) == 1:
                expected[next(iter(slot_uris))] = 0
        normalized = {str(uri): int(revision) for uri, revision in sorted(expected.items())}
        for operation in operations:
            declared = operation.payload.get("expected_revisions")
            if not isinstance(declared, dict):
                raise RevisionPrefixError("canonical transaction has no complete expected revision map")
            try:
                operation_map = {str(uri): int(revision) for uri, revision in sorted(declared.items())}
            except (TypeError, ValueError) as exc:
                raise RevisionPrefixError("canonical expected revision map is invalid") from exc
            if operation_map != normalized:
                raise RevisionPrefixError(
                    "canonical expected revision map does not describe the complete committed Slot state"
                )

    def _validate_claim_revision(
        self,
        current: MemoryClaim | None,
        desired: MemoryClaim,
        operation: ContextOperation,
    ) -> None:
        expected = int(operation.payload.get("expected_revision", 0))
        actual = current.latest_revision.revision if current is not None else 0
        if expected != actual:
            raise RevisionPrefixError(f"Claim expected revision {expected} does not match committed {actual}")
        if current is None:
            if desired.latest_revision.revision != 1 or len(desired.revisions) != 1:
                raise RevisionPrefixError("new canonical Claim must start at revision one")
            if operation.action != OperationAction.ADD:
                raise RevisionPrefixError("new canonical Claim requires ADD")
            return
        if operation.action != OperationAction.UPDATE:
            raise RevisionPrefixError("existing canonical Claim requires UPDATE")
        if len(desired.revisions) != len(current.revisions) + 1:
            raise RevisionPrefixError("Claim UPDATE must append exactly one revision")
        if canonical_json([item.to_dict() for item in desired.revisions[:-1]]) != canonical_json(
            [item.to_dict() for item in current.revisions]
        ):
            raise RevisionPrefixError("Claim UPDATE modified, removed, or reordered historical revisions")
        latest = desired.latest_revision
        if latest.revision != current.latest_revision.revision + 1:
            raise RevisionPrefixError("Claim UPDATE revision must increase by exactly one")
        if latest.previous_revision != current.latest_revision.revision:
            raise RevisionPrefixError("Claim UPDATE previous_revision is inconsistent")
        if not latest.historical_only and canonical_json(
            {
                "state": current.current.state,
                "value_fields": current.current.value_fields,
                "qualifiers": current.current.qualifiers,
            }
        ) == canonical_json(
            {
                "state": desired.current.state,
                "value_fields": desired.current.value_fields,
                "qualifiers": desired.current.qualifiers,
            }
        ):
            raise OperationCompletenessError("canonical no-op Claim UPDATE must not append an empty revision")

    def _validate_identity(
        self,
        slot: MemorySlot,
        claim: MemoryClaim,
        scope_obj: ContextObject,
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> None:
        metadata = dict(scope_obj.metadata or {})
        if (
            slot.identity_algorithm_version != IDENTITY_ALGORITHM_V2
            or claim.identity_algorithm_version != IDENTITY_ALGORITHM_V2
        ):
            raise IdentityValidationError("canonical final state must use Identity V2")
        scope = MemoryScope.from_dict(dict(metadata.get("scope", {}) or {}))
        revision = claim.current
        proposal = MemorySemanticProposal(
            proposal_id=revision.proposal_id or "final-state-validation",
            memory_type=slot.memory_type,
            identity_fields=slot.identity_fields,
            value_fields=revision.value_fields,
            semantic=SemanticAssessment("observation", "weak", "unspecified", "unrelated"),
            epistemic_status=EpistemicStatus(revision.epistemic_status),
            suggested_scope_refs=scope.applicability.all_of,
            related_memory_ids=(),
            evidence_refs=revision.evidence_refs,
            field_evidence_refs=revision.field_evidence_refs,
            confidence=1.0,
            extractor_version=revision.extractor_version or "final-state-validator",
            model_id=revision.model_id,
            prompt_version=revision.prompt_version,
        )
        resolved = self.identity.resolve(
            proposal,
            scope,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        if (
            resolved.slot_id != slot.slot_id
            or resolved.slot_uri != slot.uri
            or resolved.claim_id != claim.claim_id
            or resolved.claim_uri != claim.uri
            or resolved.canonical_value != claim.canonical_value
            or resolved.canonical_subject_key != slot.canonical_subject_key
        ):
            raise IdentityValidationError("canonical operation identity does not recompute under Identity V2")

    def _validate_revision_evidence(
        self,
        current: MemoryClaim | None,
        desired: MemoryClaim,
        operation: ContextOperation,
        *,
        memory_type: str,
        proposal_proofs: dict[str, MemorySemanticProposal],
        transition_relations: dict[str, str],
    ) -> None:
        latest = desired.latest_revision
        operation_evidence = {canonical_json(item) for item in operation.evidence}
        if not latest.proposal_fingerprint:
            raise RevisionEvidenceError("new canonical revision has no proposal fingerprint")
        declared_fingerprints = {str(item) for item in operation.payload.get("proposal_fingerprints", []) or []}
        if latest.proposal_fingerprint not in declared_fingerprints:
            raise RevisionEvidenceError("new canonical revision fingerprint is not bound to its transaction")
        proposal = proposal_proofs.get(latest.proposal_fingerprint)
        if proposal is None:
            raise RevisionEvidenceError("new canonical revision has no matching proposal proof")
        if (
            proposal.proposal_id != latest.proposal_id
            or proposal.memory_type != memory_type
            or canonical_json([ref.to_dict() for ref in proposal.evidence_refs])
            != canonical_json([ref.to_dict() for ref in latest.evidence_refs])
            or proposal.extractor_version != latest.extractor_version
            or proposal.model_id != latest.model_id
            or proposal.prompt_version != latest.prompt_version
            or proposal.epistemic_status.value != latest.epistemic_status
        ):
            raise RevisionEvidenceError("new canonical revision is detached from its proposal proof")
        semantic = proposal.semantic
        if not isinstance(semantic, NormalizedSemanticAssessment):
            raise RevisionEvidenceError("new canonical revision proposal is not semantically normalized")
        if latest.relation != transition_relations.get(latest.proposal_fingerprint):
            raise RevisionEvidenceError("new canonical revision relation disagrees with committed-state reconciliation")
        if (
            latest.policy_version != MemoryTransitionPolicy.VERSION
            or latest.schema_version != "canonical_memory_v2"
            or operation.payload.get("policy_version") != MemoryTransitionPolicy.VERSION
            or operation.payload.get("schema_version") != "canonical_memory_v2"
            or str(operation.payload.get("claim_id") or "") != desired.claim_id
            or str(operation.payload.get("memory_type") or "") != memory_type
        ):
            raise RevisionEvidenceError("new canonical revision policy or transaction mirror is invalid")
        if not operation.payload.get("planning_digest"):
            raise RevisionEvidenceError("new canonical revision has no planning digest binding")
        transition_refs = tuple(latest.field_evidence_refs.get("transition", ()))
        if not transition_refs or any(
            canonical_json(ref.to_dict()) not in operation_evidence for ref in transition_refs
        ):
            raise RevisionEvidenceError("new canonical revision transition evidence is missing")
        previous_values = dict(current.current.value_fields) if current is not None else {}
        previous_evidence = dict(current.current.field_evidence_refs) if current is not None else {}
        current_values = dict(latest.value_fields)
        try:
            merge_rules = self.registry.get(memory_type).field_merge_rules
        except ValueError as exc:
            raise RevisionEvidenceError("new canonical revision uses an unsupported memory schema") from exc
        for field_name in sorted(set(previous_values) | set(current_values)):
            key = f"value.{field_name}"
            changed = canonical_json(previous_values.get(field_name)) != canonical_json(current_values.get(field_name))
            refs = tuple(latest.field_evidence_refs.get(key, ()))
            if changed:
                self._validate_changed_field_evidence(
                    field_name,
                    refs,
                    tuple(previous_evidence.get(key, ())),
                    operation_evidence,
                    mode=merge_rules.get(field_name, FieldMergeMode.REPLACE),
                    display=False,
                )
            elif canonical_json([ref.to_dict() for ref in refs]) != canonical_json(
                [ref.to_dict() for ref in previous_evidence.get(key, ())]
            ):
                raise RevisionEvidenceError(f"unchanged field provenance was rewritten: {field_name}")
        previous_display = (
            dict(current.current.qualifiers.get("display_fields", {}) or {}) if current is not None else {}
        )
        previous_display_evidence = (
            dict(current.current.qualifiers.get("display_field_evidence_refs", {}) or {}) if current is not None else {}
        )
        current_display = dict(latest.qualifiers.get("display_fields", {}) or {})
        current_display_evidence = dict(latest.qualifiers.get("display_field_evidence_refs", {}) or {})
        for field_name in sorted(set(previous_display) | set(current_display)):
            key = f"value.{field_name}"
            changed = canonical_json(previous_display.get(field_name)) != canonical_json(
                current_display.get(field_name)
            )
            refs = tuple(current_display_evidence.get(key, ()))
            if changed:
                self._validate_changed_field_evidence(
                    field_name,
                    refs,
                    tuple(previous_display_evidence.get(key, ())),
                    operation_evidence,
                    mode=merge_rules.get(field_name, FieldMergeMode.REPLACE),
                    display=True,
                )
            elif canonical_json(refs) != canonical_json(previous_display_evidence.get(key, ())):
                raise RevisionEvidenceError(f"unchanged display field provenance was rewritten: {field_name}")

    def _validate_changed_field_evidence(
        self,
        field_name: str,
        refs: tuple[object, ...],
        previous_refs: tuple[object, ...],
        operation_evidence: set[str],
        *,
        mode: FieldMergeMode,
        display: bool,
    ) -> None:
        label = "display field" if display else "field"
        normalized = [canonical_json(_payload(ref)) for ref in refs]
        previous = [canonical_json(_payload(ref)) for ref in previous_refs]
        if not normalized:
            raise RevisionEvidenceError(f"changed {label} has no current evidence: {field_name}")
        if mode != FieldMergeMode.APPEND_UNIQUE:
            if any(ref not in operation_evidence for ref in normalized):
                raise RevisionEvidenceError(f"changed {label} has no current evidence: {field_name}")
            return

        # APPEND_UNIQUE is the sole merge mode whose complete materialized
        # provenance intentionally contains both historical and current
        # evidence.  Historical refs must remain in their stable prefix, every
        # other ref must belong to this operation, and the operation must
        # contribute evidence for the appended value.
        stable_previous = list(dict.fromkeys(previous))
        if normalized[: len(stable_previous)] != stable_previous:
            raise RevisionEvidenceError(f"APPEND_UNIQUE {label} rewrote historical provenance: {field_name}")
        if any(ref not in operation_evidence and ref not in stable_previous for ref in normalized) or not any(
            ref in operation_evidence for ref in normalized
        ):
            raise RevisionEvidenceError(f"changed {label} has no current evidence: {field_name}")

    def _validate_relation_membership(
        self,
        slot: MemorySlot,
        final_by_id: dict[str, MemoryClaim],
        *,
        tenant_id: str,
    ) -> None:
        if self.relation_store is None:
            return
        related_claim_uris = {
            relation.target_uri
            for relation in self.relation_store.relations_of(
                slot.uri,
                tenant_id=tenant_id,
            )
            if relation.source_uri == slot.uri and relation.relation_type == "has_claim"
        }
        allowed = {claim.uri for claim in final_by_id.values() if claim.claim_id in slot.claim_ids}
        # Existing stale derived edges cannot authorize an omitted Claim.  The
        # relation manifest may remove them in this transaction, so only reject
        # an edge whose target is also declared by the final Slot domain.
        if any(uri not in allowed for uri in related_claim_uris):
            raise OperationCompletenessError("formal Slot relations disagree with final membership")
