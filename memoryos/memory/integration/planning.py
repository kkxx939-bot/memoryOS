"""Canonical and pending-memory commit planning owned by memory."""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.errors import RevisionConflictError
from memoryos.core.integrity import canonical_json
from memoryos.memory.canonical.current_head import (
    artifact_root_for,
    load_current_head,
)
from memoryos.memory.canonical.proposal import (
    PENDING_PROPOSAL_TRANSITIONS,
    MemorySemanticProposal,
    PendingMemoryProposal,
)
from memoryos.memory.canonical.review_command import (
    PendingReviewCommandIntegrityError,
    PendingReviewCommandStore,
)
from memoryos.memory.canonical.state import materialized_current_revision_payload
from memoryos.memory.canonical.visibility import (
    read_committed_canonical,
    read_committed_pending,
)
from memoryos.memory.integration.planning_envelope import (
    PlanningEnvelopeIntegrityError,
)
from memoryos.operations.commit.planning_proof import (
    PlanningProofIntegrityError,
)
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class CanonicalCommitPlanning:
    """Validate immutable canonical and pending planning proofs."""

    @staticmethod
    def _ensure_canonical_planning_digest(
        committer,
        operations: list[ContextOperation],
        *,
        publish: bool = True,
    ) -> str:
        declared = {
            str(operation.payload.get("planning_digest") or "")
            for operation in operations
            if operation.payload.get("planning_digest")
        }
        if len(declared) > 1:
            raise ValueError("canonical transaction contains multiple planning digests")
        task_ids = {str(operation.payload.get("planning_task_id") or "") for operation in operations} - {""}
        if len(task_ids) > 1:
            raise ValueError("canonical transaction crosses planning task identities")
        task_id = next(iter(task_ids), "")
        proof_operations = [operation for operation in operations if not committer._canonical_pending_effect(operation)]
        if not proof_operations:
            raise ValueError("canonical transaction has no domain operation proposal proof")
        proof_payloads: dict[str, dict] = {}
        proof_sets: set[str] = set()
        missing_proof_count = 0
        for operation in proof_operations:
            raw_proofs = operation.payload.get("proposal_proofs")
            if (
                not isinstance(raw_proofs, list)
                or not raw_proofs
                or any(not isinstance(item, dict) for item in raw_proofs)
            ):
                missing_proof_count += 1
                continue
            proof_sets.add(canonical_json(raw_proofs))
            for raw in raw_proofs:
                try:
                    proposal = MemorySemanticProposal.from_dict(raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("canonical transaction proposal proof is invalid") from exc
                fingerprint = proposal.fingerprint
                if fingerprint in proof_payloads and canonical_json(proof_payloads[fingerprint]) != canonical_json(raw):
                    raise ValueError("canonical transaction contains conflicting proposal proofs")
                proof_payloads[fingerprint] = raw
            declared_fingerprints = {str(item) for item in operation.payload.get("proposal_fingerprints", []) or []}
            if declared_fingerprints != set(proof_payloads):
                raise ValueError("canonical transaction proposal proof identity is inconsistent")
        if missing_proof_count not in {0, len(proof_operations)}:
            raise ValueError("canonical transaction has a partial proposal proof set")
        if not missing_proof_count and len(proof_sets) != 1:
            raise ValueError("canonical transaction operations disagree on proposal proof")
        envelope_path = committer.planning_envelopes.path(task_id) if task_id else None
        anchor_path = committer.planning_envelopes.anchor_path(task_id) if task_id else None
        if task_id and (
            (envelope_path is not None and (envelope_path.exists() or envelope_path.is_symlink()))
            or (anchor_path is not None and (anchor_path.exists() or anchor_path.is_symlink()))
        ):
            if missing_proof_count:
                raise ValueError("canonical transaction has no complete proposal proof")
            try:
                envelope = committer.planning_envelopes.load_validated_payload(task_id)
            except PlanningEnvelopeIntegrityError as exc:
                raise ValueError("canonical transaction planning envelope is invalid") from exc
            commit_groups = {
                str(operation.payload.get("commit_group_id") or "")
                for operation in operations
                if operation.payload.get("commit_group_id")
            }
            fingerprints = {
                str(value)
                for operation in operations
                for value in operation.payload.get("proposal_fingerprints", []) or []
            }
            envelope_fingerprints = {str(value) for value in envelope.get("proposal_fingerprints", []) or []}
            envelope_proofs = {
                MemorySemanticProposal.from_dict(dict(item.get("proposal", {}) or {})).fingerprint: dict(
                    item.get("proposal", {}) or {}
                )
                for item in envelope.get("proposal_inputs", []) or []
                if isinstance(item, dict) and isinstance(item.get("proposal"), dict)
            }
            digest = str(envelope["planning_digest"])
            if (
                len(commit_groups) > 1
                or (commit_groups and commit_groups != {str(envelope.get("operation_group_identity") or "")})
                or not fingerprints.issubset(envelope_fingerprints)
                or set(proof_payloads) != fingerprints
                or any(
                    fingerprint not in envelope_proofs
                    or canonical_json(payload) != canonical_json(envelope_proofs[fingerprint])
                    for fingerprint, payload in proof_payloads.items()
                )
                or (declared and declared != {digest})
            ):
                raise ValueError("canonical transaction is detached from its durable planning envelope")
        else:
            transaction_ids = {str(operation.payload.get("transaction_id") or "") for operation in operations}
            idempotency_keys = {str(operation.payload.get("idempotency_key") or "") for operation in operations}
            commit_groups = {
                str(operation.payload.get("commit_group_id") or "")
                for operation in operations
                if operation.payload.get("commit_group_id")
            }
            if (
                len(transaction_ids) != 1
                or "" in transaction_ids
                or len(idempotency_keys) != 1
                or "" in idempotency_keys
            ):
                raise ValueError("direct canonical plan has invalid transaction identity")
            transaction_id = next(iter(transaction_ids))
            idempotency_key = next(iter(idempotency_keys))
            marker_path = committer._transaction_marker(idempotency_key)
            committer._reject_control_symlink(marker_path, "canonical transaction receipt")
            try:
                if marker_path.exists():
                    proof = committer.planning_proofs.load_direct(
                        transaction_id,
                        operations=operations,
                    )
                elif publish:
                    proof = committer.planning_proofs.ensure_direct(
                        operations,
                        kind="canonical",
                        transaction_id=transaction_id,
                        idempotency_key=idempotency_key,
                        user_id=operations[0].user_id,
                        commit_group_id=next(iter(commit_groups), ""),
                    )
                else:
                    proof = committer.planning_proofs.build_direct(
                        operations,
                        kind="canonical",
                        transaction_id=transaction_id,
                        idempotency_key=idempotency_key,
                        user_id=operations[0].user_id,
                        commit_group_id=next(iter(commit_groups), ""),
                    )
            except PlanningProofIntegrityError as exc:
                raise ValueError("canonical transaction has no valid immutable planning proof") from exc
            digest = str(proof["planning_digest"])
        for operation in operations:
            operation.payload["planning_digest"] = digest
        return digest

    @staticmethod
    def _ensure_pending_planning_digest(committer, operation: ContextOperation) -> str:
        task_id = str(operation.payload.get("planning_task_id") or "")
        if task_id and (
            committer.planning_envelopes.path(task_id).exists()
            or committer.planning_envelopes.path(task_id).is_symlink()
            or committer.planning_envelopes.anchor_path(task_id).exists()
            or committer.planning_envelopes.anchor_path(task_id).is_symlink()
        ):
            try:
                envelope = committer.planning_envelopes.load_validated_payload(task_id)
            except PlanningEnvelopeIntegrityError as exc:
                raise ValueError("pending lifecycle planning envelope is invalid") from exc
            proposal_id = str(operation.payload.get("pending_proposal_id") or "")
            envelope_proposal_ids = {
                str(dict(item.get("proposal", {}) or {}).get("proposal_id") or "")
                for item in envelope.get("proposal_inputs", []) or []
                if isinstance(item, dict)
            }
            digest = str(envelope["planning_digest"])
            declared = str(operation.payload.get("planning_digest") or "")
            if (
                str(operation.payload.get("commit_group_id") or "")
                != str(envelope.get("operation_group_identity") or "")
                or (proposal_id and proposal_id not in envelope_proposal_ids)
                or (declared and declared != digest)
            ):
                raise ValueError("pending lifecycle is detached from its durable planning envelope")
        else:
            marker_path = committer._operation_marker(operation.operation_id)
            committer._reject_control_symlink(marker_path, "pending operation receipt")
            try:
                if marker_path.exists():
                    proof = committer.planning_proofs.load_direct(
                        operation.operation_id,
                        operations=[operation],
                    )
                else:
                    proof = committer.planning_proofs.ensure_direct(
                        [operation],
                        kind="pending",
                        transaction_id=operation.operation_id,
                        idempotency_key=str(operation.payload.get("idempotency_key") or operation.operation_id),
                        user_id=operation.user_id,
                        commit_group_id=str(operation.payload.get("commit_group_id") or ""),
                    )
            except PlanningProofIntegrityError as exc:
                raise ValueError("pending lifecycle has no valid immutable planning proof") from exc
            digest = str(proof["planning_digest"])
        operation.payload["planning_digest"] = digest
        return digest

    @staticmethod
    def _validate_pending_lifecycle_cas(
        committer,
        operation: ContextOperation,
        *,
        validate_resolution_links: bool = True,
    ) -> None:
        if operation.payload.get("pending_lifecycle_transition") is not True:
            return
        if operation.action != OperationAction.UPDATE or operation.context_type != ContextType.MEMORY:
            raise ValueError("pending lifecycle transition must be a memory UPDATE")
        target_uri = str(operation.target_uri or "")
        if not target_uri:
            raise ValueError("pending lifecycle transition requires a target URI")
        committed_pending = read_committed_pending(
            committer.source_store,
            target_uri,
            committer.relation_store,
        )
        current_obj = committed_pending.object
        current = PendingMemoryProposal.from_context_object(current_obj)
        expected_state = str(operation.payload.get("expected_pending_lifecycle_state") or "")
        expected_revision = int(operation.payload.get("expected_pending_lifecycle_revision", 0) or 0)
        expected_updated_at = str(operation.payload.get("expected_pending_updated_at") or "")
        if (
            not expected_state
            or expected_revision < 1
            or not expected_updated_at
            or current.lifecycle_state.value != expected_state
            or current.lifecycle_revision != expected_revision
            or current.updated_at != expected_updated_at
        ):
            raise RevisionConflictError(
                "pending proposal lifecycle conflict: "
                f"expected {expected_state}@{expected_revision}, "
                f"actual {current.lifecycle_state.value}@{current.lifecycle_revision}"
            )
        desired_payload = operation.payload.get("context_object")
        if not isinstance(desired_payload, dict):
            raise ValueError("pending lifecycle transition requires context_object")
        desired_obj = ContextObject.from_dict(desired_payload)
        desired = PendingMemoryProposal.from_context_object(desired_obj)
        decision_for_state = {
            LifecycleState.CONFIRMED: "CONFIRM",
            LifecycleState.RESOLVED: "CONFIRM_AND_APPLY",
            LifecycleState.RETRYABLE: "RETRY",
            LifecycleState.REJECTED: "REJECT",
            LifecycleState.EXPIRED: "EXPIRE",
        }.get(desired.lifecycle_state)
        if decision_for_state is not None:
            current.assert_review_decision(decision_for_state)
        if (
            current_obj.uri != target_uri
            or current_obj.context_type != ContextType.MEMORY
            or current_obj.owner_user_id != operation.user_id
            or desired_obj.owner_user_id != current_obj.owner_user_id
            or str(desired_obj.tenant_id or "default") != str(current_obj.tenant_id or "default")
            or desired_obj.context_type != current_obj.context_type
        ):
            raise ValueError("pending lifecycle transition cannot change owner, tenant, URI, or context type")
        if (
            current_obj.lifecycle_state != current.lifecycle_state
            or desired_obj.lifecycle_state != desired.lifecycle_state
        ):
            raise ValueError("pending lifecycle object and payload state disagree")
        expected_current_obj = current.to_context_object(
            tenant_id=str(current_obj.tenant_id or "default"),
            owner_user_id=str(current_obj.owner_user_id or ""),
        )
        expected_desired_obj = desired.to_context_object(
            tenant_id=str(current_obj.tenant_id or "default"),
            owner_user_id=str(current_obj.owner_user_id or ""),
        )
        if canonical_json(current_obj.to_dict()) != canonical_json(expected_current_obj.to_dict()):
            raise ValueError("stored pending proposal object is internally inconsistent")
        if canonical_json(desired_obj.to_dict()) != canonical_json(expected_desired_obj.to_dict()):
            raise ValueError("pending lifecycle context_object is internally inconsistent")
        try:
            current_content = (
                committed_pending.content_override
                if committed_pending.content_override is not None
                else committer.source_store.read_content(current_obj.layers.l2_uri or current_obj.uri)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
            raise ValueError("stored pending proposal content is missing") from exc
        if current_content != current.content():
            raise ValueError("stored pending proposal content does not match its object")
        if operation.payload.get("content") != desired.content():
            raise ValueError("pending lifecycle content does not match its desired object")
        if desired.uri != current.uri or desired.lifecycle_revision != current.lifecycle_revision + 1:
            raise ValueError("pending lifecycle transition must advance exactly one lifecycle revision")
        mutable_fields = {
            "lifecycle_state",
            "retry_count",
            "lifecycle_revision",
            "lifecycle_history",
            "updated_at",
        }
        current_core = {key: value for key, value in current.to_payload().items() if key not in mutable_fields}
        desired_core = {key: value for key, value in desired.to_payload().items() if key not in mutable_fields}
        if canonical_json(current_core) != canonical_json(desired_core):
            raise ValueError("pending lifecycle transition cannot rewrite proposal content or scope")

        retry_delta = desired.retry_count - current.retry_count
        if retry_delta not in {0, 1}:
            raise ValueError("pending lifecycle retry count must stay stable or increment once")
        if desired.lifecycle_state == current.lifecycle_state:
            if desired.lifecycle_state != LifecycleState.RETRYABLE or retry_delta != 1:
                raise ValueError("pending lifecycle transition cannot silently retain the current state")
        elif desired.lifecycle_state not in PENDING_PROPOSAL_TRANSITIONS.get(current.lifecycle_state, frozenset()):
            raise ValueError(
                "illegal pending proposal lifecycle transition: "
                f"{current.lifecycle_state.value}->{desired.lifecycle_state.value}"
            )
        if len(desired.lifecycle_history) != len(current.lifecycle_history) + 1 or canonical_json(
            desired.lifecycle_history[:-1]
        ) != canonical_json(current.lifecycle_history):
            raise ValueError("pending lifecycle history must append exactly one transition")
        expected_history = {
            "from": current.lifecycle_state.value,
            "to": desired.lifecycle_state.value,
            "from_revision": current.lifecycle_revision,
            "to_revision": desired.lifecycle_revision,
            "reason": str(operation.payload.get("pending_lifecycle_reason") or ""),
            "updated_at": desired.updated_at,
        }
        review_binding = operation.payload.get("pending_review_binding", {})
        if not isinstance(review_binding, dict):
            raise ValueError("pending lifecycle review binding must be an object")
        if decision_for_state is not None:
            committer._validate_pending_review_command(
                operation,
                current,
                review_binding,
            )
        if review_binding:
            if set(review_binding) != {"command_id", "decision", "request_digest"}:
                raise ValueError("pending lifecycle review binding has unexpected fields")
            command_id = str(review_binding.get("command_id") or "")
            decision = str(review_binding.get("decision") or "").strip().upper()
            request_digest = str(review_binding.get("request_digest") or "")
            if (
                not command_id
                or len(request_digest) != 64
                or any(character not in "0123456789abcdef" for character in request_digest)
            ):
                raise ValueError("pending lifecycle review binding is incomplete")
            expected_decisions = {
                LifecycleState.CONFIRMED: {"CONFIRM", "CONFIRM_AND_APPLY"},
                LifecycleState.RESOLVED: {"CONFIRM_AND_APPLY"},
                LifecycleState.RETRYABLE: {"RETRY"},
                LifecycleState.REJECTED: {"REJECT", "CORRECT"},
                LifecycleState.EXPIRED: {"EXPIRE"},
            }.get(desired.lifecycle_state, set())
            if decision not in expected_decisions:
                raise ValueError("pending lifecycle review decision disagrees with its desired state")
            expected_history.update(
                {
                    "review_command_id": command_id,
                    "review_decision": decision,
                    "review_request_digest": request_digest,
                }
            )
        if canonical_json(desired.lifecycle_history[-1]) != canonical_json(expected_history):
            raise ValueError("pending lifecycle history does not match the requested transition")

        expected_fields = {
            "canonical_pending_proposal": True,
            "pending_proposal_id": desired.proposal_id,
            "pending_lifecycle_state": desired.lifecycle_state.value,
            "pending_lifecycle_revision": desired.lifecycle_revision,
            "memory_type": desired.proposal.memory_type,
            "schema_version": PendingMemoryProposal.SCHEMA_VERSION,
            "tenant_id": str(current_obj.tenant_id or "default"),
        }
        if any(operation.payload.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("pending lifecycle operation envelope disagrees with its desired proposal")
        resolution_flag = operation.payload.get("pending_lifecycle_resolution")
        if not isinstance(resolution_flag, bool) or resolution_flag != (
            desired.lifecycle_state == LifecycleState.RESOLVED
        ):
            raise ValueError("pending lifecycle resolution flag disagrees with the desired state")
        resolution_keys = operation.payload.get("resolution_idempotency_keys", [])
        resolved_claims = operation.payload.get("resolved_claim_uris", [])
        if not isinstance(resolution_keys, list | tuple) or not isinstance(resolved_claims, list | tuple):
            raise ValueError("pending lifecycle resolution links must be lists")
        if not resolution_flag and (resolution_keys or resolved_claims):
            raise ValueError("non-RESOLVED pending transition cannot carry canonical resolution links")
        if resolution_flag and (not resolution_keys or not resolved_claims):
            raise ValueError("RESOLVED pending transition requires canonical resolution links")
        if resolution_flag and validate_resolution_links:
            committer._validate_pending_resolution_commit(operation, current)

    @staticmethod
    def _validate_pending_review_command(
        committer,
        operation: ContextOperation,
        current: PendingMemoryProposal,
        review_binding: dict,
    ) -> None:
        if set(review_binding) != {"command_id", "decision", "request_digest"}:
            raise ValueError("pending lifecycle transition requires a durable review command")
        command_id = str(review_binding.get("command_id") or "")
        decision = str(review_binding.get("decision") or "").strip().upper()
        request_digest = str(review_binding.get("request_digest") or "")
        if not command_id or not decision or not request_digest:
            raise ValueError("pending lifecycle transition requires a durable review command")
        try:
            record = PendingReviewCommandStore(
                committer.root,
                tenant_id=committer.tenant_id,
            ).load(command_id)
        except (OSError, PendingReviewCommandIntegrityError) as exc:
            raise ValueError("pending lifecycle transition has no valid durable review command") from exc
        request = dict(record.get("request", {}) or {})
        historical = [
            dict(item)
            for item in current.lifecycle_history
            if str(dict(item).get("review_command_id") or "") == command_id
        ]
        initial_revision = (
            min(int(item.get("from_revision", 0) or 0) for item in historical)
            if historical
            else current.lifecycle_revision
        )
        if (
            record.get("status") == "failed"
            or record.get("request_digest") != request_digest
            or request.get("tenant_id") != committer.tenant_id
            or request.get("owner_user_id") != operation.user_id
            or request.get("pending_uri") != current.uri
            or str(request.get("decision") or "").strip().upper() != decision
            or request.get("expected_proposal_fingerprint") != current.proposal.fingerprint
            or int(request.get("expected_lifecycle_revision", 0) or 0) != initial_revision
        ):
            raise ValueError("pending lifecycle transition conflicts with its durable review command")

    @staticmethod
    def _validate_pending_resolution_commit(
        committer,
        operation: ContextOperation,
        pending: PendingMemoryProposal,
    ) -> None:
        keys = tuple(
            dict.fromkeys(str(item) for item in operation.payload.get("resolution_idempotency_keys", []) or [] if item)
        )
        claim_uris = tuple(
            dict.fromkeys(str(item) for item in operation.payload.get("resolved_claim_uris", []) or [] if item)
        )
        if not keys or not claim_uris:
            raise ValueError("RESOLVED pending transition requires committed canonical Claim links")
        committed_claims_by_key: dict[str, set[str]] = {}
        for key in keys:
            marker = committer._transaction_marker(key)
            committer._reject_control_symlink(marker, "canonical transaction receipt")
            if not marker.exists():
                raise RevisionConflictError("pending proposal cannot resolve before its canonical transaction commits")
            committer._validate_transaction_marker_tenant(marker)
            diff = committer._transaction_marker_diff(marker)
            committer._validate_and_bind_operations(operation.user_id, diff.operations)
            committed_claims_by_key[key] = {
                str(payload.get("uri"))
                for marker_operation in diff.operations
                if marker_operation.payload.get("idempotency_key") == key
                and isinstance((payload := marker_operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            }
        operation_tenant = str(operation.payload.get("tenant_id") or "default")
        for uri in claim_uris:
            claim = read_committed_canonical(committer.source_store, uri, committer.relation_store).object
            metadata = dict(claim.metadata or {})
            linked_key = str(metadata.get("canonical_idempotency_key") or "")
            if (
                claim.lifecycle_state != LifecycleState.ACTIVE
                or metadata.get("canonical_kind") != "claim"
                or metadata.get("state") != "ACTIVE"
                or claim.owner_user_id != operation.user_id
                or str(claim.tenant_id or "default") != operation_tenant
                or str(metadata.get("memory_type") or "") != pending.proposal.memory_type
                or linked_key not in keys
                or uri not in committed_claims_by_key.get(linked_key, set())
            ):
                raise RevisionConflictError(
                    "pending proposal resolution Claim is not the linked committed ACTIVE Claim"
                )

    @staticmethod
    def _validate_pending_resolution_batch(committer, operations: list[ContextOperation]) -> None:
        resolutions = [
            operation for operation in operations if operation.payload.get("canonical_pending_resolution") is True
        ]
        if not resolutions:
            return
        if len(resolutions) != 1:
            raise ValueError("canonical transaction can resolve exactly one pending proposal")
        resolution = resolutions[0]
        artifact_root = artifact_root_for(committer.source_store)
        if artifact_root is None or not resolution.target_uri:
            raise ValueError("pending resolution has no current-head artifact root")
        confirmation_head, _confirmation_receipt, _confirmation_snapshot = load_current_head(
            artifact_root,
            resolution.target_uri,
            canonical_kind="pending_proposal",
        )
        if (
            resolution.payload.get("confirmation_receipt_digest") != confirmation_head.get("receipt_digest")
            or resolution.payload.get("confirmation_operation_id") != confirmation_head.get("current_operation_id")
            or int(resolution.payload.get("confirmation_lifecycle_revision", 0))
            != int(confirmation_head.get("current_revision", 0))
            or str(confirmation_head.get("current_lifecycle_state") or "").upper() != "CONFIRMED"
        ):
            raise ValueError("pending resolution is not bound to its current CONFIRM receipt")
        keys = {str(item) for item in resolution.payload.get("resolution_idempotency_keys", []) or [] if item}
        transaction_keys = {str(operation.payload.get("idempotency_key") or "") for operation in operations}
        claim_uris = {str(item) for item in resolution.payload.get("resolved_claim_uris", []) or [] if item}
        active_claims = {
            str(payload.get("uri") or ""): dict(payload.get("metadata", {}) or {})
            for operation in operations
            if operation is not resolution
            and isinstance((payload := operation.payload.get("context_object")), dict)
            and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
        }
        if (
            len(transaction_keys) != 1
            or "" in transaction_keys
            or keys != transaction_keys
            or not claim_uris
            or not claim_uris.issubset(active_claims)
        ):
            raise ValueError("pending resolution must link ACTIVE Claims in the same canonical transaction")
        resolution_tenant = str(resolution.payload.get("tenant_id") or "default")
        resolution_memory_type = str(resolution.payload.get("memory_type") or "")
        for uri in claim_uris:
            claim_payload = next(
                operation.payload["context_object"]
                for operation in operations
                if isinstance(operation.payload.get("context_object"), dict)
                and str(operation.payload["context_object"].get("uri") or "") == uri
            )
            metadata = active_claims[uri]
            if (
                str(claim_payload.get("owner_user_id") or "") != resolution.user_id
                or str(claim_payload.get("tenant_id") or "default") != resolution_tenant
                or str(metadata.get("memory_type") or "") != resolution_memory_type
            ):
                raise ValueError("pending resolution Claim crosses owner, tenant, or memory type")

    @staticmethod
    def _validate_pending_correction_batch(committer, operations: list[ContextOperation]) -> None:
        corrections = [
            operation for operation in operations if operation.payload.get("canonical_pending_correction") is True
        ]
        if not corrections:
            return
        if len(corrections) != 1:
            raise ValueError("canonical transaction can correct exactly one pending proposal")
        correction = corrections[0]
        if correction.payload.get("canonical_pending_resolution") is True:
            raise ValueError("pending correction cannot also be a confirmation resolution")
        desired_payload = correction.payload.get("context_object")
        if not isinstance(desired_payload, dict):
            raise ValueError("pending correction requires a terminal pending object")
        desired = PendingMemoryProposal.from_context_object(ContextObject.from_dict(desired_payload))
        if desired.lifecycle_state != LifecycleState.REJECTED:
            raise ValueError("a corrected predecessor pending must become REJECTED")
        committed = read_committed_pending(
            committer.source_store,
            str(correction.target_uri or ""),
            committer.relation_store,
        )
        predecessor = PendingMemoryProposal.from_context_object(committed.object)
        if not predecessor.reason_policy.requires_new_proposal:
            raise ValueError("only a non-reviewable pending reason can use correction")
        predecessor_fingerprint = str(correction.payload.get("predecessor_proposal_fingerprint") or "")
        corrected_fingerprint = str(correction.payload.get("corrected_proposal_fingerprint") or "")
        corrected_proposal_id = str(correction.payload.get("corrected_proposal_id") or "")
        correction_task_id = str(correction.payload.get("correction_task_id") or "")
        if (
            predecessor_fingerprint != predecessor.proposal.fingerprint
            or not corrected_fingerprint
            or corrected_fingerprint == predecessor_fingerprint
            or not corrected_proposal_id
            or not correction_task_id
        ):
            raise ValueError("pending correction proposal identity is incomplete or unchanged")
        if bool(correction.payload.get("correction_requires_reextraction")) != bool(
            predecessor.reason_policy.requires_reextraction
        ):
            raise ValueError("pending correction re-extraction proof disagrees with its reason policy")
        if predecessor.reason_policy.requires_reextraction and correction_task_id == predecessor.request_identity:
            raise ValueError("fallback correction reused the predecessor extraction task")

        claim_uris = {str(item) for item in correction.payload.get("corrected_claim_uris", []) or [] if item}
        active_claims: dict[str, dict] = {}
        for operation in operations:
            if operation is correction:
                continue
            raw = operation.payload.get("context_object")
            if not isinstance(raw, dict):
                continue
            metadata = dict(raw.get("metadata", {}) or {})
            if metadata.get("canonical_kind") != "claim" or metadata.get("state") != "ACTIVE":
                continue
            current_revision = materialized_current_revision_payload(metadata)
            qualifiers = dict(current_revision.get("qualifiers", {}) or {})
            if (
                str(current_revision.get("proposal_id") or "") != corrected_proposal_id
                or str(current_revision.get("proposal_fingerprint") or "") != corrected_fingerprint
                or qualifiers.get("corrects_pending_uri") != correction.target_uri
                or qualifiers.get("corrects_pending_fingerprint") != predecessor_fingerprint
                or operation.payload.get("corrects_pending_uri") != correction.target_uri
                or operation.payload.get("corrects_pending_fingerprint") != predecessor_fingerprint
            ):
                raise ValueError("corrected Claim is not bound to its predecessor pending proposal")
            active_claims[str(raw.get("uri") or "")] = raw
        if not claim_uris or not claim_uris.issubset(active_claims):
            raise ValueError("pending correction must link an ACTIVE Claim in the same transaction")
        if any(
            str(payload.get("owner_user_id") or "") != correction.user_id
            or str(payload.get("tenant_id") or "default") != str(correction.payload.get("tenant_id") or "default")
            for payload in (active_claims[uri] for uri in claim_uris)
        ):
            raise ValueError("pending correction Claim crosses owner or tenant boundary")
