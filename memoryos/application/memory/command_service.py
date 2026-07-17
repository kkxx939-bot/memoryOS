"""Authoritative memory command orchestration."""

from __future__ import annotations

import json
from typing import Any

from memoryos.application.memory.command_support import (
    _explicit_field_evidence,
    _explicit_identity_fields,
    _explicit_retrieval_views,
    _explicit_rule_modal_force,
    _normalize_explicit_memory_type,
    _require_committed_diff,
)
from memoryos.application.service import ApplicationService
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical import (
    IDENTITY_ALGORITHM_V2,
    Atomicity,
    Attribution,
    CanonicalMemoryRepository,
    Durability,
    EpistemicStatus,
    EvidenceRef,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemorySemanticReconciler,
    MemoryTransactionPlanner,
    MemoryTransitionPolicy,
    ModalForce,
    ProposalEvidenceValidator,
    ResolvedMemoryIdentity,
    SemanticAssessment,
    UtteranceMode,
)
from memoryos.memory.canonical.current_head import artifact_root_for, load_current_head
from memoryos.memory.canonical.visibility import read_committed_canonical
from memoryos.memory.schema import MemoryType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.security.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    READ_CONTEXT,
    TrustedRequestContext,
)


class MemoryCommandService(ApplicationService):
    def remember(
        self,
        *,
        user_id: str,
        content: str,
        title: str = "",
        memory_type: str = "project_decision",
        project_id: str = "",
        constraint_polarity: str = "",
        condition: str = "",
        exception: str = "",
        identity_fields: dict[str, Any] | None = None,
        connect_metadata: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """Commit a structured explicit-memory command through the canonical chain."""

        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if caller is not None:
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            if caller.actor_kind != "user":
                raise PermissionError("authoritative remember requires a trusted user actor")
        if not content.strip():
            raise ValueError("content is required")
        normalized_type = _normalize_explicit_memory_type(memory_type)
        if caller is not None:
            if normalized_type in {MemoryType.PROFILE.value, MemoryType.PREFERENCE.value} and not project_id:
                project_id = ""
            else:
                project_id = caller.bind_write_workspace(project_id)
        retrieval_views = _explicit_retrieval_views(normalized_type, user_id=user_id, project_id=project_id)
        connect = self._parse_connect_metadata(connect_metadata).to_dict()
        event_id = "explicit_" + stable_hash(
            [
                user_id,
                project_id,
                normalized_type,
                tenant_id,
                title,
                content,
                identity_fields or {},
                constraint_polarity,
                condition,
                exception,
            ],
            length=32,
        )
        identity_fields = _explicit_identity_fields(
            normalized_type,
            title=title,
            user_id=user_id,
            project_id=project_id,
            event_id=event_id,
            explicit_fields=identity_fields,
        )
        value_fields: dict[str, Any] = {"canonical_value": content}
        modal_force = ModalForce.PREFER
        if normalized_type == MemoryType.PROJECT_RULE.value:
            modal_force = _explicit_rule_modal_force(
                constraint_polarity,
                has_condition=bool(condition.strip() or exception.strip()),
            )
            value_fields["constraint_polarity"] = modal_force.value
            value_fields["rule"] = content
            if condition.strip():
                value_fields["condition"] = condition.strip()
            if exception.strip():
                value_fields["exception"] = exception.strip()
        elif normalized_type != MemoryType.PREFERENCE.value:
            modal_force = ModalForce.NONE
        command_payload = {
            "command": "REMEMBER_CANONICAL_VALUE",
            "memory_type": normalized_type,
            "identity_fields": identity_fields,
            "value_fields": value_fields,
        }
        command_text = json.dumps(command_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        archive_uri = f"memoryos://user/{user_id}/sessions/history/{event_id}"
        archive = SessionArchive(
            user_id=user_id,
            session_id=event_id,
            archive_uri=archive_uri,
            messages=[
                {
                    "id": event_id,
                    "role": "user",
                    "actor_id": user_id,
                    "event_type": "EXPLICIT_MEMORY_COMMAND",
                    "content": command_text,
                }
            ],
            metadata={
                "connect": connect,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "structured_memory_command": True,
            },
        )
        archive = self._persist_structured_command_archive(archive)
        connect = dict(archive.metadata.get("connect", {}) or {})
        planner = self.session_commit_service.memory_planner
        episode = planner.episode_adapter.adapt(archive)
        system_fields = tuple(identity_fields)
        suggested_scopes = tuple(
            scope
            for scope in episode.legal_scope_candidates()
            if (
                normalized_type in {MemoryType.PROFILE.value, MemoryType.PREFERENCE.value} and scope.kind == "principal"
            )
            or (
                normalized_type not in {MemoryType.PROFILE.value, MemoryType.PREFERENCE.value}
                and scope.kind == ("workspace" if project_id else "principal")
            )
        )
        event_text = episode.events[0].text()
        evidence_refs = (
            EvidenceRef.from_event(
                episode.events[0],
                source_uri=archive.archive_uri,
                span_start=0,
                span_end=len(event_text),
            ),
        )
        proposal = MemorySemanticProposal(
            proposal_id=f"proposal_{event_id}",
            memory_type=normalized_type,
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(
                "confirmation",
                "confirmed",
                "current",
                "unrelated",
                UtteranceMode.ASSERTION.value,
                Attribution.SOURCE_ACTOR.value,
                Durability.DURABLE.value,
                modal_force.value,
                Atomicity.ATOMIC.value,
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=suggested_scopes,
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=_explicit_field_evidence(identity_fields, value_fields, evidence_refs),
            confidence=1.0,
            extractor_version="explicit_remember_v3",
            prompt_version="explicit_remember_v3",
            semantic_contract_version="v3",
            atomic_evidence_ref=evidence_refs[0],
            metadata={
                "source_role": "user",
                "source_adapter_id": str(connect.get("adapter_id", "")),
                "source_session_id": event_id,
                "system_identity_fields": system_fields,
                "effect_authority": "structured_explicit_command",
            },
        )
        formed = planner.formation.plan(
            proposal,
            archive=archive,
            episode=episode,
            retrieval_views=retrieval_views,
        )
        operations = list(formed.operations)
        if formed.decision.value == "PENDING":
            diff = self.committer.commit(user_id, operations) if operations else None
            pending_uri = formed.pending_uri or next(
                (
                    str(operation.target_uri)
                    for operation in operations
                    if operation.payload.get("canonical_pending_proposal") is True
                ),
                "",
            )
            if not pending_uri:
                raise RuntimeError("pending formation did not identify its durable proposal")
            lifecycle_state = (formed.pending_lifecycle_state or "PENDING").upper()
            pending_outstanding = lifecycle_state in {"PENDING", "CONFIRMED", "RETRYABLE"}
            return {
                "uri": pending_uri,
                "status": lifecycle_state,
                "lifecycle_revision": formed.pending_lifecycle_revision or 1,
                "diff_id": diff.diff_id if diff is not None else "",
                "pending_count": 1 if pending_outstanding else 0,
                "pending_persisted": pending_outstanding,
                "proposal_record_persisted": True,
                "canonical_active_operation_count": 0,
            }
        if formed.decision.value != "ACCEPT_FOR_RECONCILE":
            raise ValueError(f"explicit memory was not admitted: {formed.reason}")
        if not operations:
            # A semantic no-op may be a retry of an authoritative commit whose
            # derived CurrentSlot publication previously failed.  Drain that
            # exact durable outbox before reporting an idempotent success.
            self._process_memory_projections_or_raise()
            identity = formed.resolved_identity
            if identity is None:
                raise RuntimeError("canonical no-op has no resolved Identity V2 proof")
            _slot, existing_claims = CanonicalMemoryRepository(
                self.source_store,
                self.relation_store,
            ).load(identity)
            existing_claim = next(
                (
                    claim
                    for claim in existing_claims
                    if claim.claim_id == identity.claim_id and claim.current.state == "ACTIVE"
                ),
                None,
            )
            if existing_claim is None:
                raise RuntimeError("canonical no-op does not resolve to an exact committed ACTIVE Claim")
            artifact_root = artifact_root_for(self.source_store)
            if artifact_root is None:
                raise RuntimeError("canonical no-op has no tenant artifact root")
            head, receipt, _snapshot = load_current_head(
                artifact_root,
                existing_claim.uri,
                canonical_kind="claim",
            )
            return {
                "uri": existing_claim.uri,
                "status": "COMMITTED",
                "diff_id": str(dict(receipt.get("diff", {}) or {}).get("diff_id") or ""),
                "transaction_id": str(head["current_transaction_id"]),
                "receipt_digest": str(head["receipt_digest"]),
                "idempotent_replay": True,
            }
        diff = self.committer.commit(user_id, operations)
        self._process_memory_projections_or_raise()
        uri = next(
            str(operation.target_uri)
            for operation in operations
            if dict(operation.payload.get("context_object", {}).get("metadata", {}) or {}).get("canonical_kind")
            == "claim"
        )
        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is None:
            raise RuntimeError("canonical commit has no tenant artifact root")
        head, _receipt, _snapshot = load_current_head(
            artifact_root,
            uri,
            canonical_kind="claim",
        )
        return {
            "uri": uri,
            "status": "COMMITTED",
            "diff_id": diff.diff_id,
            "transaction_id": str(head["current_transaction_id"]),
            "receipt_digest": str(head["receipt_digest"]),
            "idempotent_replay": False,
        }

    def forget(
        self,
        *,
        user_id: str,
        uri: str,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """撤回或软删除自己拥有的记忆，同时保留审计信息。"""

        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if caller is not None:
            caller.require(AUTHORITATIVE_FORGET)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
        parsed = ContextURI.parse(uri)
        if "/memories/canonical/" in uri or "/memories/pending/" in uri:
            obj = read_committed_canonical(self.source_store, uri, self.relation_store).object
        else:
            obj = self.context_db.read_object(uri)
            if dict(obj.metadata or {}).get("canonical_kind") in {"slot", "claim", "pending_proposal"}:
                obj = read_committed_canonical(self.source_store, uri, self.relation_store).object
        metadata = dict(obj.metadata or {})
        if caller is not None:
            self._require_exact_workspace(metadata, caller, uri)
        scope = dict(metadata.get("scope", {}) or {})
        authority = dict(scope.get("authority", {}) or {})
        authority_principals = {str(item) for item in authority.get("principal_ids", []) or []}
        if str(obj.tenant_id or "default") != tenant_id:
            raise PermissionError("forget tenant does not match trusted identity")
        if (
            obj.owner_user_id != user_id
            and metadata.get("asserted_by") != user_id
            and user_id not in authority_principals
        ):
            raise PermissionError("forget requires an exact URI owned by user_id")
        canonical_kind = str(metadata.get("canonical_kind") or "")
        if parsed.authority != "user" or (
            parsed.user_id != user_id and not (canonical_kind == "claim" and obj.owner_user_id == user_id)
        ):
            raise PermissionError("forget URI owner does not match user_id")
        if obj.metadata.get("canonical_kind") == "claim":
            return self._forget_canonical_claim(user_id, obj)
        operation = ContextOperation(
            user_id=user_id,
            context_type=obj.context_type,
            action=OperationAction.DELETE,
            target_uri=uri,
            payload={"reason": "explicit_forget"},
            evidence=[{"source": "explicit_forget"}],
        )
        diff = self.context_db.commit_operation(operation)
        _require_committed_diff(diff, {operation.operation_id})
        committed_operation = next(item for item in diff.operations if item.operation_id == operation.operation_id)
        raw_tombstone_ids = committed_operation.payload.get("projection_tombstone_ids", ())
        if not isinstance(raw_tombstone_ids, list | tuple):
            raise RuntimeError("committed DELETE has an invalid durable tombstone binding")
        tombstone_ids = tuple(str(item) for item in raw_tombstone_ids if str(item))
        if callable(getattr(self.index_store, "enqueue_tombstone", None)) and not tombstone_ids:
            raise RuntimeError("committed production DELETE has no durable tombstone binding")
        return {
            "uri": uri,
            "status": "COMMITTED",
            "lifecycle_state": LifecycleState.DELETED.value,
            "diff_id": diff.diff_id,
            "tombstone_ids": list(tombstone_ids),
        }

    def list_pending(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        lifecycle_states: list[str] | None = None,
        project_id: str = "",
        caller: TrustedRequestContext | None = None,
    ) -> list[dict[str, Any]]:
        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            project_id = caller.bind_read_workspace(project_id)
        records = CanonicalMemoryRepository(self.source_store, self.relation_store).list_pending(
            tenant_id=tenant_id,
            owner_user_id=user_id,
            lifecycle_states=tuple(lifecycle_states or ()),
        )
        visible = []
        for record in records:
            metadata = {"scope": record.scope.to_dict()}
            if self._workspace_matches(metadata, project_id, caller):
                visible.append({"uri": record.uri, **record.to_payload()})
        return visible

    def _forget_canonical_claim(self, user_id: str, obj) -> dict[str, Any]:  # noqa: ANN001
        metadata = dict(obj.metadata or {})
        slot_uri = obj.uri.rsplit("/claims/", 1)[0]
        slot_obj = read_committed_canonical(
            self.source_store,
            slot_uri,
            self.relation_store,
        ).object
        slot_metadata = dict(slot_obj.metadata or {})
        memory_scope = MemoryScope.from_dict(dict(metadata.get("scope", {}) or {}))
        canonical_subject = memory_scope.canonical_subject
        if canonical_subject is None:
            raise ValueError("Identity V2 canonical memory is missing its subject")
        identity = ResolvedMemoryIdentity(
            slot_id=str(metadata["slot_id"]),
            slot_uri=slot_uri,
            claim_id=str(metadata["claim_id"]),
            claim_uri=obj.uri,
            slot_identity=dict(slot_metadata.get("identity_fields", {}) or {}),
            canonical_value=str(metadata["canonical_value"]),
            scope_keys=tuple(str(item) for item in slot_metadata.get("scope_keys", []) or []),
            identity_algorithm_version=str(metadata.get("identity_algorithm_version") or IDENTITY_ALGORITHM_V2),
            canonical_subject=canonical_subject,
        )
        slot, claims = CanonicalMemoryRepository(self.source_store, self.relation_store).load(identity)
        if slot is None:
            raise FileNotFoundError(slot_uri)
        claim = next(item for item in claims if item.claim_id == identity.claim_id)
        state = "RETRACTED"
        if claim.current.state == state:
            return {"uri": obj.uri, "status": "COMMITTED", "memory_state": state, "diff_id": ""}
        event_id = f"forget:{stable_hash([user_id, obj.uri, claim.latest_revision.revision], length=24)}"
        command_payload = {
            "command": "RETRACT_CANONICAL_CLAIM",
            "claim_id": claim.claim_id,
            "claim_uri": obj.uri,
            "memory_type": str(metadata["memory_type"]),
        }
        command_text = json.dumps(command_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        archive = SessionArchive(
            user_id=user_id,
            session_id=event_id,
            archive_uri=f"memoryos://user/{user_id}/sessions/history/{event_id}",
            messages=[
                {
                    "id": event_id,
                    "role": "user",
                    "event_type": "EXPLICIT_MEMORY_COMMAND",
                    "content": command_text,
                }
            ],
            metadata={
                "tenant_id": str(obj.tenant_id or "default"),
                "structured_memory_command": True,
            },
        )
        archive = self._persist_structured_command_archive(archive)
        episode = self.session_commit_service.memory_planner.episode_adapter.adapt(archive)
        event_text = episode.events[0].text()
        evidence = EvidenceRef.from_event(
            episode.events[0],
            source_uri=archive.archive_uri,
            span_start=0,
            span_end=len(event_text),
        )
        raw_proposal = MemorySemanticProposal(
            proposal_id=event_id,
            memory_type=str(metadata["memory_type"]),
            identity_fields=slot.identity_fields,
            value_fields=claim.current.value_fields,
            semantic=SemanticAssessment(
                "retraction",
                "confirmed",
                "current",
                "corrects",
                UtteranceMode.ASSERTION.value,
                Attribution.SOURCE_ACTOR.value,
                Durability.DURABLE.value,
                ModalForce.NONE.value,
                Atomicity.ATOMIC.value,
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=memory_scope.applicability.all_of,
            related_memory_ids=(claim.claim_id,),
            evidence_refs=(evidence,),
            field_evidence_refs=_explicit_field_evidence(
                slot.identity_fields,
                claim.current.value_fields,
                (evidence,),
            ),
            confidence=1.0,
            extractor_version="explicit_forget_v3",
            prompt_version="explicit_forget_v3",
            semantic_contract_version="v3",
            atomic_evidence_ref=evidence,
            metadata={
                "source_role": "user",
                "source_session_id": event_id,
                "asserted_by": user_id,
                "system_identity_fields": list(slot.identity_fields),
                "system_value_fields": list(claim.current.value_fields),
                "effect_authority": "structured_explicit_command",
            },
        )
        validated = ProposalEvidenceValidator().validate(raw_proposal, episode)
        if not validated.valid:
            raise ValueError(f"explicit forget evidence validation failed: {','.join(validated.errors)}")
        proposal = MemorySemanticNormalizer().normalize(validated.proposal)
        reconciliation = MemorySemanticReconciler().reconcile(
            proposal,
            identity,
            slot=slot,
            claims=claims,
        )
        transition_policy = MemoryTransitionPolicy()
        transition = transition_policy._apply_structured_retraction(
            proposal,
            identity,
            reconciliation,
            authorization_id=event_id,
            owner_user_id=user_id,
            tenant_id=str(obj.tenant_id or "default"),
        )
        plan = MemoryTransactionPlanner().build(
            proposal,
            memory_scope,
            transition,
            tenant_id=str(obj.tenant_id or "default"),
            owner_user_id=user_id,
            episode_id=event_id,
        )
        operations = plan.to_context_operations(
            user_id=user_id,
            tenant_id=str(obj.tenant_id or "default"),
            episode_id=event_id,
        )
        for operation in operations:
            payload = operation.payload.get("context_object")
            if isinstance(payload, dict) and payload.get("uri") == obj.uri:
                payload["relations"] = [relation.to_dict() for relation in obj.relations]
        diff = self.committer.commit(
            user_id,
            operations,
        )
        _require_committed_diff(diff, {operation.operation_id for operation in operations})
        self._process_memory_projections_or_raise()
        return {"uri": obj.uri, "status": "COMMITTED", "memory_state": state, "diff_id": diff.diff_id}

    def _persist_structured_command_archive(self, archive: SessionArchive) -> SessionArchive:
        """Create one immutable evidence archive for a stable structured command id."""

        tenant_id = self.session_archive_store.archive_tenant(archive)
        with PathLock(self.lock_store).acquire(f"structured-command:{tenant_id}:{archive.archive_uri}"):
            if not self.session_archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self.session_archive_store.write_sync_archive(archive)
                return archive
            persisted = self.session_archive_store.read_archive(archive.archive_uri, tenant_id=tenant_id)
            stable_metadata = ("tenant_id", "project_id", "structured_memory_command")
            if (
                persisted.user_id != archive.user_id
                or persisted.session_id != archive.session_id
                or persisted.messages != archive.messages
                or any(persisted.metadata.get(key) != archive.metadata.get(key) for key in stable_metadata)
            ):
                raise ValueError("structured memory command archive identity conflict")
            return persisted



__all__ = ["MemoryCommandService"]
