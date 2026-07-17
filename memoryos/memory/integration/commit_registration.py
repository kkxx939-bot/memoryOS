"""Register Memory-owned commit components with the generic operation plane."""

from __future__ import annotations

from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    load_current_head,
    publish_current_head_sets,
)
from memoryos.memory.canonical.final_state import CanonicalFinalStateValidator
from memoryos.memory.canonical.scope import MemoryScope, scope_key_from_payload, scope_keys_from_payloads
from memoryos.memory.canonical.state import materialized_current_revision_payload
from memoryos.memory.canonical.visibility import (
    committed_content,
    committed_relations,
    read_committed_canonical,
)
from memoryos.memory.integration.archive_reader import (
    SessionEvidenceArchiveReaderFactory,
    session_evidence_archive_reader,
)
from memoryos.memory.integration.commit_handler import (
    CanonicalMemoryCommitHandler,
    bind_canonical_commit_domain_classifier,
)
from memoryos.memory.integration.coordinator import CanonicalCommitCoordinator
from memoryos.memory.integration.planning import CanonicalCommitPlanning
from memoryos.memory.integration.planning_envelope import PlanningEnvelopeStore
from memoryos.memory.integration.relation_policy import CanonicalMemoryRelationPolicy
from memoryos.operations.commit.domain_registry import (
    RegisteredMemoryCommitHandlers,
    register_memory_commit_handlers,
)


def build_memory_commit_handlers(
    *,
    session_evidence_reader_factory: SessionEvidenceArchiveReaderFactory | None = None,
) -> RegisteredMemoryCommitHandlers:
    reader_factory = session_evidence_reader_factory or session_evidence_archive_reader
    return RegisteredMemoryCommitHandlers(
        canonical_handler=CanonicalMemoryCommitHandler,
        canonical_coordinator=CanonicalCommitCoordinator,
        canonical_planning=CanonicalCommitPlanning,
        final_state_validator_factory=CanonicalFinalStateValidator,
        planning_envelope_store_factory=lambda root, tenant_id: PlanningEnvelopeStore(
            root,
            tenant_id=tenant_id,
        ),
        session_evidence_reader_factory=reader_factory,
        domain_classifier_binder=bind_canonical_commit_domain_classifier,
        current_head_integrity_error=CurrentHeadIntegrityError,
        load_current_head=load_current_head,
        publish_current_head_sets=publish_current_head_sets,
        read_committed_canonical=read_committed_canonical,
        committed_content=committed_content,
        committed_relations=committed_relations,
        materialized_current_revision_payload=materialized_current_revision_payload,
        memory_scope_from_dict=MemoryScope.from_dict,
        scope_key_from_payload=scope_key_from_payload,
        scope_keys_from_payloads=scope_keys_from_payloads,
        relation_domain_policy_factory=CanonicalMemoryRelationPolicy,
    )


def register_default_memory_commit_handlers() -> RegisteredMemoryCommitHandlers:
    handlers = build_memory_commit_handlers()
    register_memory_commit_handlers(handlers)
    return handlers


__all__ = ["build_memory_commit_handlers", "register_default_memory_commit_handlers"]
