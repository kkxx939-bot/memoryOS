from __future__ import annotations

from memoryos.action_policy.integration.commit_handler import ActionPolicyCommitHandler
from memoryos.adapters.persistence.filesystem import FileSystemSourceStore
from memoryos.adapters.persistence.in_memory import InMemoryIndexStore, InMemoryRelationStore
from memoryos.contextdb.session.planning import PlanningContext as LegacyPlanningContext
from memoryos.contextdb.session.planning_envelope import PlanningEnvelopeStore as LegacyPlanningEnvelopeStore
from memoryos.contextdb.transaction.recovery import RecoveryService as LegacyRecoveryService
from memoryos.memory.integration.commit_handler import CanonicalMemoryCommitHandler
from memoryos.memory.integration.planning_context import PlanningContext
from memoryos.memory.integration.planning_envelope import PlanningEnvelopeStore
from memoryos.operations.commit.coordinator import CommitCoordinator
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.recovery import RecoveryService


def test_recovery_and_memory_planning_compatibility_exports_keep_object_identity() -> None:
    assert LegacyRecoveryService is RecoveryService
    assert LegacyPlanningContext is PlanningContext
    assert LegacyPlanningEnvelopeStore is PlanningEnvelopeStore
    assert RecoveryService.__module__ == "memoryos.operations.commit.recovery"
    assert PlanningContext.__module__ == "memoryos.memory.integration.planning_context"
    assert PlanningEnvelopeStore.__module__ == "memoryos.memory.integration.planning_envelope"


def test_committer_uses_explicit_components_without_dynamic_attribute_forwarding() -> None:
    assert "__getattr__" not in OperationCommitter.__dict__
    assert OperationCommitter.commit.__name__ == "commit"
    assert CommitCoordinator.commit.__name__ == "commit"
    assert CanonicalMemoryCommitHandler._validate_canonical_evidence.__name__ == "_validate_canonical_evidence"
    assert ActionPolicyCommitHandler._apply_action_policy_mutation.__name__ == "_apply_action_policy_mutation"


def test_fault_injection_hooks_remain_concrete_facade_methods() -> None:
    hooks = {
        "_apply_source",
        "_apply_index",
        "_write_operation_marker",
        "_write_transaction_marker",
        "_write_outbox_event",
        "_finalize_canonical_outbox",
        "_capture_regular_source_effect",
        "_build_regular_relation_manifest",
    }
    assert hooks <= OperationCommitter.__dict__.keys()


def test_direct_committer_construction_registers_canonical_store_classification(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()

    OperationCommitter(source, index, str(tmp_path), relation_store=relations)

    canonical_uri = "memoryos://user/u1/memories/canonical/slot-1"
    assert source.domain_classifier.owns_uri(canonical_uri)
    assert index.domain_classifier.owns_uri(canonical_uri)
    assert relations.domain_classifier.owns_uri(canonical_uri)
