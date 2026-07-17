from __future__ import annotations

import json
import subprocess
import sys


def test_root_import_does_not_load_delivery_persistence_or_worker_graph() -> None:
    code = """
import json
import sys
import memoryos

blocked = (
    "memoryos.adapters.persistence.sqlite",
    "memoryos.api.http",
    "memoryos.api.mcp",
    "memoryos.api.sdk",
    "memoryos.contextdb.store.sqlite",
    "memoryos.workers",
)
print(json.dumps(sorted(name for name in sys.modules if name.startswith(blocked))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_contextdb_facade_import_does_not_load_operations_graph() -> None:
    code = """
import json
import sys
import memoryos.contextdb.context_db

print(json.dumps(sorted(name for name in sys.modules if name.startswith("memoryos.operations"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_prediction_pipeline_package_does_not_eagerly_load_execution() -> None:
    code = """
import json
import sys
import memoryos.prediction.pipeline

print(json.dumps(sorted(name for name in sys.modules if name.startswith("memoryos.execution"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_historical_prediction_executor_module_does_not_restore_eager_cycle() -> None:
    code = """
import json
import sys
import memoryos.prediction.pipeline.executor

print(json.dumps(sorted(name for name in sys.modules if name.startswith("memoryos.execution"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_root_and_historical_public_imports_resolve_identical_objects() -> None:
    import memoryos
    from memoryos.api.sdk import MemoryOSClient as PackageClient
    from memoryos.api.sdk.client import MemoryOSClient
    from memoryos.contextdb.context_db import ContextDB
    from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan
    from memoryos.prediction.model.prediction_request import PredictionRequest

    assert memoryos.MemoryOSClient is PackageClient is MemoryOSClient
    assert memoryos.ContextDB is ContextDB
    assert memoryos.RetrievalOptions is RetrievalOptions
    assert memoryos.RetrievalQueryPlan is RetrievalQueryPlan
    assert memoryos.PredictionRequest is PredictionRequest
    assert set(memoryos.__all__) == {
        "__version__",
        "ActionCandidate",
        "ActionPolicy",
        "ContextDB",
        "MemoryOSClient",
        "PredictionRequest",
        "RetrievalOptions",
        "RetrievalQueryPlan",
    }


def test_moved_public_objects_keep_historical_import_identity() -> None:
    from memoryos.action_policy.update import FeedbackCommitPlanner as PackageFeedbackCommitPlanner
    from memoryos.action_policy.update.feedback_commit_planner import (
        FeedbackCommitPlanner as OldFeedbackCommitPlanner,
    )
    from memoryos.adapters.agent_hooks.cli import main as old_agent_hook_main
    from memoryos.adapters.agent_hooks.events import AgentHookEvent as OldAgentHookEvent
    from memoryos.adapters.agent_hooks.mcp_client import AgentHookTransportClient as OldAgentHookTransportClient
    from memoryos.adapters.agent_hooks.sanitizer import sanitize_text as old_sanitize_text
    from memoryos.adapters.persistence.filesystem.session_archive import (
        EvidenceArchiveIntegrityError as AdapterEvidenceArchiveIntegrityError,
    )
    from memoryos.adapters.persistence.filesystem.session_archive import (
        SessionArchiveStore as NewSessionArchiveStore,
    )
    from memoryos.adapters.persistence.sqlite import SQLiteIndexStore as NewSQLiteIndexStore
    from memoryos.adapters.vector import (
        ChromaStore as PackageChromaStore,
    )
    from memoryos.adapters.vector import (
        LocalVectorStore as PackageLocalVectorStore,
    )
    from memoryos.adapters.vector import (
        MilvusStore as PackageMilvusStore,
    )
    from memoryos.adapters.vector import (
        QdrantStore as PackageQdrantStore,
    )
    from memoryos.adapters.vector.chroma import ChromaStore
    from memoryos.adapters.vector.chroma_store import ChromaStore as OldChromaStore
    from memoryos.adapters.vector.in_memory import LocalVectorStore
    from memoryos.adapters.vector.local_vector_store import LocalVectorStore as OldLocalVectorStore
    from memoryos.adapters.vector.milvus import MilvusStore
    from memoryos.adapters.vector.milvus_store import MilvusStore as OldMilvusStore
    from memoryos.adapters.vector.qdrant import QdrantStore
    from memoryos.adapters.vector.qdrant_store import QdrantStore as OldQdrantStore
    from memoryos.api.cli.agent_hook_transport import AgentHookTransportClient
    from memoryos.api.cli.agent_hooks import main as agent_hook_main
    from memoryos.api.limits import bounded_int as old_bounded_int
    from memoryos.api.sdk import ProcessObservationResult as PackageObservationResult
    from memoryos.api.sdk.result import ProcessObservationResult as PublicObservationResult
    from memoryos.api.trusted_context import TrustedRequestContext as PublicTrustedRequestContext
    from memoryos.application.memory.behavior_lifecycle import (
        BehaviorLifecycleResult,
        BehaviorLifecycleService,
    )
    from memoryos.application.memory.feedback_commit_planner import FeedbackCommitPlanner
    from memoryos.application.prediction.observation_processor import (
        PredictiveObservationProcessor as ApplicationPredictiveObservationProcessor,
    )
    from memoryos.application.prediction.result import ProcessObservationResult
    from memoryos.application.session.events import AgentHookEvent
    from memoryos.application.session.planners import MemoryCommitPlanner as NewMemoryCommitPlanner
    from memoryos.behavior.update import (
        BehaviorLifecycleResult as PackageBehaviorLifecycleResult,
    )
    from memoryos.behavior.update import (
        BehaviorLifecycleService as PackageBehaviorLifecycleService,
    )
    from memoryos.behavior.update.behavior_lifecycle import (
        BehaviorLifecycleResult as OldBehaviorLifecycleResult,
    )
    from memoryos.behavior.update.behavior_lifecycle import (
        BehaviorLifecycleService as OldBehaviorLifecycleService,
    )
    from memoryos.contextdb.context_db import CommitResult
    from memoryos.contextdb.context_db import ContextOperation as HistoricalContextOperation
    from memoryos.contextdb.context_db import OperationCommitter as HistoricalOperationCommitter
    from memoryos.contextdb.context_db import ProjectionRecordStore as HistoricalProjectionRecordStore
    from memoryos.contextdb.retrieval.limits import bounded_int
    from memoryos.contextdb.scope import ContextScope as OldContextScope
    from memoryos.contextdb.scope import ScopeRef as OldScopeRef
    from memoryos.contextdb.session import SessionArchiveStore as PackageSessionArchiveStore
    from memoryos.contextdb.session.errors import EvidenceArchiveIntegrityError
    from memoryos.contextdb.session.planners import MemoryCommitPlanner
    from memoryos.contextdb.session.session_archive import SessionArchiveStore
    from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
    from memoryos.contextdb.transaction.recovery import RecoveryService
    from memoryos.core.durable_io import atomic_write_json
    from memoryos.core.integrity import canonical_json
    from memoryos.core.readiness import RuntimeReadinessState, require_source_store_ready
    from memoryos.core.types import ContextScope, ScopeRef
    from memoryos.execution.action_executor import ActionExecutor, ExecutionResult, Executor
    from memoryos.execution.tool_registry import ToolRegistry
    from memoryos.memory.canonical.event import canonical_json as old_canonical_json
    from memoryos.memory.canonical.projection_state import ProjectionRecordStore
    from memoryos.operations.commit.effect_marker import atomic_write_json as old_atomic_write_json
    from memoryos.operations.commit.operation_committer import OperationCommitter
    from memoryos.operations.commit.recovery import RecoveryService as NewRecoveryService
    from memoryos.operations.model.context_diff import ContextDiff
    from memoryos.operations.model.context_operation import ContextOperation
    from memoryos.prediction.pipeline import ActionExecutor as PackageActionExecutor
    from memoryos.prediction.pipeline import ExecutionResult as PackageExecutionResult
    from memoryos.prediction.pipeline import Executor as PackageExecutor
    from memoryos.prediction.pipeline import PredictiveObservationProcessor as PackagePredictiveObservationProcessor
    from memoryos.prediction.pipeline.executor import ActionExecutor as OldActionExecutor
    from memoryos.prediction.pipeline.executor import ExecutionResult as OldExecutionResult
    from memoryos.prediction.pipeline.executor import Executor as OldExecutor
    from memoryos.prediction.pipeline.predictive_observation_processor import (
        PredictiveObservationProcessor as OldPredictiveObservationProcessor,
    )
    from memoryos.runtime.agent_hook_transport import AgentHookTransportClient as RuntimeAgentHookTransportClient
    from memoryos.runtime.readiness import RuntimeReadinessState as OldRuntimeReadinessState
    from memoryos.security.sanitization import sanitize_text
    from memoryos.security.trusted_context import TrustedRequestContext
    from memoryos.skill.tool_registry import ToolRegistry as OldToolRegistry
    from memoryos.workers.readiness import require_source_store_ready as old_require_source_store_ready

    assert old_canonical_json is canonical_json
    assert CommitResult is ContextDiff
    assert HistoricalContextOperation is ContextOperation
    assert HistoricalOperationCommitter is OperationCommitter
    assert HistoricalProjectionRecordStore is ProjectionRecordStore
    assert old_agent_hook_main is agent_hook_main
    assert OldAgentHookEvent is AgentHookEvent
    assert OldAgentHookTransportClient is RuntimeAgentHookTransportClient is AgentHookTransportClient
    assert old_atomic_write_json is atomic_write_json
    assert old_require_source_store_ready is require_source_store_ready
    assert OldRuntimeReadinessState is RuntimeReadinessState
    assert old_sanitize_text is sanitize_text
    assert old_bounded_int is bounded_int
    assert OldContextScope is ContextScope
    assert OldScopeRef is ScopeRef
    assert SQLiteIndexStore is NewSQLiteIndexStore
    assert MemoryCommitPlanner is NewMemoryCommitPlanner
    assert SessionArchiveStore is PackageSessionArchiveStore is NewSessionArchiveStore
    assert RecoveryService is NewRecoveryService
    assert OldToolRegistry is ToolRegistry
    assert OldActionExecutor is PackageActionExecutor is ActionExecutor
    assert OldExecutionResult is PackageExecutionResult is ExecutionResult
    assert OldExecutor is PackageExecutor is Executor
    assert (
        OldPredictiveObservationProcessor
        is PackagePredictiveObservationProcessor
        is ApplicationPredictiveObservationProcessor
    )
    assert PublicObservationResult is PackageObservationResult is ProcessObservationResult
    assert (
        OldFeedbackCommitPlanner
        is PackageFeedbackCommitPlanner
        is FeedbackCommitPlanner
    )
    assert (
        OldBehaviorLifecycleResult
        is PackageBehaviorLifecycleResult
        is BehaviorLifecycleResult
    )
    assert (
        OldBehaviorLifecycleService
        is PackageBehaviorLifecycleService
        is BehaviorLifecycleService
    )
    assert OldLocalVectorStore is PackageLocalVectorStore is LocalVectorStore
    assert OldQdrantStore is PackageQdrantStore is QdrantStore
    assert OldMilvusStore is PackageMilvusStore is MilvusStore
    assert OldChromaStore is PackageChromaStore is ChromaStore
    assert PublicTrustedRequestContext is TrustedRequestContext
    assert AdapterEvidenceArchiveIntegrityError is EvidenceArchiveIntegrityError
