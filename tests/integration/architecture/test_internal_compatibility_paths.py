from __future__ import annotations

import ast
from pathlib import Path

from tests.support.import_graph import production_imports

ROOT = Path(__file__).resolve().parents[3]

OLD_MODULES = (
    "memoryos.action_policy.update.feedback_commit_planner",
    "memoryos.adapters.agent_hooks.cli",
    "memoryos.adapters.agent_hooks.events",
    "memoryos.adapters.agent_hooks.mcp_client",
    "memoryos.adapters.agent_hooks.sanitizer",
    "memoryos.adapters.filesystem.fs_lock_store",
    "memoryos.adapters.filesystem.fs_source_store",
    "memoryos.adapters.sqlite",
    "memoryos.adapters.vector.chroma_store",
    "memoryos.adapters.vector.local_vector_store",
    "memoryos.adapters.vector.milvus_store",
    "memoryos.adapters.vector.qdrant_store",
    "memoryos.api.limits",
    "memoryos.api.sdk.result",
    "memoryos.behavior.update.behavior_lifecycle",
    "memoryos.contextdb.retrieval.candidate_generator",
    "memoryos.contextdb.retrieval.canonical_resolver",
    "memoryos.contextdb.retrieval.context_assembler",
    "memoryos.contextdb.retrieval.orchestrator",
    "memoryos.contextdb.retrieval.packing",
    "memoryos.contextdb.retrieval.query_planner",
    "memoryos.contextdb.retrieval.service",
    "memoryos.contextdb.scope",
    "memoryos.contextdb.session.commit_group",
    "memoryos.contextdb.session.context_projector",
    "memoryos.contextdb.session.planners",
    "memoryos.contextdb.session.planning",
    "memoryos.contextdb.session.planning_envelope",
    "memoryos.contextdb.session.session_archive",
    "memoryos.contextdb.session.session_commit",
    "memoryos.contextdb.store.local_stores",
    "memoryos.contextdb.store.sqlite_index_store",
    "memoryos.contextdb.store.sqlite_lock_store",
    "memoryos.contextdb.store.sqlite_queue_store",
    "memoryos.contextdb.store.sqlite_relation_store",
    "memoryos.contextdb.store.vector_store",
    "memoryos.contextdb.transaction.recovery",
    "memoryos.contextdb.transaction.redo_log",
    "memoryos.core.time",
    "memoryos.operations.commit.quarantine",
    "memoryos.prediction.pipeline.executor",
    "memoryos.prediction.pipeline.predictive_observation_processor",
    "memoryos.runtime.agent_hook_transport",
    "memoryos.skill.tool_registry",
)

MOVED_SYMBOLS = {
    "memoryos.memory.canonical.event": {
        "canonical_digest",
        "canonical_json",
        "canonicalize",
        "immutable_snapshot",
    },
    "memoryos.operations.commit.effect_marker": {
        "atomic_write_bytes",
        "atomic_write_json",
        "read_json",
    },
    "memoryos.contextdb.store.source_store": {
        "IndexHit",
        "IndexStore",
        "LeaseLostError",
        "LockLostError",
        "LockStore",
        "LockToken",
        "QueueIdempotencyConflictError",
        "QueueJob",
        "QueueLeaseIdentityError",
        "QueueStore",
        "RelationStore",
        "is_canonical_memory_object",
        "is_canonical_memory_uri",
    },
    "memoryos.contextdb.scope": {
        "AuthorityPolicy",
        "CORE_SCOPE_KINDS",
        "ContextScope",
        "HIERARCHICAL_SCOPE_KINDS",
        "ScopeRef",
        "ScopeResolutionSource",
        "ScopeSelector",
        "VisibilityPolicy",
        "scope_key_candidates_from_payload",
        "scope_key_from_payload",
        "scope_keys_from_payloads",
    },
    "memoryos.providers.rerank": {"Reranker"},
}


def test_production_code_does_not_import_whole_compatibility_modules() -> None:
    violations = [
        f"{edge.source.relative_to(ROOT)}:{edge.line} -> {edge.target}"
        for edge in production_imports(ROOT)
        if any(edge.target == old or edge.target.startswith(f"{old}.") for old in OLD_MODULES)
    ]
    assert violations == []


def test_production_code_imports_moved_symbols_from_their_real_owner() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "memoryos").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in MOVED_SYMBOLS:
                continue
            forbidden = MOVED_SYMBOLS[node.module]
            names = sorted(alias.name for alias in node.names if alias.name in forbidden)
            if names:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno} -> {node.module}:{','.join(names)}")
    assert violations == []
