from __future__ import annotations

import ast
from pathlib import Path

from tests.support.import_graph import production_imports, production_paths

ROOT = Path(__file__).resolve().parents[3]

OLD_MODULES = (
    "memoryos.adapters.agent_hooks",
    "memoryos.adapters.locks",
    "memoryos.adapters.persistence.filesystem",
    "memoryos.adapters.persistence.in_memory",
    "memoryos.adapters.filesystem.fs_lock_store",
    "memoryos.adapters.filesystem.fs_source_store",
    "memoryos.adapters.sqlite",
    "memoryos.adapters.vector",
    "openApi.limits",
    "openApi.sdk.result",
    "memoryos.behavior",
    "memoryos.connect",
    "memoryos.application.context",
    "memoryos.contextdb",
    "memoryos.contextdb.retrieval",
    "memoryos.contextdb.layers",
    "memoryos.contextdb.maintenance",
    "memoryos.memory",
    "memoryos.observability",
    "memoryos.prediction",
    "memoryos.application.prediction",
    "memoryos.application.session",
    "infrastructure.context.assembler",
    "infrastructure.context.candidate_generator",
    "infrastructure.context.retrieval_service",
    "infrastructure.context.trace_erase",
    "infrastructure.context.retrieval.errors",
    "infrastructure.context.retrieval.lexical",
    "infrastructure.context.trace.erase",
    "infrastructure.context.trace.store",
    "memoryos.contextdb.scope",
    "memoryos.contextdb.session.commit_group",
    "memoryos.contextdb.session.session_commit",
    "memoryos.contextdb.session.context_projector",
    "memoryos.contextdb.session.planners",
    "memoryos.contextdb.session.planning",
    "memoryos.contextdb.session.planning_envelope",
    "memoryos.contextdb.session.session_archive",
    "memoryos.contextdb.session.commit",
    "memoryos.contextdb.store.local_stores",
    "memoryos.contextdb.store.sqlite_index_store",
    "memoryos.contextdb.store.sqlite_lock_store",
    "memoryos.contextdb.store.sqlite_queue_store",
    "memoryos.contextdb.store.sqlite_relation_store",
    "infrastructure.store.contracts.vector_store",
    "memoryos.contextdb.transaction.recovery",
    "memoryos.contextdb.transaction.redo_log",
    "memoryos.core",
    "memoryos.runtime",
    "memoryos.runtime.readiness",
    "behavior.model",
    "behavior.update",
    "behavior.extraction",
    "policy.action_policy.model.action_candidate",
    "policy.action_policy.model.action_lifecycle",
    "policy.action_policy.model.action_value",
    "policy.action_policy.model.penalty_signal",
    "policy.action_policy.ranking.candidate_generator",
    "policy.action_policy.ranking.candidate_ranker",
    "policy.action_policy.update.cooldown_updater",
    "policy.action_policy.update.penalty_updater",
    "policy.action_policy.update.reward_updater",
    "memoryos.runtime.agent_hook_transport",
    "memoryos.execution",
    "memoryos.skill.tool_registry",
    "policy.action_policy.execute",
)

MOVED_SYMBOLS = {
    "infrastructure.store.contracts.source": {
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
    },
    "memoryos.contextdb.scope": {
        "CORE_SCOPE_KINDS",
        "HIERARCHICAL_SCOPE_KINDS",
        "ScopeRef",
        "ScopeResolutionSource",
        "ScopeSelector",
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
    for path in production_paths(ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in MOVED_SYMBOLS:
                continue
            forbidden = MOVED_SYMBOLS[node.module]
            names = sorted(alias.name for alias in node.names if alias.name in forbidden)
            if names:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno} -> {node.module}:{','.join(names)}")
    assert violations == []
