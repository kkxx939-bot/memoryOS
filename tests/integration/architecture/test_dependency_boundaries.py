from __future__ import annotations

from pathlib import Path

from tests.support.import_graph import ImportEdge, production_imports

ROOT = Path(__file__).resolve().parents[3]

COMPATIBILITY_MODULES = {
    "memoryos/adapters/agent_hooks/cli.py",
    "memoryos/adapters/agent_hooks/mcp_client.py",
    "memoryos/adapters/agent_hooks/sanitizer.py",
    "memoryos/adapters/filesystem/fs_lock_store.py",
    "memoryos/adapters/filesystem/fs_source_store.py",
    "memoryos/adapters/sqlite/sqlite_index_store.py",
    "memoryos/adapters/sqlite/sqlite_metadata_store.py",
    "memoryos/adapters/sqlite/sqlite_queue_store.py",
    "memoryos/adapters/sqlite/sqlite_relation_store.py",
    "memoryos/adapters/vector/chroma_store.py",
    "memoryos/adapters/vector/local_vector_store.py",
    "memoryos/adapters/vector/milvus_store.py",
    "memoryos/adapters/vector/qdrant_store.py",
    "memoryos/action_policy/update/feedback_commit_planner.py",
    "memoryos/api/limits.py",
    "memoryos/api/sdk/result.py",
    "memoryos/behavior/update/behavior_lifecycle.py",
    "memoryos/contextdb/retrieval/candidate_generator.py",
    "memoryos/contextdb/retrieval/canonical_resolver.py",
    "memoryos/contextdb/retrieval/context_assembler.py",
    "memoryos/contextdb/retrieval/orchestrator.py",
    "memoryos/contextdb/retrieval/packing.py",
    "memoryos/contextdb/retrieval/query_planner.py",
    "memoryos/contextdb/retrieval/service.py",
    "memoryos/contextdb/scope.py",
    "memoryos/contextdb/session/commit_group.py",
    "memoryos/contextdb/session/context_projector.py",
    "memoryos/contextdb/session/planning.py",
    "memoryos/contextdb/session/planning_envelope.py",
    "memoryos/contextdb/session/session_archive.py",
    "memoryos/contextdb/session/session_commit.py",
    "memoryos/contextdb/store/local_stores.py",
    "memoryos/contextdb/store/sqlite_index_store.py",
    "memoryos/contextdb/store/sqlite_lock_store.py",
    "memoryos/contextdb/store/sqlite_queue_store.py",
    "memoryos/contextdb/store/sqlite_relation_store.py",
    "memoryos/contextdb/store/vector_store.py",
    "memoryos/contextdb/transaction/recovery.py",
    "memoryos/contextdb/transaction/redo_log.py",
    "memoryos/contextdb/transaction/__init__.py",
    "memoryos/operations/commit/quarantine.py",
    "memoryos/prediction/pipeline/executor.py",
    "memoryos/prediction/pipeline/predictive_observation_processor.py",
    "memoryos/runtime/readiness.py",
    "memoryos/runtime/agent_hook_transport.py",
    "memoryos/skill/tool_registry.py",
    "memoryos/workers/readiness.py",
}

COMPATIBILITY_PREFIXES = (
    "memoryos/contextdb/session/planners/",
)

COMPATIBILITY_IMPORTS = {
    (
        "memoryos/prediction/pipeline/__init__.py",
        "memoryos.application.prediction.observation_processor",
    ),
}

FORBIDDEN_TARGETS = {
    "core": {
        "action_policy",
        "adapters",
        "api",
        "application",
        "behavior",
        "contextdb",
        "execution",
        "memory",
        "observability",
        "operations",
        "prediction",
        "providers",
        "runtime",
        "security",
        "workers",
    },
    "contextdb": {
        "action_policy",
        "api",
        "application",
        "behavior",
        "memory",
        "operations",
        "prediction",
        "providers",
        "runtime",
        "workers",
    },
    "memory": {"action_policy", "adapters", "api", "behavior", "prediction", "providers", "runtime", "workers"},
    "behavior": {"action_policy", "adapters", "api", "memory", "prediction", "providers", "runtime", "workers"},
    "action_policy": {"adapters", "api", "behavior", "memory", "prediction", "providers", "runtime", "workers"},
    "operations": {
        "action_policy",
        "adapters",
        "api",
        "behavior",
        "memory",
        "prediction",
        "providers",
        "runtime",
        "workers",
    },
    "prediction": {"adapters", "api", "application", "providers", "runtime", "workers"},
    "application": {"adapters", "api", "providers", "runtime", "workers"},
    "workers": {"api", "runtime"},
    "security": {"adapters", "api", "application", "runtime", "workers"},
}

CONTEXTDB_OPERATIONS_COMPATIBILITY = {
    (
        "memoryos/contextdb/context_db.py",
        "memoryos.operations.model.context_operation",
    ),
    (
        "memoryos/contextdb/session/commit_group.py",
        "memoryos.operations.commit.commit_group",
    ),
    (
        "memoryos/contextdb/transaction/__init__.py",
        "memoryos.operations.commit.recovery",
    ),
    (
        "memoryos/contextdb/transaction/__init__.py",
        "memoryos.operations.commit.redo_log",
    ),
    (
        "memoryos/contextdb/transaction/recovery.py",
        "memoryos.operations.commit.recovery",
    ),
    (
        "memoryos/contextdb/transaction/redo_log.py",
        "memoryos.operations.commit.redo_log",
    ),
}


def _relative(edge: ImportEdge) -> str:
    return edge.source.relative_to(ROOT).as_posix()


def _top_level(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "memoryos" else ""


def _is_compatibility(path: str) -> bool:
    return path in COMPATIBILITY_MODULES or path.startswith(COMPATIBILITY_PREFIXES)


def _is_compatibility_import(edge: ImportEdge) -> bool:
    path = _relative(edge)
    return (
        _is_compatibility(path)
        or (path, edge.target) in COMPATIBILITY_IMPORTS
        or (path, edge.target) in CONTEXTDB_OPERATIONS_COMPATIBILITY
    )


def test_inner_packages_do_not_import_forbidden_outer_or_domain_packages() -> None:
    violations: list[str] = []
    for edge in production_imports(ROOT):
        path = _relative(edge)
        if _is_compatibility_import(edge):
            continue
        source_top = Path(path).parts[1]
        target_top = _top_level(edge.target)
        if target_top in FORBIDDEN_TARGETS.get(source_top, set()):
            violations.append(f"{path}:{edge.line} [{edge.kind}] -> {edge.target}")
    assert violations == []


def test_lower_layers_never_import_api_entrypoints() -> None:
    lower_layers = {
        "action_policy",
        "adapters",
        "application",
        "behavior",
        "contextdb",
        "core",
        "execution",
        "memory",
        "operations",
        "prediction",
        "providers",
        "security",
        "workers",
    }
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if not _is_compatibility_import(edge)
        and Path(_relative(edge)).parts[1] in lower_layers
        and _top_level(edge.target) == "api"
    ]
    assert violations == []


def test_contextdb_only_references_operations_from_exact_compatibility_exports() -> None:
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("memoryos/contextdb/")
        and _top_level(edge.target) == "operations"
        and (_relative(edge), edge.target) not in CONTEXTDB_OPERATIONS_COMPATIBILITY
    ]
    assert violations == []


def test_api_does_not_import_agent_hook_adapters() -> None:
    composition_roots = {"memoryos/api/cli/agent_hooks.py"}
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if Path(_relative(edge)).parts[1] == "api"
        and _relative(edge) not in composition_roots
        and (
            edge.target == "memoryos.adapters.agent_hooks"
            or edge.target.startswith("memoryos.adapters.agent_hooks.")
        )
    ]
    assert violations == []


def test_agent_hook_adapters_do_not_import_api_or_runtime() -> None:
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("memoryos/adapters/agent_hooks/")
        and not _is_compatibility(_relative(edge))
        and _top_level(edge.target) in {"api", "runtime"}
    ]
    assert violations == []


def test_security_does_not_import_contextdb() -> None:
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("memoryos/security/")
        and _top_level(edge.target) == "contextdb"
    ]
    assert violations == []


def test_compatibility_modules_remain_thin_and_side_effect_free() -> None:
    oversized = []
    for relative in sorted(COMPATIBILITY_MODULES):
        path = ROOT / relative
        if not path.exists():
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 50:
            oversized.append(f"{relative}: {line_count} lines")
    assert oversized == []
