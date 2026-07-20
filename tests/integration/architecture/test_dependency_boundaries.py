from __future__ import annotations

from pathlib import Path

from tests.support.import_graph import ImportEdge, production_imports

ROOT = Path(__file__).resolve().parents[3]

COMPATIBILITY_MODULES = {
    "openApi/limits.py",
    "openApi/sdk/result.py",
    "infrastructure/context/assembler.py",
    "infrastructure/context/candidate_generator.py",
    "infrastructure/context/retrieval_service.py",
    "infrastructure/context/trace_erase.py",
    "memoryos/contextdb/scope.py",
    "memoryos/contextdb/session/commit_group.py",
    "memoryos/contextdb/session/context_projector.py",
    "memoryos/contextdb/session/planning.py",
    "memoryos/contextdb/session/planning_envelope.py",
    "memoryos/contextdb/session/commit.py",
    "memoryos/contextdb/store/local_stores.py",
    "memoryos/contextdb/store/sqlite_index_store.py",
    "memoryos/contextdb/store/sqlite_lock_store.py",
    "memoryos/contextdb/store/sqlite_queue_store.py",
    "memoryos/contextdb/store/sqlite_relation_store.py",
    "memoryos/contextdb/transaction/recovery.py",
    "memoryos/contextdb/transaction/redo_log.py",
    "memoryos/contextdb/transaction/__init__.py",
    "memoryos/runtime/agent_hook_transport.py",
}

COMPATIBILITY_PREFIXES = (
    "memoryos/contextdb/retrieval/",
    "memoryos/contextdb/layers/",
    "memoryos/contextdb/maintenance/",
    "memoryos/contextdb/session/planners/",
)

COMPATIBILITY_IMPORTS: set[tuple[str, str]] = set()

FORBIDDEN_TARGETS = {
    "foundation": {
        "agent_hook",
        "action_policy",
        "application",
        "behavior_core",
        "infrastructure",
        "memory_core",
        "openApi",
        "transaction",
        "policy",
        "providers",
        "runtime",
        "security",
        "workers",
        "pre",
    },
    "pre": {
        "agent_hook",
        "action_policy",
        "application",
        "behavior_core",
        "infrastructure",
        "memory_core",
        "openApi",
        "transaction",
        "policy",
        "providers",
        "runtime",
        "workers",
        "security",
    },
    "memory_core": {
        "agent_hook",
        "action_policy",
        "application",
        "behavior_core",
        "infrastructure",
        "openApi",
        "transaction",
        "policy",
        "providers",
        "runtime",
        "security",
        "workers",
    },
    "behavior_core": {
        "agent_hook",
        "action_policy",
        "application",
        "infrastructure",
        "memory_core",
        "openApi",
        "transaction",
        "policy",
        "providers",
        "runtime",
        "security",
        "workers",
    },
    "core": {
        "agent_hook",
        "action_policy",
        "adapters",
        "openApi",
        "application",
        "behavior",
        "contextdb",
        "infrastructure",
        "memory",
        "observability",
        "transaction",
        "providers",
        "runtime",
        "security",
        "workers",
    },
    "contextdb": {
        "agent_hook",
        "action_policy",
        "openApi",
        "application",
        "behavior",
        "memory",
        "transaction",
        "providers",
        "runtime",
        "workers",
    },
    "memory": {
        "agent_hook",
        "action_policy",
        "adapters",
        "infrastructure",
        "openApi",
        "behavior",
        "providers",
        "runtime",
        "workers",
    },
    "behavior": {
        "agent_hook",
        "action_policy",
        "adapters",
        "openApi",
        "memory",
        "providers",
        "runtime",
        "workers",
    },
    "action_policy": {
        "agent_hook",
        "adapters",
        "openApi",
        "memory",
        "providers",
        "runtime",
        "workers",
    },
    "transaction": {
        "agent_hook",
        "action_policy",
        "adapters",
        "openApi",
        "behavior",
        "memory",
        "providers",
        "runtime",
        "workers",
    },
    "application": {"agent_hook", "adapters", "openApi", "providers", "runtime", "workers"},
    "workers": {"agent_hook", "openApi", "runtime"},
    "security": {
        "agent_hook",
        "adapters",
        "infrastructure",
        "openApi",
        "application",
        "runtime",
        "workers",
    },
}

CONTEXTDB_OPERATIONS_COMPATIBILITY: set[tuple[str, str]] = set()


def _relative(edge: ImportEdge) -> str:
    return edge.source.relative_to(ROOT).as_posix()


def _top_level(module: str) -> str:
    parts = module.split(".")
    if parts[0] == "foundation":
        return "foundation"
    if parts[0] == "pre":
        return "pre"
    if parts[:2] == ["memory", "core"]:
        return "memory_core"
    if parts[0] == "memory":
        return "application"
    if parts[:2] == ["behavior", "core"]:
        return "behavior_core"
    if parts[0] == "behavior":
        return "application"
    if parts[0] == "agent_hook":
        return "agent_hook"
    if parts[0] == "openApi":
        return "openApi"
    if parts[0] == "infrastructure":
        return "infrastructure"
    if parts[0] == "transaction":
        return "transaction"
    if parts[:2] == ["policy", "action_policy"]:
        return "action_policy"
    if parts[0] == "policy":
        return "policy"
    return parts[1] if len(parts) > 1 and parts[0] == "memoryos" else ""


def _source_top(path: str) -> str:
    parts = Path(path).parts
    if not parts:
        return ""
    if parts[0] == "foundation":
        return "foundation"
    if parts[0] == "pre":
        return "pre"
    if parts[:2] == ("memory", "core"):
        return "memory_core"
    if parts[0] == "memory":
        return "application"
    if parts[:2] == ("behavior", "core"):
        return "behavior_core"
    if parts[0] == "behavior":
        return "application"
    if parts[0] == "agent_hook":
        return "agent_hook"
    if parts[0] == "openApi":
        return "openApi"
    if parts[0] == "infrastructure":
        return "infrastructure"
    if parts[0] == "transaction":
        return "transaction"
    if parts[:2] == ("policy", "action_policy"):
        return "action_policy"
    if parts[0] == "policy":
        return "policy"
    return parts[1] if len(parts) > 1 else path


def _is_compatibility(path: str) -> bool:
    return path in COMPATIBILITY_MODULES or path.startswith(COMPATIBILITY_PREFIXES)


def _is_compatibility_import(edge: ImportEdge) -> bool:
    path = _relative(edge)
    return (
        _is_compatibility(path)
        or (path, edge.target) in COMPATIBILITY_IMPORTS
        or (path, edge.target) in CONTEXTDB_OPERATIONS_COMPATIBILITY
    )


def _is_store_contract_import(edge: ImportEdge) -> bool:
    """领域代码可以依赖存储协议，但不能依赖具体持久化实现。"""

    return edge.target == "infrastructure.store.contracts" or edge.target.startswith("infrastructure.store.contracts.")


def test_inner_packages_do_not_import_forbidden_outer_or_domain_packages() -> None:
    violations: list[str] = []
    for edge in production_imports(ROOT):
        path = _relative(edge)
        if _is_compatibility_import(edge) or _is_store_contract_import(edge):
            continue
        source_top = _source_top(path)
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
        "infrastructure",
        "memory",
        "transaction",
        "providers",
        "security",
        "workers",
        "pre",
    }
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if not _is_compatibility_import(edge)
        and _source_top(_relative(edge)) in lower_layers
        and _top_level(edge.target) == "openApi"
    ]
    assert violations == []


def test_store_infrastructure_never_imports_context_infrastructure() -> None:
    """持久化实现不得反向依赖上下文规划或召回语义。"""

    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("infrastructure/store/")
        and (edge.target == "infrastructure.context" or edge.target.startswith("infrastructure.context."))
    ]
    assert violations == []


def test_operation_transactions_never_import_context_infrastructure() -> None:
    """通用事务层只能依赖上下文端口，不能反向依赖具体上下文语义。"""

    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("transaction/")
        and (edge.target == "infrastructure.context" or edge.target.startswith("infrastructure.context."))
    ]
    assert violations == []


def test_transaction_kernel_only_imports_store_contracts_and_context_models() -> None:
    """事务内核不能直接选择文件、SQLite 或其他具体持久化实现。"""

    allowed_prefixes = (
        "infrastructure.store.contracts",
        "infrastructure.store.model",
    )
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("transaction/")
        and edge.target.startswith("infrastructure.store")
        and not edge.target.startswith(allowed_prefixes)
    ]
    assert violations == []


def test_memory_transactions_depend_on_erase_and_consolidation_ports() -> None:
    """删除和合并事务不得重新导入其文件 Store 实现。"""

    forbidden_targets = {
        "infrastructure.store.memory.erasure_store",
        "infrastructure.store.memory.consolidation_store",
    }
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("memory/commit/") and edge.target in forbidden_targets
    ]
    assert violations == []


def test_memory_store_implementations_do_not_import_erase_or_consolidation_services() -> None:
    """具体 Store 只依赖共享端口，不能反向依赖提交协调器。"""

    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("infrastructure/store/memory/")
        and edge.target in {"memory.commit.erase", "memory.commit.consolidation"}
    ]
    assert violations == []


def test_contextdb_only_references_operations_from_exact_compatibility_exports() -> None:
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("memoryos/contextdb/")
        and _top_level(edge.target) == "transaction"
        and (_relative(edge), edge.target) not in CONTEXTDB_OPERATIONS_COMPATIBILITY
    ]
    assert violations == []


def test_openapi_only_composition_root_imports_agent_hook() -> None:
    composition_roots = {
        "openApi/cli/agent_hooks.py",
        "openApi/http/app.py",
    }
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _source_top(_relative(edge)) == "openApi"
        and _relative(edge) not in composition_roots
        and (edge.target == "agent_hook" or edge.target.startswith("agent_hook."))
    ]
    assert violations == []


def test_agent_hook_does_not_import_openapi_or_runtime() -> None:
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("agent_hook/")
        and not _is_compatibility(_relative(edge))
        and _top_level(edge.target) in {"openApi", "runtime"}
    ]
    assert violations == []


def test_security_does_not_import_contextdb() -> None:
    violations = [
        f"{_relative(edge)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if _relative(edge).startswith("memoryos/security/") and _top_level(edge.target) == "contextdb"
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
