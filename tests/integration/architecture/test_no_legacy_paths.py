from __future__ import annotations

from pathlib import Path

from tests.support.import_graph import module_imports


def test_runtime_entry_has_one_root_owner() -> None:
    """整个项目只允许从运行时组合根启动。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "main.py").exists()
    assert not (root / "app").exists()
    assert not (root / "application").exists()
    assert not (root / "memoryos" / "application" / "service.py").exists()
    assert not (root / "bootstrap").exists()
    assert (root / "runtime" / "__init__.py").is_file()
    assert (root / "runtime" / "__main__.py").is_file()
    assert (root / "runtime" / "entry.py").is_file()
    assert not (root / "openApi" / "cli" / "main.py").exists()
    assert (root / "openApi" / "cli" / "commands.py").is_file()


def test_legacy_memoryos_python_package_is_removed() -> None:
    """发行名和 URI 可以保留，但旧 Python 聚合包不得恢复。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos").exists()
    assert (root / "openApi" / "version.py").is_file()
    assert (root / "openApi" / "sdk" / "__init__.py").is_file()
    production_roots = (
        root / "agent_hook",
        root / "behavior",
        root / "foundation",
        root / "infrastructure",
        root / "memory",
        root / "openApi",
        root / "policy",
        root / "pre",
        root / "runtime",
        root / "transaction",
    )
    legacy_imports = [
        f"{path.relative_to(root).as_posix()}:{edge.line} -> {edge.target}"
        for base in production_roots
        for path in base.rglob("*.py")
        for edge in module_imports(path)
        if edge.target == "memoryos" or edge.target.startswith("memoryos.")
    ]
    assert legacy_imports == []


def test_process_and_model_config_have_explicit_owners() -> None:
    """公共进程配置归根级文件，模型连接配置归模型基础设施。"""

    root = Path(__file__).resolve().parents[3]
    assert (root / "config.py").is_file()
    assert (root / "infrastructure" / "model" / "config.py").is_file()
    assert not (root / "memoryos" / "config.py").exists()
    assert "infrastructure" not in (root / "config.py").read_text(encoding="utf-8")
    production_roots = (
        root / "agent_hook",
        root / "infrastructure",
        root / "memoryos",
        root / "openApi",
        root / "runtime",
    )
    legacy_imports = [
        path.relative_to(root).as_posix()
        for base in production_roots
        for path in base.rglob("*.py")
        if "memoryos.config" in path.read_text(encoding="utf-8")
    ]
    assert legacy_imports == []


def test_openapi_delivery_package_has_one_root_owner() -> None:
    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "api").exists()
    assert (root / "openApi" / "__init__.py").is_file()
    assert (root / "openApi" / "sdk" / "client.py").is_file()
    assert (root / "openApi" / "http" / "app.py").is_file()
    assert (root / "openApi" / "mcp" / "stdio.py").is_file()


def test_action_policy_domain_has_one_root_owner() -> None:
    root = Path(__file__).resolve().parents[3]
    feedback_planner = root / "policy" / "action_policy" / "update" / "feedback_commit_planner.py"
    assert not (root / "memoryos" / "action_policy").exists()
    assert not (root / "memoryos" / "prediction").exists()
    assert not (root / "memoryos" / "application" / "prediction").exists()
    assert not (root / "memoryos" / "application" / "session" / "planners" / "action_policy_commit_planner.py").exists()
    assert not (root / "action_policy").exists()
    assert not (root / "memory" / "execute" / "feedback_commit_planner.py").exists()
    assert (root / "policy" / "__init__.py").is_file()
    assert (root / "policy" / "action_policy" / "__init__.py").is_file()
    assert (root / "policy" / "action_policy" / "model" / "action_policy.py").is_file()
    assert (root / "policy" / "action_policy" / "retrieval" / "action_policy_retriever.py").is_file()
    assert (root / "policy" / "action_policy" / "integration" / "commit_handler.py").is_file()
    assert (root / "policy" / "action_policy" / "decision" / "engine.py").is_file()
    assert (root / "policy" / "action_policy" / "decision" / "gate.py").is_file()
    assert not (root / "policy" / "action_policy" / "execute").exists()
    assert (root / "policy" / "action_policy" / "feedback" / "mapper.py").is_file()
    assert (root / "policy" / "action_policy" / "workflow" / "service.py").is_file()
    assert (root / "policy" / "action_policy" / "planning" / "session_commit_planner.py").is_file()
    assert (root / "infrastructure" / "store" / "action_policy" / "decision_ledger.py").is_file()
    retired_aliases = {
        "model/action_candidate.py",
        "model/action_lifecycle.py",
        "model/action_value.py",
        "model/penalty_signal.py",
        "ranking/candidate_generator.py",
        "ranking/candidate_ranker.py",
        "update/cooldown_updater.py",
        "update/penalty_updater.py",
        "update/reward_updater.py",
    }
    assert all(not (root / "policy" / "action_policy" / relative).exists() for relative in retired_aliases)
    assert feedback_planner.is_file()
    feedback_source = feedback_planner.read_text(encoding="utf-8")
    assert "class FeedbackCommitPlanner" in feedback_source
    assert "memory.execute" not in feedback_source


def test_action_execution_is_owned_by_action_policy() -> None:
    """动作执行归 ActionPolicy；Resource 和 Skill 只保留统一 Context 数据类型。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "execution").exists()
    assert not (root / "memoryos" / "skill").exists()
    assert not (root / "execution").exists()
    assert not (root / "capability").exists()
    execution = root / "policy" / "action_policy" / "execution"
    assert (execution / "executor.py").is_file()
    assert (execution / "result.py").is_file()
    assert (execution / "tool_registry.py").is_file()


def test_memory_execution_has_one_root_owner() -> None:
    """用户记忆用例只允许由根目录 memory/execute 承载。"""

    root = Path(__file__).resolve().parents[3]
    execute = root / "memory" / "execute"
    assert not (root / "memoryos" / "application" / "memory").exists()
    assert not (execute / "behavior_lifecycle.py").exists()
    assert not (execute / "feedback_commit_planner.py").exists()
    assert (root / "memory" / "__init__.py").is_file()
    expected_operations = {
        "base.py",
        "command_service.py",
        "consolidate.py",
        "contracts.py",
        "edit.py",
        "external_change.py",
        "forget.py",
        "history.py",
        "remember.py",
    }
    assert (execute / "__init__.py").is_file()
    assert (execute / "pending_review_service.py").is_file()
    assert (execute / "write_planner.py").is_file()
    assert all((execute / name).is_file() for name in expected_operations)
    assert len((execute / "command_service.py").read_text(encoding="utf-8").splitlines()) < 100
    oversized = {
        name: len((execute / name).read_text(encoding="utf-8").splitlines())
        for name in expected_operations
        if name != "command_service.py" and len((execute / name).read_text(encoding="utf-8").splitlines()) > 600
    }
    assert oversized == {}


def test_behavior_execution_has_one_root_owner() -> None:
    """行为核心、Session 编排与 Context 投影必须各自只有一个归属。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memory" / "execute" / "behavior_lifecycle.py").exists()
    assert not (root / "memoryos" / "behavior" / "update" / "behavior_lifecycle.py").exists()
    assert not (root / "memoryos" / "connect").exists()
    assert not (root / "behavior" / "connect").exists()
    assert not (root / "behavior" / "model").exists()
    assert not (root / "behavior" / "update").exists()
    assert not (root / "behavior" / "extraction").exists()
    assert (root / "behavior" / "__init__.py").is_file()
    assert (root / "behavior" / "core" / "model" / "behavior_pattern.py").is_file()
    assert (root / "behavior" / "core" / "formation" / "lifecycle.py").is_file()
    assert (root / "behavior" / "core" / "evaluation" / "behavior_window.py").is_file()
    assert (root / "behavior" / "projection" / "behavior_pattern.py").is_file()
    assert (root / "pre" / "connect" / "__init__.py").is_file()
    assert (root / "pre" / "connect" / "model.py").is_file()
    assert (root / "behavior" / "execute" / "__init__.py").is_file()
    assert (root / "behavior" / "execute" / "session_commit_planner.py").is_file()
    assert not (root / "behavior" / "execute" / "behavior_lifecycle.py").exists()


def test_background_workers_follow_domain_ownership() -> None:
    """后台执行器必须跟随领域归属，旧聚合 workers 包不得恢复。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "workers").exists()
    expected = {
        "runtime/worker/contracts.py",
        "runtime/worker/runner.py",
        "runtime/recovery/transaction_worker.py",
        "memory/worker/document_edit.py",
        "memory/worker/document_scan.py",
        "memory/worker/session_commit.py",
        "behavior/execute/cooling_worker.py",
        "infrastructure/context/maintenance/embedding_worker.py",
        "infrastructure/context/maintenance/semantic_worker.py",
    }
    assert all((root / relative).is_file() for relative in expected)
    assert not (root / "infrastructure" / "context" / "maintenance" / "reindex_worker.py").exists()
    projection_root = root / "memory" / "worker" / "projection"
    projection_files = tuple(projection_root.glob("*.py"))
    assert {path.name for path in projection_files} >= {
        "catalog.py",
        "erase_backend.py",
        "erasure.py",
        "event.py",
        "model.py",
        "publication.py",
        "worker.py",
    }
    assert all(len(path.read_text(encoding="utf-8").splitlines()) < 600 for path in projection_files)


def test_domain_cores_and_technical_foundation_have_one_owner() -> None:
    """旧万能 Core 必须消失；记忆、行为和技术基础分别拥有自己的核心。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "core").exists()
    assert not (root / "memoryos" / "memory" / "documents").exists()
    assert not (root / "evidence").exists()
    assert (root / "pre" / "session" / "archive.py").is_file()
    assert (root / "pre" / "evidence" / "model" / "episode.py").is_file()
    assert (root / "pre" / "evidence" / "session" / "adapter.py").is_file()
    assert (root / "foundation" / "clock.py").is_file()
    assert (root / "foundation" / "ids.py").is_file()
    assert (root / "foundation" / "readiness.py").is_file()
    assert (root / "foundation" / "integrity" / "digest.py").is_file()
    assert (root / "foundation" / "scope.py").is_file()
    assert not (root / "memoryos" / "security" / "scope.py").exists()
    assert not (root / "memoryos" / "runtime" / "readiness.py").exists()
    assert (root / "memory" / "core" / "model" / "document.py").is_file()
    assert (root / "memory" / "core" / "structure" / "frontmatter.py").is_file()
    assert (root / "memory" / "core" / "write" / "router.py").is_file()
    assert (root / "memory" / "ports" / "document_store.py").is_file()
    assert (root / "memory" / "ports" / "erase.py").is_file()
    assert (root / "memory" / "ports" / "consolidation.py").is_file()
    assert (root / "memory" / "formation" / "llm" / "backend.py").is_file()
    assert (root / "memory" / "formation" / "llm" / "prompt.py").is_file()
    assert (root / "memory" / "formation" / "llm" / "validation.py").is_file()
    assert not (root / "memoryos" / "providers" / "memory_extractor").exists()
    assert not (root / "memoryos" / "providers" / "embedding").exists()
    assert not (root / "memoryos" / "providers").exists()
    assert (root / "infrastructure" / "model" / "client.py").is_file()
    assert (root / "infrastructure" / "model" / "factory.py").is_file()
    assert (root / "infrastructure" / "model" / "contracts.py").is_file()
    for path in (root / "memory" / "formation" / "llm").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "memory.execute" not in source
        assert "memory.commit" not in source
    assert (root / "memory" / "commit" / "document_commit.py").is_file()
    assert (root / "memory" / "commit" / "document_prepare.py").is_file()
    assert (root / "memory" / "commit" / "document_publication.py").is_file()
    assert (root / "memory" / "commit" / "document_recovery.py").is_file()
    assert (root / "memory" / "commit" / "erase_service.py").is_file()
    assert (root / "memory" / "commit" / "consolidation_service.py").is_file()
    assert (root / "memory" / "commit" / "session_consumers.py").is_file()
    assert (root / "memory" / "commit" / "session_recovery.py").is_file()
    assert (root / "memory" / "commit" / "session_support.py").is_file()
    assert (root / "infrastructure" / "store" / "memory" / "control_store.py").is_file()
    assert (root / "infrastructure" / "store" / "memory" / "erasure_store.py").is_file()
    assert (root / "infrastructure" / "store" / "memory" / "consolidation_store.py").is_file()
    assert (root / "infrastructure" / "store" / "memory" / "evidence" / "proposal_store.py").is_file()
    assert not (root / "memoryos" / "memory" / "evidence" / "proposal_store.py").exists()
    assert not (root / "memoryos" / "memory" / "evidence" / "salience_ledger.py").exists()
    assert (root / "infrastructure" / "context" / "projection" / "memory_document.py").is_file()


def test_shared_scope_model_does_not_depend_on_legacy_memoryos_package() -> None:
    """Evidence、Context 和 Store 必须共同依赖 Foundation 的唯一作用域模型。"""

    root = Path(__file__).resolve().parents[3]
    production_roots = (
        root / "pre",
        root / "foundation",
        root / "infrastructure",
        root / "memory",
        root / "memoryos",
        root / "openApi",
        root / "policy",
        root / "transaction",
    )
    violations = [
        path.relative_to(root).as_posix()
        for base in production_roots
        for path in base.rglob("*.py")
        if "memoryos.security.scope" in path.read_text(encoding="utf-8")
    ]
    assert violations == []
    assert not (root / "memoryos" / "security").exists()
    assert not (root / "memory" / "security").exists()
    assert not (root / "foundation" / "security").exists()


def test_behavior_and_action_policy_own_their_support_models() -> None:
    """支撑对象必须归属具体领域，旧 memoryos.support 不得恢复。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "support").exists()
    assert (root / "behavior" / "core" / "support" / "behavior_support.py").is_file()
    assert (root / "behavior" / "projection" / "behavior_support.py").is_file()
    assert (root / "policy" / "action_policy" / "model" / "policy_support_rule.py").is_file()
    assert (root / "policy" / "action_policy" / "update" / "policy_support_writer.py").is_file()

    production_roots = (root / "behavior", root / "policy", root / "infrastructure", root / "memory")
    legacy_imports = [
        path.relative_to(root).as_posix()
        for base in production_roots
        for path in base.rglob("*.py")
        if "memoryos.support" in path.read_text(encoding="utf-8")
    ]
    assert legacy_imports == []


def test_agent_hook_integration_has_one_root_owner() -> None:
    """Hook 的真实实现只能位于根目录 agent_hook，旧 adapter 路径不得复活。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "adapters" / "agent_hooks").exists()
    assert (root / "agent_hook" / "__init__.py").is_file()
    assert (root / "agent_hook" / "base.py").is_file()
    assert (root / "agent_hook" / "session_service.py").is_file()
    assert (root / "agent_hook" / "queue.py").is_file()


def test_in_memory_test_doubles_have_one_test_owner() -> None:
    """模拟持久化属于测试支持；正式进程锁位于基础设施存储层。"""

    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "adapters" / "persistence" / "in_memory").exists()
    assert not (root / "memoryos" / "contextdb" / "store" / "local_stores.py").exists()
    assert (root / "tests" / "support" / "persistence" / "in_memory" / "index_store.py").is_file()
    assert (root / "tests" / "support" / "persistence" / "in_memory" / "queue_store.py").is_file()
    assert (root / "tests" / "support" / "persistence" / "in_memory" / "relation_store.py").is_file()
    assert (root / "tests" / "support" / "persistence" / "in_memory" / "vector_store.py").is_file()
    assert (root / "infrastructure" / "store" / "locks" / "process_local.py").is_file()
    assert not (root / "infrastructure" / "store" / "vector").exists()


def test_store_infrastructure_has_one_root_owner() -> None:
    """具体存储实现只能位于 infrastructure/store，旧 adapters 路径必须消失。"""

    root = Path(__file__).resolve().parents[3]
    adapters = root / "memoryos" / "adapters"
    store = root / "infrastructure" / "store"
    assert not adapters.exists()
    assert not tuple((root / "memoryos" / "contextdb" / "store").glob("*.py"))
    assert (store / "filesystem" / "source_store.py").is_file()
    assert (store / "filesystem" / "source_bundle.py").is_file()
    assert (store / "filesystem" / "memory_document_store.py").is_file()
    assert (store / "filesystem" / "memory_document_io.py").is_file()
    assert (store / "filesystem" / "memory_document_scan.py").is_file()
    assert (store / "filesystem" / "session_archive.py").is_file()
    assert (store / "filesystem" / "session_archive_io.py").is_file()
    assert (store / "filesystem" / "session_archive_layout.py").is_file()
    assert (store / "filesystem" / "session_async_outputs.py").is_file()
    assert (store / "locks" / "process_local.py").is_file()
    assert not (store / "vector").exists()


def test_operation_plane_has_one_transaction_and_store_owner() -> None:
    """通用事务归根目录 transaction，控制文件持久化只属于 Store。"""

    root = Path(__file__).resolve().parents[3]
    transaction = root / "transaction"
    operation_store = root / "infrastructure" / "store" / "operation"
    assert not (root / "memoryos" / "operations").exists()
    assert (transaction / "commit" / "operation_committer.py").is_file()
    assert (transaction / "commit" / "domain_protocols.py").is_file()
    assert (transaction / "commit" / "effect_proof.py").is_file()
    assert (transaction / "resolver" / "target_resolver.py").is_file()
    assert len((transaction / "commit" / "operation_committer.py").read_text(encoding="utf-8").splitlines()) < 300
    transaction_source = "\n".join(path.read_text(encoding="utf-8") for path in transaction.rglob("*.py"))
    assert "policy.action_policy" not in transaction_source
    assert "infrastructure.context" not in transaction_source
    assert all(
        (operation_store / name).is_file()
        for name in {"audit.py", "control_stores.py", "diff.py", "marker.py", "redo.py"}
    )


def test_context_infrastructure_has_one_root_owner() -> None:
    """上下文检索、分层、维护和轨迹语义只允许位于 infrastructure/context。"""

    root = Path(__file__).resolve().parents[3]
    context = root / "infrastructure" / "context"
    assert not (root / "memoryos" / "contextdb").exists()
    assert not (root / "memoryos" / "application" / "context").exists()
    expected_modules = {
        "__init__.py",
        "exact_reader.py",
        "hydration.py",
        "orchestrator.py",
        "operation_target.py",
        "query_planner.py",
        "query_service.py",
        "query_support.py",
        "reranking.py",
        "selection.py",
    }
    assert all((context / name).is_file() for name in expected_modules)
    assert all(
        (context / name / "__init__.py").is_file()
        for name in {"candidate", "layers", "maintenance", "retrieval", "trace"}
    )
    assert (context / "trace" / "service.py").is_file()
    assert not (context / "trace" / "store.py").exists()
    assert not (context / "trace" / "erase.py").exists()
    assert not (context / "retrieval" / "errors.py").exists()
    assert not (context / "retrieval" / "lexical.py").exists()
    assert not (context / "assembler.py").exists()
    assert not (context / "candidate_generator.py").exists()
    assert not (context / "retrieval_service.py").exists()
    assert not (context / "trace_erase.py").exists()
    assert not (root / "openApi" / "limits.py").exists()
    for retired in (
        root / "memoryos" / "contextdb" / "retrieval",
        root / "memoryos" / "contextdb" / "layers",
        root / "memoryos" / "contextdb" / "maintenance",
    ):
        assert not tuple(retired.glob("*.py"))
    assert not (root / "memoryos" / "memory" / "documents" / "context_overlay.py").exists()
    assert not (root / "memoryos" / "observability").exists()
    assert not (root / "memoryos" / "contextdb" / "skill" / "skill_context_builder.py").exists()
    assert not tuple((root / "memoryos" / "contextdb" / "transaction").glob("*.py"))
    assert not tuple((root / "memoryos" / "contextdb" / "session").glob("*.py"))


def test_store_infrastructure_owns_persistence_implementations() -> None:
    """轨迹、SQLite、文件、向量和锁的持久化实现统一归 Store。"""

    store = Path(__file__).resolve().parents[3] / "infrastructure" / "store"
    assert (store / "query.py").is_file()
    assert all((store / "trace" / name).is_file() for name in {"__init__.py", "repository.py", "erase.py"})


def test_context_modules_remain_responsibility_sized() -> None:
    """防止候选、内容回源和编排职责再次聚合为超大文件。"""

    context = Path(__file__).resolve().parents[3] / "infrastructure" / "context"
    oversized = {
        path.relative_to(context).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in context.rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 600
    }
    assert oversized == {}


def test_store_modules_remain_responsibility_sized() -> None:
    """防止扫描、原子 I/O、归档发布等职责再次回流成超大单文件。"""

    root = Path(__file__).resolve().parents[3]
    store = root / "infrastructure" / "store"
    oversized = {
        path.relative_to(store).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in store.rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 600
    }
    assert oversized == {}


def test_memory_commit_modules_remain_responsibility_sized() -> None:
    """提交事务按准备、发布、恢复和消费者职责拆分，防止重新聚合。"""

    commit = Path(__file__).resolve().parents[3] / "memory" / "commit"
    oversized = {
        path.relative_to(commit).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in commit.rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 600
    }
    assert oversized == {}


def test_memory_file_stores_have_one_infrastructure_owner() -> None:
    """删除纪元和合并 Saga 的文件实现不得回流到提交层。"""

    root = Path(__file__).resolve().parents[3]
    commit_text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "memory" / "commit").rglob("*.py"))
    assert "class MemoryDocumentEraseStore" not in commit_text
    assert "class MemoryDocumentConsolidationStore" not in commit_text
    assert "MemoryDocumentEraseStore" not in (root / "memory" / "commit" / "__init__.py").read_text(encoding="utf-8")
    assert "MemoryDocumentConsolidationStore" not in (root / "memory" / "commit" / "__init__.py").read_text(
        encoding="utf-8"
    )


def test_legacy_paths_and_imports_do_not_return() -> None:
    root = Path(__file__).resolve().parents[3]
    forbidden_dirs = [
        root / "memoryos" / "domain",
        root / "memoryos" / "services",
        root / "memoryos" / "usecases" / "episode",
        root / "memoryos" / "usecases" / "feedback",
        root / "memoryos" / "usecases" / "session",
        root / "memoryos" / "interfaces",
        root / "memoryos" / "ports",
        root / "architecture",
    ]
    assert all(not any(path.rglob("*.py")) for path in forbidden_dirs)

    forbidden = [
        "".join(["Episode", "Processor"]),
        ".".join(["memoryos", "domain"]),
        ".".join(["memoryos", "services"]),
        ".".join(["memoryos", "usecases", "episode"]),
        "".join(["pending", "_extraction"]),
    ]
    for base in (
        root / "memoryos",
        root / "memory",
        root / "behavior",
        root / "openApi",
        root / "policy",
        root / "agent_hook",
        root / "infrastructure",
        root / "tests",
    ):
        for path in base.rglob("*.py"):
            if path.name == "test_no_legacy_paths.py":
                continue
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden), path

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert " ".join(["Personal", "Memory", "OS"]) not in readme
    assert "第一阶段只解决记忆" not in readme
    assert "".join(["Episode", "Processor"]) not in readme


def test_markdown_memory_has_one_source_domain_and_no_retired_python_packages() -> None:
    root = Path(__file__).resolve().parents[3]
    memory_root = root / "memoryos" / "memory"
    retired_dirs = (
        memory_root / "canonical",
        memory_root / "integration",
        memory_root / "lifecycle",
        memory_root / "model",
        memory_root / "service",
        memory_root / "store",
    )
    # 这些包必须彻底退出，而不是只删源码。只剩字节码的目录仍可能作为
    # 命名空间包被导入，从而悄悄保留已经废弃的公开路径。
    assert all(not path.exists() for path in retired_dirs)
    assert not memory_root.exists()
    assert any((root / "memory" / "core").rglob("*.py"))
    assert any((root / "memory" / "commit").glob("*.py"))
    assert any((root / "infrastructure" / "store" / "memory").glob("*.py"))
    assert not (root / "evidence").exists()
    assert any((root / "pre" / "evidence" / "model").glob("*.py"))
    assert (root / "memory" / "core" / "formation" / "signals.py").is_file()
    assert (root / "memory" / "formation" / "llm" / "backend.py").is_file()
    assert not (root / "memoryos" / "providers" / "memory_extractor").exists()

    forbidden_symbols = (
        "Memory" + "Slot",
        "Memory" + "Claim",
        "Current" + "Head",
        "Bounded" + "Canonical" + "Resolver",
    )
    for base in (
        root / "memoryos",
        root / "memory",
        root / "behavior",
        root / "openApi",
        root / "policy",
        root / "agent_hook",
        root / "infrastructure",
        root / "tests",
    ):
        for path in base.rglob("*.py"):
            if path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            assert not any(symbol in text for symbol in forbidden_symbols), path

    removed_files = (
        root / "memoryos" / "memory" / "merge.py",
        root / "memoryos" / "memory" / "extraction" / "llm_memory_extractor.py",
        root / "memoryos" / "memory" / "extraction" / "rule_memory_extractor.py",
        root / "memoryos" / "memory" / "update" / "__init__.py",
    )
    assert all(not path.is_file() for path in removed_files)

    behavior_case = (root / "behavior" / "core" / "model" / "behavior_case.py").read_text(encoding="utf-8")
    assert "related_" + "memory_uris" not in behavior_case
