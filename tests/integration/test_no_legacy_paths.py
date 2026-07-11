from __future__ import annotations

from pathlib import Path


def test_legacy_paths_and_imports_do_not_return() -> None:
    root = Path(__file__).resolve().parents[2]
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
    assert all(not path.exists() for path in forbidden_dirs)

    forbidden = [
        "".join(["Episode", "Processor"]),
        ".".join(["memoryos", "domain"]),
        ".".join(["memoryos", "services"]),
        ".".join(["memoryos", "usecases", "episode"]),
        "".join(["pending", "_extraction"]),
    ]
    for base in (root / "memoryos", root / "tests"):
        for path in base.rglob("*.py"):
            if path.name == "test_no_legacy_paths.py":
                continue
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden), path

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert " ".join(["Personal", "Memory", "OS"]) not in readme
    assert "第一阶段只解决记忆" not in readme
    assert "".join(["Episode", "Processor"]) not in readme


def test_memory_v2_has_no_v1_execution_or_compatibility_surface() -> None:
    import memoryos.memory.canonical as canonical
    from memoryos.contextdb.session.planning import MemoryPlanningResult
    from memoryos.contextdb.session.session_archive import SessionArchiveStore

    root = Path(__file__).resolve().parents[2]
    removed_exports = (
        "IDENTITY_ALGORITHM_V1",
        "IdentityAliasOperation",
        "IdentityMigrationRequired",
        "LegacyIdentityCandidate",
        "LegacyCandidateProposalAdapter",
    )
    assert all(not hasattr(canonical, name) for name in removed_exports)
    assert not hasattr(SessionArchiveStore, "migrate_legacy_archive")
    assert not hasattr(MemoryPlanningResult, "to_list")
    assert not hasattr(MemoryPlanningResult, "__getitem__")
    removed_files = (
        root / "memoryos" / "memory" / "merge.py",
        root / "memoryos" / "memory" / "extraction" / "llm_memory_extractor.py",
        root / "memoryos" / "memory" / "extraction" / "rule_memory_extractor.py",
        root / "memoryos" / "memory" / "update" / "__init__.py",
    )
    assert all(not path.exists() for path in removed_files)
