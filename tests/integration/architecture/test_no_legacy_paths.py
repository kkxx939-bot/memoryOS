from __future__ import annotations

from pathlib import Path


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
    # These packages are retired, not merely empty.  Leaving a directory that
    # contains only bytecode makes it importable as a namespace package and
    # silently preserves an obsolete public path.
    assert all(not path.exists() for path in retired_dirs)
    assert any((memory_root / "documents").glob("*.py"))
    assert any((memory_root / "evidence").glob("*.py"))

    forbidden_symbols = (
        "Memory" + "Slot",
        "Memory" + "Claim",
        "Current" + "Head",
        "Bounded" + "Canonical" + "Resolver",
    )
    for base in (root / "memoryos", root / "tests"):
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

    behavior_case = (root / "memoryos" / "behavior" / "model" / "behavior_case.py").read_text(
        encoding="utf-8"
    )
    assert "related_" + "memory_uris" not in behavior_case
