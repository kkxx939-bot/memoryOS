from __future__ import annotations

import ast
from pathlib import Path

from tests.support.import_graph import production_paths

ROOT = Path(__file__).resolve().parents[3]


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_test_modules_do_not_import_other_test_modules() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        for module in _imports(path):
            if any(part.startswith("test_") for part in module.split(".")):
                violations.append(f"{path.relative_to(ROOT)} -> {module}")
    assert violations == []


def test_production_modules_do_not_import_test_support() -> None:
    violations = [
        f"{path.relative_to(ROOT)} -> {module}"
        for path in production_paths(ROOT)
        for module in _imports(path)
        if module == "tests" or module.startswith("tests.")
    ]
    assert violations == []
