"""AST import graph helpers used by architecture tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ImportKind = Literal["eager", "type_checking", "delayed"]
PRODUCTION_PACKAGES = (
    "pre",
    "memory",
    "behavior",
    "openApi",
    "policy",
    "agent_hook",
    "foundation",
    "infrastructure",
    "runtime",
    "sanitization",
    "transaction",
)


@dataclass(frozen=True)
class ImportEdge:
    source: Path
    target: str
    kind: ImportKind
    line: int


def module_imports(path: Path) -> tuple[ImportEdge, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collector = _ImportCollector(path)
    collector.visit(tree)
    return tuple(collector.edges)


def production_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for package in PRODUCTION_PACKAGES
        for path in sorted((root / package).rglob("*.py"))
    )


def production_imports(root: Path) -> tuple[ImportEdge, ...]:
    return tuple(
        edge
        for path in production_paths(root)
        for edge in module_imports(path)
        if edge.target in PRODUCTION_PACKAGES
        or any(edge.target.startswith(f"{package}.") for package in PRODUCTION_PACKAGES)
    )


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.edges: list[ImportEdge] = []
        self.function_depth = 0
        self.type_checking_depth = 0

    @property
    def kind(self) -> ImportKind:
        if self.type_checking_depth:
            return "type_checking"
        if self.function_depth:
            return "delayed"
        return "eager"

    def visit_Import(self, node: ast.Import) -> None:
        self.edges.extend(
            ImportEdge(self.path, alias.name, self.kind, node.lineno)
            for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.edges.extend(
            ImportEdge(self.path, target, self.kind, node.lineno)
            for target in self._import_from_targets(node)
            if target
        )

    def _import_from_targets(self, node: ast.ImportFrom) -> tuple[str, ...]:
        if not node.level:
            return (node.module,) if node.module else ()
        parts = list(self.path.with_suffix("").parts)
        package_start = next(
            (parts.index(package) for package in PRODUCTION_PACKAGES if package in parts),
            None,
        )
        if package_start is None:
            return (node.module,) if node.module else ()
        module_parts = parts[package_start:]
        package = module_parts if module_parts[-1] == "__init__" else module_parts[:-1]
        if package and package[-1] == "__init__":
            package = package[:-1]
        ascend = node.level - 1
        if ascend > len(package):
            return (node.module,) if node.module else ()
        base = package[: len(package) - ascend]
        if node.module:
            return (".".join([*base, *node.module.split(".")]),)
        return tuple(".".join([*base, alias.name]) for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        guarded = _is_type_checking_guard(node.test)
        if guarded:
            self.type_checking_depth += 1
        for statement in node.body:
            self.visit(statement)
        if guarded:
            self.type_checking_depth -= 1
        for statement in node.orelse:
            self.visit(statement)


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


__all__ = [
    "ImportEdge",
    "ImportKind",
    "PRODUCTION_PACKAGES",
    "module_imports",
    "production_imports",
    "production_paths",
]
