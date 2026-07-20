from __future__ import annotations

from pathlib import Path

from tests.support.import_graph import production_imports, production_paths

ROOT = Path(__file__).resolve().parents[3]


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_module(target: str, modules: set[str]) -> str | None:
    candidate = target
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _eager_components() -> list[tuple[str, ...]]:
    paths = production_paths(ROOT)
    path_modules = {path: _module_name(path) for path in paths}
    modules = set(path_modules.values())
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for edge in production_imports(ROOT):
        if edge.kind != "eager":
            continue
        target = _resolve_module(edge.target, modules)
        source = path_modules[edge.source]
        if target is not None and target != source:
            graph[source].add(target)

    index = 0
    stack: list[str] = []
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    active: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indexes[module] = lowlinks[module] = index
        index += 1
        stack.append(module)
        active.add(module)
        for target in graph[module]:
            if target not in indexes:
                visit(target)
                lowlinks[module] = min(lowlinks[module], lowlinks[target])
            elif target in active:
                lowlinks[module] = min(lowlinks[module], indexes[target])
        if lowlinks[module] != indexes[module]:
            return
        component: list[str] = []
        while True:
            target = stack.pop()
            active.remove(target)
            component.append(target)
            if target == module:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for module in sorted(modules):
        if module not in indexes:
            visit(module)
    return sorted(components)


def test_production_has_no_eager_module_initialization_cycle() -> None:
    assert _eager_components() == []
