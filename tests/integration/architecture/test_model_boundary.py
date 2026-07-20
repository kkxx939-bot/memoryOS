"""通用模型基础设施与领域模型调用端口之间的依赖约束。"""

from __future__ import annotations

from pathlib import Path

from tests.support.import_graph import production_imports

ROOT = Path(__file__).resolve().parents[3]


def test_domain_logic_does_not_import_concrete_model_client() -> None:
    domain_prefixes = (
        "memory/core/",
        "memory/formation/",
        "memory/execute/",
        "behavior/core/",
        "policy/action_policy/",
    )
    violations = [
        f"{edge.source.relative_to(ROOT)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if edge.source.relative_to(ROOT).as_posix().startswith(domain_prefixes)
        and (edge.target == "infrastructure.model" or edge.target.startswith("infrastructure.model."))
    ]
    assert violations == []


def test_model_infrastructure_does_not_import_domain_logic() -> None:
    violations = [
        f"{edge.source.relative_to(ROOT)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if edge.source.relative_to(ROOT).as_posix().startswith("infrastructure/model/")
        and edge.target.split(".", 1)[0] in {"behavior", "memory", "policy"}
    ]
    assert violations == []


def test_legacy_provider_package_is_removed() -> None:
    assert not (ROOT / "memoryos" / "providers").exists()
    assert (ROOT / "infrastructure" / "model" / "providers" / "openai_compatible.py").is_file()
