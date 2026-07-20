from __future__ import annotations

import json
import subprocess
import sys
from inspect import getsource
from pathlib import Path

from tests.support.import_graph import production_imports

ROOT = Path(__file__).resolve().parents[3]


def test_only_composition_and_delivery_layers_import_runtime() -> None:
    """领域模块不能反向依赖进程组合根。"""

    allowed_roots = {"runtime", "openApi"}
    violations = []
    for edge in production_imports(ROOT):
        if edge.target != "runtime" and not edge.target.startswith("runtime."):
            continue
        source_root = edge.source.relative_to(ROOT).parts[0]
        if source_root not in allowed_roots:
            violations.append(f"{edge.source.relative_to(ROOT)}:{edge.line} -> {edge.target}")
    assert violations == []


def test_domain_package_imports_do_not_trigger_hidden_runtime_registration() -> None:
    code = """
import json
import sys

import policy.action_policy
import pre.evidence
import memory.core

print(json.dumps({
    "commit_registration_loaded": "policy.action_policy.integration.commit_registration" in sys.modules,
    "legacy_encoder_module_loaded": "memory.commit.evidence.encoder" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "commit_registration_loaded": False,
        "legacy_encoder_module_loaded": False,
    }


def test_runtime_build_explicitly_injects_domain_handlers_and_archive_encoder() -> None:
    code = """
import json
import tempfile

from runtime import RuntimeBuilder, RuntimeConfig

with tempfile.TemporaryDirectory() as root:
    container = RuntimeBuilder(RuntimeConfig(root=root)).build()
    container.start()
    handlers_injected = bool(container.transaction.committer.domain_extensions.handlers)
    encoder_name = type(container.session.archive_store.evidence_encoder).__name__

after = {
    "domain_extensions": handlers_injected,
    "encoder": encoder_name,
}
print(json.dumps({"after": after}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "after": {
            "domain_extensions": True,
            "encoder": "SessionEvidenceArchiveEncoder",
        },
    }


def test_runtime_calls_consolidation_recovery_before_projection_drain(tmp_path) -> None:  # noqa: ANN001
    from runtime import RuntimeBuilder, RuntimeConfig
    from runtime.recovery.coordinator import RuntimeRecoveryCoordinator

    container = RuntimeBuilder(RuntimeConfig(root=str(tmp_path))).build()
    container.start()

    assert container.memory.command_service.consolidator is container.memory.consolidator
    assert container.memory.consolidator.saga_store is container.memory.consolidation_store
    recovery_source = getsource(RuntimeRecoveryCoordinator.recover)
    intent_recovery = recovery_source.index('details["document_intents"]')
    saga_recovery = recovery_source.index('details["memory_consolidations_pre_projection"]')
    projection_drain = recovery_source.index('details["memory_projection_queue"]')
    assert intent_recovery < saga_recovery < projection_drain
