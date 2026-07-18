from __future__ import annotations

import json
import subprocess
import sys
from inspect import getsource


def test_domain_package_imports_do_not_register_commit_or_evidence_handlers() -> None:
    code = """
import json

from memoryos.contextdb.session.evidence_encoder import session_evidence_encoder
from memoryos.operations.commit.domain_registry import (
    action_policy_commit_handlers,
)

import memoryos.action_policy
import memoryos.memory

try:
    session_evidence_encoder()
except RuntimeError as exc:
    encoder_error = str(exc)
else:
    encoder_error = ""

print(json.dumps({
    "action_policy_handlers": action_policy_commit_handlers() is not None,
    "encoder_error": encoder_error,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "action_policy_handlers": False,
        "encoder_error": "Session evidence encoder is not registered",
    }


def test_runtime_build_explicitly_registers_domain_handlers_and_encoder() -> None:
    code = """
import json
import tempfile

from memoryos.contextdb.session.evidence_encoder import session_evidence_encoder
from memoryos.operations.commit.domain_registry import (
    action_policy_commit_handlers,
)
from memoryos.runtime.config import RuntimeConfig
from memoryos.runtime.container import build_runtime_container

before = {
    "action_policy_handlers": action_policy_commit_handlers() is not None,
}
try:
    session_evidence_encoder()
except RuntimeError:
    before["encoder"] = False
else:
    before["encoder"] = True

with tempfile.TemporaryDirectory() as root:
    build_runtime_container(RuntimeConfig(root=root))

after = {
    "action_policy_handlers": action_policy_commit_handlers() is not None,
    "encoder": session_evidence_encoder() is not None,
}
print(json.dumps({"after": after, "before": before}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "after": {
            "action_policy_handlers": True,
            "encoder": True,
        },
        "before": {
            "action_policy_handlers": False,
            "encoder": False,
        },
    }


def test_runtime_calls_consolidation_recovery_before_projection_drain(tmp_path) -> None:  # noqa: ANN001
    from memoryos.runtime.config import RuntimeConfig
    from memoryos.runtime.container import _recover_runtime, build_runtime_container

    container = build_runtime_container(RuntimeConfig(root=str(tmp_path)))

    assert container.memory_command_service.consolidator is container.memory_document_consolidator
    assert container.memory_document_consolidator.saga_store is container.memory_document_consolidation_store
    recovery_source = getsource(_recover_runtime)
    intent_recovery = recovery_source.index('details["document_intents"]')
    saga_recovery = recovery_source.index('details["memory_consolidations_pre_projection"]')
    projection_drain = recovery_source.index('details["memory_projection_queue"]')
    assert intent_recovery < saga_recovery < projection_drain
