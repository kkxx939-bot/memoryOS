from __future__ import annotations

import json
import subprocess
import sys


def test_domain_package_imports_do_not_register_commit_or_evidence_handlers() -> None:
    code = """
import json

from memoryos.contextdb.session.evidence_encoder import session_evidence_encoder
from memoryos.operations.commit.domain_registry import (
    action_policy_commit_handlers,
    memory_commit_handlers,
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
    "memory_handlers": memory_commit_handlers() is not None,
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
        "memory_handlers": False,
    }


def test_runtime_build_explicitly_registers_domain_handlers_and_encoder() -> None:
    code = """
import json
import tempfile

from memoryos.contextdb.session.evidence_encoder import session_evidence_encoder
from memoryos.operations.commit.domain_registry import (
    action_policy_commit_handlers,
    memory_commit_handlers,
)
from memoryos.runtime.config import RuntimeConfig
from memoryos.runtime.container import build_runtime_container

before = {
    "action_policy_handlers": action_policy_commit_handlers() is not None,
    "memory_handlers": memory_commit_handlers() is not None,
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
    "memory_handlers": memory_commit_handlers() is not None,
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
            "memory_handlers": True,
        },
        "before": {
            "action_policy_handlers": False,
            "encoder": False,
            "memory_handlers": False,
        },
    }
