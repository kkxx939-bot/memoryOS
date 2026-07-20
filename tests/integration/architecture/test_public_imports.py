from __future__ import annotations

import json
import subprocess
import sys


def _loaded_modules(code: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(json.loads(result.stdout))


def test_openapi_root_import_does_not_load_sdk_persistence_or_worker_graph() -> None:
    loaded = _loaded_modules(
        """
import json
import sys
import openApi

blocked = (
    "infrastructure.store.sqlite",
    "openApi.http",
    "openApi.mcp",
    "openApi.sdk",
    "runtime.worker",
)
print(json.dumps(sorted(name for name in sys.modules if name.startswith(blocked))))
"""
    )
    assert loaded == []


def test_contextdb_facade_import_does_not_load_operations_graph() -> None:
    loaded = _loaded_modules(
        """
import json
import sys
import infrastructure.context.facade

print(json.dumps(sorted(name for name in sys.modules if name.startswith("transaction"))))
"""
    )
    assert loaded == []


def test_action_policy_decision_package_does_not_eagerly_load_execution() -> None:
    loaded = _loaded_modules(
        """
import json
import sys
import policy.action_policy.decision

print(json.dumps(sorted(name for name in sys.modules if name.startswith("policy.action_policy.execution"))))
"""
    )
    assert loaded == []


def test_openapi_public_imports_resolve_current_objects() -> None:
    import openApi
    import openApi.sdk as sdk
    from openApi.sdk.client import MemoryOSClient
    from policy.action_policy.decision.request import PredictionRequest
    from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy

    assert openApi.__version__ == "0.1.0"
    assert sdk.MemoryOSClient is MemoryOSClient
    assert sdk.PredictionRequest is PredictionRequest
    assert sdk.ActionCandidate is ActionCandidate
    assert sdk.ActionPolicy is ActionPolicy
    assert set(sdk.__all__) == {
        "ActionCandidate",
        "ActionPolicy",
        "HTTPMemoryOSClient",
        "LocalMemoryOSClient",
        "MemoryOSClient",
        "ProcessObservationResult",
        "PredictionRequest",
        "RetrievalOptions",
        "RetrievalQueryPlan",
    }
