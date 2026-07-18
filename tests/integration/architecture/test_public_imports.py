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


def test_root_import_does_not_load_delivery_persistence_or_worker_graph() -> None:
    loaded = _loaded_modules(
        """
import json
import sys
import memoryos

blocked = (
    "memoryos.adapters.persistence.sqlite",
    "memoryos.api.http",
    "memoryos.api.mcp",
    "memoryos.api.sdk",
    "memoryos.contextdb.store.sqlite",
    "memoryos.workers",
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
import memoryos.contextdb.context_db

print(json.dumps(sorted(name for name in sys.modules if name.startswith("memoryos.operations"))))
"""
    )
    assert loaded == []


def test_prediction_pipeline_package_does_not_eagerly_load_execution() -> None:
    loaded = _loaded_modules(
        """
import json
import sys
import memoryos.prediction.pipeline

print(json.dumps(sorted(name for name in sys.modules if name.startswith("memoryos.execution"))))
"""
    )
    assert loaded == []


def test_root_public_imports_resolve_current_objects() -> None:
    import memoryos
    from memoryos.api.sdk import MemoryOSClient as PackageClient
    from memoryos.api.sdk.client import MemoryOSClient
    from memoryos.contextdb.context_db import ContextDB
    from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan
    from memoryos.prediction.model.prediction_request import PredictionRequest

    assert memoryos.MemoryOSClient is PackageClient is MemoryOSClient
    assert memoryos.ContextDB is ContextDB
    assert memoryos.RetrievalOptions is RetrievalOptions
    assert memoryos.RetrievalQueryPlan is RetrievalQueryPlan
    assert memoryos.PredictionRequest is PredictionRequest
    assert set(memoryos.__all__) == {
        "__version__",
        "ActionCandidate",
        "ActionPolicy",
        "ContextDB",
        "MemoryOSClient",
        "PredictionRequest",
        "RetrievalOptions",
        "RetrievalQueryPlan",
    }
