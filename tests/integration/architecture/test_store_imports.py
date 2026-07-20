from __future__ import annotations

import json
import subprocess
import sys


def _run_import_probe(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(json.loads(result.stdout))


def test_store_package_does_not_eagerly_load_implementations() -> None:
    probe = _run_import_probe(
        """
import json
import sys
import infrastructure.store.contracts

watched = (
    "infrastructure.store.sqlite",
    "infrastructure.store.locks.process_local",
    "infrastructure.context.maintenance.index_consistency",
)
print(json.dumps({"loaded": sorted(name for name in sys.modules if name.startswith(watched))}))
"""
    )
    assert probe == {"loaded": []}


def test_operation_committer_import_is_independent_of_store_import_order() -> None:
    probe = _run_import_probe(
        """
import json
import infrastructure.store.contracts
from transaction.commit.operation_committer import OperationCommitter

print(json.dumps({"committer": OperationCommitter.__name__}))
"""
    )
    assert probe == {"committer": "OperationCommitter"}


def test_sqlite_package_exports_resolve_to_canonical_implementations() -> None:
    from infrastructure.store.sqlite import (
        SQLiteIndexStore,
        SQLiteLockStore,
        SQLiteQueueStore,
        SQLiteRelationStore,
    )
    from infrastructure.store.sqlite.index_store import SQLiteIndexStore as ConcreteIndexStore
    from infrastructure.store.sqlite.lock_store import SQLiteLockStore as ConcreteLockStore
    from infrastructure.store.sqlite.queue_store import SQLiteQueueStore as ConcreteQueueStore
    from infrastructure.store.sqlite.relation_store import (
        SQLiteRelationStore as ConcreteRelationStore,
    )

    assert ConcreteIndexStore is SQLiteIndexStore
    assert ConcreteLockStore is SQLiteLockStore
    assert ConcreteQueueStore is SQLiteQueueStore
    assert ConcreteRelationStore is SQLiteRelationStore


def test_trace_package_exports_storage_implementations_only() -> None:
    from infrastructure.store.trace import (
        RecallTraceEraseBackend,
        RecallTraceRepository,
    )
    from infrastructure.store.trace.erase import RecallTraceEraseBackend as ConcreteEraseBackend
    from infrastructure.store.trace.repository import RecallTraceRepository as ConcreteRepository

    assert ConcreteEraseBackend is RecallTraceEraseBackend
    assert ConcreteRepository is RecallTraceRepository
