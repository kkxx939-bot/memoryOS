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
import memoryos.contextdb.store

watched = (
    "memoryos.adapters.persistence",
    "memoryos.contextdb.store.index_consistency",
    "memoryos.contextdb.store.local_stores",
)
print(json.dumps({"loaded": sorted(name for name in sys.modules if name.startswith(watched))}))
"""
    )
    assert probe == {"loaded": []}


def test_operation_committer_import_is_independent_of_store_import_order() -> None:
    probe = _run_import_probe(
        """
import json
import memoryos.contextdb.store
from memoryos.operations.commit.operation_committer import OperationCommitter

print(json.dumps({"committer": OperationCommitter.__name__}))
"""
    )
    assert probe == {"committer": "OperationCommitter"}


def test_historical_sqlite_exports_resolve_to_canonical_adapters() -> None:
    from memoryos.adapters.persistence.sqlite import (
        SQLiteIndexStore,
        SQLiteLockStore,
        SQLiteQueueStore,
        SQLiteRelationStore,
    )
    from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore as HistoricalIndexStore
    from memoryos.contextdb.store.sqlite_lock_store import SQLiteLockStore as HistoricalLockStore
    from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore as HistoricalQueueStore
    from memoryos.contextdb.store.sqlite_relation_store import (
        SQLiteRelationStore as HistoricalRelationStore,
    )

    assert HistoricalIndexStore is SQLiteIndexStore
    assert HistoricalLockStore is SQLiteLockStore
    assert HistoricalQueueStore is SQLiteQueueStore
    assert HistoricalRelationStore is SQLiteRelationStore
