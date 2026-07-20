"""Build and query a 100k-row Unified Context Catalog benchmark.

This is an explicit/offline benchmark, not part of normal pytest collection.
It writes only rebuildable serving projections and reports candidate bounds and
SQLite's selected structured query plan. Example:

    python tests/benchmark/unified_context_100k.py --records 100000
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind
from infrastructure.store.sqlite.index_store import SQLiteIndexStore


def _record(index: int) -> CatalogRecord:
    session_id = f"session-{index // 10:05d}"
    timestamp = "2026-07-14T03:30:00+00:00"
    name = f"benchmark-{index:06d}.txt"
    return CatalogRecord(
        record_key=f"session:{session_id}:tool:{index:06d}",
        uri=f"memoryos://user/benchmark/sessions/history/{session_id}/context/tool/{index:06d}",
        tenant_id="benchmark",
        owner_user_id="benchmark-user",
        workspace_id="memoryOS",
        session_id=session_id,
        adapter_id="benchmark-agent",
        context_type="session",
        source_kind="tool_result",
        record_kind=CatalogRecordKind.TOOL_RESULT.value,
        tree_paths=("timeline/2026/07/14", f"sessions/{session_id}", "resources/repository"),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title=name,
        l0_text=f"Benchmark tool result {name}",
        l1_text=f"benchmarkneedle{index:06d} bounded payload",
        source_uri=f"memoryos://user/benchmark/sessions/history/{session_id}",
        source_digest=f"benchmark-digest-{index:06d}",
        metadata={"resource_name": name, "resource_location": "repository", "vector_eligible": False},
    )


def run(path: Path, *, records: int, batch_size: int, candidate_limit: int) -> dict[str, object]:
    store = SQLiteIndexStore(path)
    started = time.monotonic()
    for start in range(0, records, batch_size):
        batch = tuple(_record(index) for index in range(start, min(records, start + batch_size)))
        store.upsert_catalog_batch(batch, tenant_id="benchmark")
    ingest_seconds = time.monotonic() - started

    filters = {
        "tenant_id": "benchmark",
        "principal_owner_id": "benchmark-user",
        "workspace_access_ids": ("", "memoryOS"),
        "context_types": ("session",),
        "source_kinds": ("tool_result",),
        "target_paths": ("resources/repository",),
        "event_time_from": "2026-07-14T00:00:00+00:00",
        "event_time_to": "2026-07-15T00:00:00+00:00",
    }
    query_started = time.monotonic()
    hits = store.search_catalog(
        f"benchmarkneedle{records - 1:06d}",
        tenant_id="benchmark",
        filters=filters,
        limit=candidate_limit,
    )
    query_seconds = time.monotonic() - query_started
    return {
        "database": str(path),
        "records": records,
        "batch_size": batch_size,
        "candidate_limit": candidate_limit,
        "returned_candidates": len(hits),
        "first_record_key": str(hits[0].metadata.get("catalog_record_key") or "") if hits else "",
        "candidate_bound_satisfied": len(hits) <= candidate_limit,
        "ingest_seconds": round(ingest_seconds, 6),
        "query_seconds": round(query_seconds, 6),
        "query_plan": store.explain_structured_query(
            tenant_id="benchmark",
            filters=filters,
            limit=candidate_limit,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.records < 1 or args.records > 1_000_000:
        raise SystemExit("--records must be between 1 and 1000000")
    if args.batch_size < 1 or args.batch_size > 10_000:
        raise SystemExit("--batch-size must be between 1 and 10000")
    if args.candidate_limit < 1 or args.candidate_limit > 1_000:
        raise SystemExit("--candidate-limit must be between 1 and 1000")

    if args.database is not None:
        result = run(
            args.database,
            records=args.records,
            batch_size=args.batch_size,
            candidate_limit=args.candidate_limit,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="memoryos-unified-context-benchmark-") as root:
            result = run(
                Path(root) / "catalog.sqlite3",
                records=args.records,
                batch_size=args.batch_size,
                candidate_limit=args.candidate_limit,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
