"""恢复尚未完成的跨文档记忆合并 Saga。"""

from __future__ import annotations

from typing import Any

from memory.commit import MemoryDocumentConsolidator


def recover_memory_consolidations(
    consolidator: MemoryDocumentConsolidator,
    *,
    tenant_id: str,
    owners: tuple[str, ...],
) -> dict[str, Any]:
    """逐 Owner 恢复合并 Saga，并汇总可观测计数。"""

    per_owner: dict[str, dict[str, object]] = {}
    totals = {
        "examined": 0,
        "completed": 0,
        "awaiting_projection": 0,
        "awaiting_input": 0,
    }
    for owner in owners:
        outcome = consolidator.resume_all(
            tenant_id=tenant_id,
            owner_user_id=owner,
            limit=1_000,
        ).to_dict()
        per_owner[owner] = outcome
        for key in totals:
            value = outcome[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("consolidation recovery report contains a non-integer count")
            totals[key] += value
    return {**totals, "owners": per_owner}


__all__ = ["recover_memory_consolidations"]
