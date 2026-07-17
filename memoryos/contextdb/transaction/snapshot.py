"""上下文数据库里的快照。"""

from __future__ import annotations

from dataclasses import dataclass

from memoryos.core.clock import utc_now
from memoryos.core.ids import new_id


@dataclass(frozen=True)
class SnapshotVersion:
    snapshot_id: str
    created_at: str

    @classmethod
    def create(cls) -> SnapshotVersion:
        return cls(snapshot_id=new_id("snapshot"), created_at=utc_now())
