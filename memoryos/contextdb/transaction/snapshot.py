from __future__ import annotations

from dataclasses import dataclass

from memoryos.core.ids import new_id
from memoryos.core.time import utc_now


@dataclass(frozen=True)
class SnapshotVersion:
    snapshot_id: str
    created_at: str

    @classmethod
    def create(cls) -> SnapshotVersion:
        return cls(snapshot_id=new_id("snapshot"), created_at=utc_now())
