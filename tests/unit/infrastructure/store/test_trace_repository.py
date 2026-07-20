"""召回轨迹文件仓库的保留策略测试。"""

from __future__ import annotations

import os
import uuid

from infrastructure.store.trace import RecallTraceRepository


def _save(repository: RecallTraceRepository, marker: str) -> str:
    trace_id = str(uuid.uuid4())
    repository.save(trace_id, {"trace_id": trace_id, "marker": marker})
    return trace_id


def test_recall_trace_retention_enforces_file_count(tmp_path) -> None:  # noqa: ANN001
    repository = RecallTraceRepository(tmp_path / "recall-traces")
    for index in range(3):
        _save(repository, f"trace-{index}")

    result = repository.prune(
        max_age_seconds=10_000,
        max_files=1,
        max_total_bytes=10_000_000,
        now_epoch=0,
    )

    assert result["scanned"] == 3
    assert result["deleted"] == 2
    assert result["retained"] == 1
    assert len(tuple(repository.trace_root.glob("*.json"))) == 1


def test_recall_trace_retention_enforces_age_and_total_bytes(tmp_path) -> None:  # noqa: ANN001
    repository = RecallTraceRepository(tmp_path / "recall-traces")
    expired_id = _save(repository, "expired")
    current_id = _save(repository, "current")
    expired_path = repository.trace_root / f"{expired_id}.json"
    current_path = repository.trace_root / f"{current_id}.json"
    os.utime(expired_path, (100.0, 100.0))
    os.utime(current_path, (1_000.0, 1_000.0))

    age_result = repository.prune(
        max_age_seconds=500,
        max_files=10,
        max_total_bytes=10_000_000,
        now_epoch=1_000.0,
    )

    assert age_result["deleted"] == 1
    assert not expired_path.exists()
    assert current_path.exists()

    bytes_result = repository.prune(
        max_age_seconds=10_000,
        max_files=10,
        max_total_bytes=0,
        now_epoch=1_000.0,
    )

    assert bytes_result["deleted"] == 1
    assert bytes_result["retained_bytes"] == 0
    assert not current_path.exists()
