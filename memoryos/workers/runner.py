"""后台任务里的启动器。"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.core.time import utc_now
from memoryos.workers.memory_proposal_worker import MemoryProposalWorker
from memoryos.workers.session_commit_worker import SessionCommitWorker


class WorkerRunner:
    def __init__(self, client: MemoryOSClient, *, poll_interval: float = 1.0, batch_size: int = 10, lease_seconds: int = 60, max_retries: int = 3) -> None:
        self.client = client
        self.poll_interval = max(0.05, poll_interval)
        self.batch_size = max(1, batch_size)
        self.lease_seconds = max(1, lease_seconds)
        self.max_retries = max(1, max_retries)
        self.stopping = False
        self.heartbeat = Path(client.root) / "system" / "worker-health.json"

    def run(self, kind: str, *, once: bool = False) -> dict[str, Any]:
        self._install_signal_handlers()
        latest: dict[str, Any] = {}
        while not self.stopping:
            latest = self.run_once(kind)
            self._write_heartbeat(kind, latest)
            if once:
                break
            time.sleep(self.poll_interval)
        return latest

    def run_once(self, kind: str) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": kind, "timestamp": utc_now()}
        if kind in {"session-commit", "all"}:
            result["session_commit"] = SessionCommitWorker(self.client.session_commit_service).process_pending(
                batch_size=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
        if kind in {"memory-projection", "all"}:
            result["memory_projection"] = self.client.memory_projection_worker.process_pending(limit=self.batch_size)
        if kind in {"memory-proposal", "all"}:
            result["memory_proposal"] = MemoryProposalWorker(self.client.session_commit_service).process_pending(
                batch_size=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
        if kind in {"maintenance", "all"}:
            result["maintenance"] = self.client.context_db.verify_consistency()
        return result

    def _install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: Any) -> None:
            self.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def _write_heartbeat(self, kind: str, result: dict[str, Any]) -> None:
        self.heartbeat.parent.mkdir(parents=True, exist_ok=True)
        payload = {"status": "ready", "kind": kind, "updated_at": utc_now(), "last_result": result}
        self.heartbeat.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
