"""统一调度后台队列、恢复和维护任务。"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any

from foundation.clock import utc_now
from foundation.identity import LOCAL_STORAGE_NAMESPACE
from foundation.readiness import RuntimeNotReadyError
from infrastructure.context.maintenance.embedding_worker import EmbeddingWorker
from infrastructure.context.maintenance.semantic_worker import SemanticWorker
from infrastructure.store.filesystem.durable_io import atomic_write_json
from infrastructure.store.filesystem.durable_io.quarantine import list_quarantine_records
from infrastructure.store.trace import RecallTraceRepository, recall_trace_root
from runtime.worker.contracts import WorkerRuntime
from runtime.worker.session_commit import SessionCommitWorker


class WorkerRunner:
    _ORDINARY_KINDS = frozenset(
        {
            "session-commit",
            "semantic",
            "embedding",
        }
    )

    def __init__(
        self,
        client: WorkerRuntime,
        *,
        poll_interval: float = 1.0,
        batch_size: int = 10,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.poll_interval = max(0.05, poll_interval)
        self.batch_size = max(1, batch_size)
        self.lease_seconds = max(1, lease_seconds)
        self.max_retries = max(1, max_retries)
        self.stopping = False
        self.artifact_root = Path(client.root)
        self.heartbeat = self.artifact_root / "system" / "worker-health.json"

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
        if kind in {"recovery", "all"}:
            result["recovery"] = self.client.runtime.transaction.recovery_worker.process_all()
        if kind in self._ORDINARY_KINDS or kind == "all":
            # 恢复任务不能把当前运行时直接提升为 READY。即使 ``all`` 修复了
            # 耐久数据，也必须重建运行时并重新完成启动证明，普通 Worker 才能
            # 租赁队列或写入数据。
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"session-commit", "all"}:
            result["commit"] = SessionCommitWorker(self.client.runtime.session.commit_service).process_pending(
                batch_size=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"semantic", "all"}:
            result["semantic"] = SemanticWorker(
                self.client.runtime.stores.source,
                self.client.runtime.stores.queue,
            ).process_pending(
                limit=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"embedding", "all"}:
            if self.client.runtime.stores.vector is None or self.client.runtime.stores.embedding is None:
                result["embedding"] = {"status": "disabled", "processed": [], "failed": []}
            else:
                result["embedding"] = EmbeddingWorker(
                    self.client.runtime.stores.source,
                    self.client.runtime.stores.queue,
                    self.client.runtime.stores.vector,
                    self.client.runtime.stores.embedding,
                ).process_pending(
                    limit=self.batch_size,
                    lease_seconds=self.lease_seconds,
                    max_retries=self.max_retries,
                )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"maintenance", "all"}:
            maintenance = self.client.runtime.context.administration_service.verify_consistency()
            maintenance["recall_trace_retention"] = RecallTraceRepository(
                recall_trace_root(self.client.root, LOCAL_STORAGE_NAMESPACE)
            ).prune()
            result["maintenance"] = maintenance
        stats = getattr(self.client.runtime.stores.queue, "stats", None)
        result["queue_stats"] = stats() if callable(stats) else {}
        result["quarantine_records"] = list_quarantine_records(self.artifact_root)
        return result

    def _stop_if_not_ready(
        self,
        result: dict[str, Any],
        *,
        allow_result: bool,
    ) -> bool:
        """运行时进入 fail-closed 状态后，立即终止本轮 ``all`` 调度。"""

        try:
            self.client.runtime.readiness.require_ready()
        except RuntimeNotReadyError:
            if not allow_result:
                raise
            result["status"] = "not_ready"
            result["runtime"] = self.client.runtime.readiness.snapshot()
            stats = getattr(self.client.runtime.stores.queue, "stats", None)
            result["queue_stats"] = stats() if callable(stats) else {}
            result["quarantine_records"] = list_quarantine_records(self.artifact_root)
            return True
        return False

    def _install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: Any) -> None:
            self.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def _write_heartbeat(self, kind: str, result: dict[str, Any]) -> None:
        metrics = self._metrics(result)
        runtime = self.client.runtime.readiness.snapshot()
        status = (
            "not_ready"
            if result.get("status") == "not_ready" or not runtime.get("ready")
            else "failed"
            if metrics["component_errors"]
            else "degraded"
            if metrics["failed"] or metrics["dead_letter"] or metrics["quarantine"]
            else "ready"
        )
        payload = {
            "schema_version": "worker_health_v1",
            "status": status,
            "kind": kind,
            "updated_at": utc_now(),
            **metrics,
            "pending": dict(result.get("queue_stats", {})),
            "last_result": result,
        }
        atomic_write_json(self.heartbeat, payload, artifact_root=self.artifact_root)

    def _metrics(self, result: dict[str, Any]) -> dict[str, Any]:
        processed = succeeded = failed = retried = dead_letter = quarantine = 0
        last_error = ""
        component_errors = 0
        for name, value in result.items():
            if name in {"kind", "timestamp", "queue_stats", "maintenance"} or not isinstance(value, dict):
                continue
            try:
                processed += self._count(value, "claimed") + self._count(value, "recovered_count")
                succeeded += self._count(value, "committed") + self._count(value, "processed")
                failed += self._count(value, "failed") + self._count(value, "failed_count")
                dead_letter += self._count(value, "dead_letter")
                quarantine += self._count(value, "quarantine") + self._count(value, "quarantine_count")
                retried += max(0, self._count(value, "failed") - self._count(value, "dead_letter"))
            except (TypeError, ValueError):
                component_errors += 1
                last_error = f"InvalidWorkerResult:{name}"
            if value.get("last_error"):
                last_error = str(value["last_error"])
        queue_stats = dict(result.get("queue_stats", {}) or {})
        dead_letter = max(dead_letter, int(queue_stats.get("dead_letter", 0) or 0))
        quarantine = max(
            quarantine,
            int(queue_stats.get("quarantine", 0) or 0),
            len(result.get("quarantine_records", []) or []),
        )
        return {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "retried": retried,
            "dead_letter": dead_letter,
            "quarantine": quarantine,
            "last_error": last_error,
            "component_errors": component_errors,
        }

    @staticmethod
    def _count(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key, 0)
        if isinstance(value, list):
            return len(value)
        return int(value or 0)
