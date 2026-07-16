"""后台任务里的启动器。"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.core.time import utc_now
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.operations.commit.quarantine import list_quarantine_records
from memoryos.runtime.readiness import RuntimeNotReadyError
from memoryos.workers.embedding_worker import EmbeddingWorker
from memoryos.workers.memory_proposal_worker import MemoryProposalWorker
from memoryos.workers.semantic_worker import SemanticWorker
from memoryos.workers.session_commit_worker import SessionCommitWorker


class WorkerRunner:
    _ORDINARY_KINDS = frozenset({"session-commit", "memory-proposal", "memory-projection", "semantic", "embedding"})

    def __init__(
        self,
        client: MemoryOSClient,
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
        self.artifact_root = (
            Path(client.root) if client.tenant_id == "default" else Path(client.root) / "tenants" / client.tenant_id
        )
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
            result["recovery"] = self.client.recovery_worker.process_all()
        if kind in self._ORDINARY_KINDS or kind == "all":
            # Recovery never promotes a live runtime to READY.  In particular,
            # ``all`` may repair artifacts, but callers must reconstruct the
            # runtime so the complete startup proof is evaluated before any
            # ordinary worker leases or writes durable work.
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"session-commit", "all"}:
            result["session_commit"] = SessionCommitWorker(self.client.session_commit_service).process_pending(
                batch_size=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"memory-projection", "all"}:
            result["memory_projection"] = self.client.memory_projection_worker.process_pending(limit=self.batch_size)
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"memory-proposal", "all"}:
            result["memory_proposal"] = MemoryProposalWorker(self.client.session_commit_service).process_pending(
                batch_size=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"semantic", "all"}:
            result["semantic"] = SemanticWorker(
                self.client.source_store,
                self.client.queue_store,
                migration_gate=self.client.migration_gate,
            ).process_pending(
                limit=self.batch_size,
                lease_seconds=self.lease_seconds,
                max_retries=self.max_retries,
            )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"embedding", "all"}:
            if self.client.vector_store is None or self.client.embedding_provider is None:
                result["embedding"] = {"status": "disabled", "processed": [], "failed": []}
            else:
                result["embedding"] = EmbeddingWorker(
                    self.client.source_store,
                    self.client.queue_store,
                    self.client.vector_store,
                    self.client.embedding_provider,
                    migration_gate=self.client.migration_gate,
                ).process_pending(
                    limit=self.batch_size,
                    lease_seconds=self.lease_seconds,
                    max_retries=self.max_retries,
                )
            if self._stop_if_not_ready(result, allow_result=kind == "all"):
                return result
        if kind in {"maintenance", "all"}:
            result["maintenance"] = self.client.context_db.verify_consistency()
        stats = getattr(self.client.queue_store, "stats", None)
        result["queue_stats"] = stats() if callable(stats) else {}
        result["quarantine_records"] = list_quarantine_records(self.artifact_root)
        return result

    def _stop_if_not_ready(
        self,
        result: dict[str, Any],
        *,
        allow_result: bool,
    ) -> bool:
        """Stop an ``all`` pass immediately after a live fail-closed flip."""

        try:
            self.client.readiness.require_ready()
        except RuntimeNotReadyError:
            if not allow_result:
                raise
            result["status"] = "not_ready"
            result["runtime"] = self.client.readiness.snapshot()
            stats = getattr(self.client.queue_store, "stats", None)
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
        runtime = self.client.readiness.snapshot()
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
