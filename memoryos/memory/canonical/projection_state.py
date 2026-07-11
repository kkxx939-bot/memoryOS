"""Durable, revision-bound state for canonical-memory projections."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

from memoryos.core.time import utc_now

try:  # pragma: no cover - all supported production platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


def _as_int(value: object) -> int:
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise ValueError("projection revision must be an integer")


class ProjectionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"


class ProjectionStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ProjectionRecord:
    """A durable projection attempt that never lives inside a canonical object."""

    claim_uri: str
    slot_uri: str
    source_revision: int
    projection_revision: int
    l0_uri: str
    l1_uri: str
    l2_uri: str
    manifest_uri: str
    current_claim_revision: int = 0
    index_status: str = ProjectionStepStatus.PENDING.value
    vector_status: str = ProjectionStepStatus.PENDING.value
    relation_status: str = ProjectionStepStatus.PENDING.value
    scope_status: str = ProjectionStepStatus.PENDING.value
    taxonomy_status: str = ProjectionStepStatus.PENDING.value
    status: str = ProjectionStatus.PENDING.value
    attempt_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    failure_reason: str = ""
    retryable: bool = True
    current: bool = False
    schema_version: str = "canonical_projection_v2"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProjectionRecord:
        return cls(
            claim_uri=str(payload["claim_uri"]),
            slot_uri=str(payload["slot_uri"]),
            source_revision=_as_int(payload["source_revision"]),
            projection_revision=_as_int(payload.get("projection_revision", payload["source_revision"])),
            l0_uri=str(payload.get("l0_uri", "")),
            l1_uri=str(payload.get("l1_uri", "")),
            l2_uri=str(payload.get("l2_uri", "")),
            manifest_uri=str(payload.get("manifest_uri", "")),
            current_claim_revision=_as_int(payload.get("current_claim_revision", payload["source_revision"])),
            index_status=str(payload.get("index_status", ProjectionStepStatus.PENDING.value)),
            vector_status=str(payload.get("vector_status", ProjectionStepStatus.PENDING.value)),
            relation_status=str(payload.get("relation_status", ProjectionStepStatus.PENDING.value)),
            scope_status=str(payload.get("scope_status", ProjectionStepStatus.PENDING.value)),
            taxonomy_status=str(payload.get("taxonomy_status", ProjectionStepStatus.PENDING.value)),
            status=str(payload.get("status", ProjectionStatus.PENDING.value)),
            attempt_count=_as_int(payload.get("attempt_count", 0)),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            failure_reason=str(payload.get("failure_reason", "")),
            retryable=bool(payload.get("retryable", True)),
            current=bool(payload.get("current", False)),
            schema_version=str(payload.get("schema_version", "canonical_projection_v2")),
        )

    @property
    def completed(self) -> bool:
        return self.status == ProjectionStatus.COMPLETED.value

    @property
    def usable(self) -> bool:
        required = (
            self.index_status,
            self.vector_status,
            self.relation_status,
            self.scope_status,
            self.taxonomy_status,
        )
        terminal = {ProjectionStepStatus.COMPLETED.value, ProjectionStepStatus.SKIPPED.value}
        return self.completed and all(status in terminal for status in required)


class ProjectionRecordStore:
    """Atomic sidecar store with a monotonic current-revision pointer per Claim."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_root = self.root / "system" / "projection-state"
        self._fallback_locks: dict[str, threading.RLock] = {}
        self._fallback_guard = threading.Lock()

    def start(
        self,
        *,
        claim_uri: str,
        slot_uri: str,
        source_revision: int,
        projection_revision: int,
        l0_uri: str,
        l1_uri: str,
        l2_uri: str,
        manifest_uri: str,
        current_claim_revision: int | None = None,
    ) -> ProjectionRecord:
        now = utc_now()
        existing = self.load(claim_uri, source_revision)
        if existing is None:
            record = ProjectionRecord(
                claim_uri=claim_uri,
                slot_uri=slot_uri,
                source_revision=source_revision,
                projection_revision=projection_revision,
                l0_uri=l0_uri,
                l1_uri=l1_uri,
                l2_uri=l2_uri,
                manifest_uri=manifest_uri,
                current_claim_revision=int(current_claim_revision or source_revision),
                status=ProjectionStatus.RUNNING.value,
                attempt_count=1,
                created_at=now,
                updated_at=now,
            )
        else:
            record = replace(
                existing,
                slot_uri=slot_uri,
                projection_revision=projection_revision,
                l0_uri=l0_uri,
                l1_uri=l1_uri,
                l2_uri=l2_uri,
                manifest_uri=manifest_uri,
                current_claim_revision=int(current_claim_revision or source_revision),
                status=ProjectionStatus.RUNNING.value,
                attempt_count=existing.attempt_count + 1,
                updated_at=now,
                failure_reason="",
                retryable=True,
                current=False,
            )
        self.save(record)
        return record

    def save(self, record: ProjectionRecord) -> ProjectionRecord:
        self._write_json_atomic(self.record_path(record.claim_uri, record.source_revision), record.to_dict())
        return record

    def update(
        self,
        record: ProjectionRecord,
        *,
        index_status: str | None = None,
        vector_status: str | None = None,
        relation_status: str | None = None,
        scope_status: str | None = None,
        taxonomy_status: str | None = None,
        status: str | None = None,
        failure_reason: str | None = None,
        retryable: bool | None = None,
        current: bool | None = None,
    ) -> ProjectionRecord:
        updated = replace(
            record,
            index_status=record.index_status if index_status is None else index_status,
            vector_status=record.vector_status if vector_status is None else vector_status,
            relation_status=record.relation_status if relation_status is None else relation_status,
            scope_status=record.scope_status if scope_status is None else scope_status,
            taxonomy_status=record.taxonomy_status if taxonomy_status is None else taxonomy_status,
            status=record.status if status is None else status,
            failure_reason=record.failure_reason if failure_reason is None else failure_reason,
            retryable=record.retryable if retryable is None else retryable,
            current=record.current if current is None else current,
            updated_at=utc_now(),
        )
        return self.save(updated)

    def fail(self, record: ProjectionRecord, reason: str, *, retryable: bool = True) -> ProjectionRecord:
        return self.update(
            record,
            status=ProjectionStatus.FAILED.value,
            failure_reason=str(reason)[:1000],
            retryable=retryable,
            current=False,
        )

    def stale(self, record: ProjectionRecord, reason: str) -> ProjectionRecord:
        return self.update(
            record,
            status=ProjectionStatus.STALE.value,
            failure_reason=str(reason)[:1000],
            retryable=False,
            current=False,
        )

    def promote(self, record: ProjectionRecord) -> ProjectionRecord:
        """Promote only monotonically; a late old revision can never replace a newer one."""

        pointer = self._read_json_optional(self.current_path(record.claim_uri)) or {}
        current_revision = _as_int(pointer.get("source_revision", 0) or 0)
        if current_revision > record.source_revision:
            return self.stale(record, "newer projection revision is already current")
        if current_revision and current_revision != record.source_revision:
            previous = self.load(record.claim_uri, current_revision)
            if previous is not None and previous.current:
                self.save(replace(previous, current=False, updated_at=utc_now()))
        completed = self.save(
            replace(
                record,
                status=ProjectionStatus.COMPLETED.value,
                failure_reason="",
                retryable=False,
                current=True,
                updated_at=utc_now(),
            )
        )
        self._write_json_atomic(
            self.current_path(record.claim_uri),
            {
                "claim_uri": record.claim_uri,
                "slot_uri": record.slot_uri,
                "source_revision": record.source_revision,
                "projection_revision": record.projection_revision,
                "record_path": str(self.record_path(record.claim_uri, record.source_revision)),
                "updated_at": completed.updated_at,
            },
        )
        return completed

    def clear_current_if(self, claim_uri: str, source_revision: int, *, reason: str) -> None:
        pointer_path = self.current_path(claim_uri)
        pointer = self._read_json_optional(pointer_path) or {}
        if _as_int(pointer.get("source_revision", 0) or 0) != int(source_revision):
            return
        pointer_path.unlink(missing_ok=True)
        record = self.load(claim_uri, source_revision)
        if record is not None:
            self.stale(record, reason)

    def load(self, claim_uri: str, source_revision: int) -> ProjectionRecord | None:
        payload = self._read_json_optional(self.record_path(claim_uri, source_revision))
        return ProjectionRecord.from_dict(payload) if payload is not None else None

    def load_current(self, claim_uri: str, *, source_revision: int | None = None) -> ProjectionRecord | None:
        pointer = self._read_json_optional(self.current_path(claim_uri))
        if pointer is None:
            return None
        revision = _as_int(pointer.get("source_revision", 0) or 0)
        if source_revision is not None and revision != int(source_revision):
            return None
        record = self.load(claim_uri, revision)
        if record is None or not record.current or not record.usable:
            return None
        return record

    def record_path(self, claim_uri: str, source_revision: int) -> Path:
        return self._claim_dir(claim_uri) / "revisions" / f"rev-{int(source_revision)}.json"

    def current_path(self, claim_uri: str) -> Path:
        return self._claim_dir(claim_uri) / "current.json"

    @contextmanager
    def claim_lock(self, claim_uri: str) -> Iterator[None]:
        """Serialize publication across workers without serializing unrelated Claims."""

        lock_path = self._claim_dir(claim_uri) / ".projection.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is not None:
            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        with self._fallback_guard:  # pragma: no cover
            lock = self._fallback_locks.setdefault(str(lock_path), threading.RLock())
        with lock:  # pragma: no cover
            yield

    def _claim_dir(self, claim_uri: str) -> Path:
        digest = hashlib.sha256(claim_uri.encode("utf-8")).hexdigest()
        return self.state_root / digest[:2] / digest

    def _read_json_optional(self, path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid projection state: {path.name}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid projection state: {path.name}")
        return value

    def _write_json_atomic(self, path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)


__all__ = [
    "ProjectionRecord",
    "ProjectionRecordStore",
    "ProjectionStatus",
    "ProjectionStepStatus",
]
