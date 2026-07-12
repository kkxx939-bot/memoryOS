"""Durable, attempt-owned state for canonical-memory projections."""

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
from memoryos.operations.commit.quarantine import quarantine_control_file

try:  # pragma: no cover - all supported production POSIX platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("projection revision must be an integer")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError("projection revision must be an integer")


class ProjectionIntegrityError(RuntimeError):
    """Raised when projection control state or an equal-revision effect conflicts."""


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
    """One immutable-identity projection attempt with mutable owned progress."""

    claim_uri: str
    slot_uri: str
    source_revision: int
    projection_revision: int
    projection_attempt_id: str
    input_effect_hash: str
    publish_token: str
    l0_uri: str
    l1_uri: str
    l2_uri: str
    manifest_uri: str
    relations_uri: str = ""
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
    schema_version: str = "canonical_projection_v3"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProjectionRecord:
        if str(payload.get("schema_version", "")) != "canonical_projection_v3":
            raise ProjectionIntegrityError("unsupported projection record schema")
        attempt_id = str(payload.get("projection_attempt_id", ""))
        effect_hash = str(payload.get("input_effect_hash", ""))
        publish_token = str(payload.get("publish_token", ""))
        if not attempt_id or not effect_hash or not publish_token:
            raise ProjectionIntegrityError("projection record is missing attempt ownership")
        return cls(
            claim_uri=str(payload["claim_uri"]),
            slot_uri=str(payload["slot_uri"]),
            source_revision=_as_int(payload["source_revision"]),
            projection_revision=_as_int(payload.get("projection_revision", payload["source_revision"])),
            projection_attempt_id=attempt_id,
            input_effect_hash=effect_hash,
            publish_token=publish_token,
            l0_uri=str(payload.get("l0_uri", "")),
            l1_uri=str(payload.get("l1_uri", "")),
            l2_uri=str(payload.get("l2_uri", "")),
            manifest_uri=str(payload.get("manifest_uri", "")),
            relations_uri=str(payload.get("relations_uri", "")),
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
            schema_version="canonical_projection_v3",
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
    """Atomic attempt records plus a claim-scoped CAS current pointer."""

    POINTER_SCHEMA = "canonical_projection_current_v3"

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
        input_effect_hash: str,
        l0_uri: str,
        l1_uri: str,
        l2_uri: str,
        manifest_uri: str,
        relations_uri: str = "",
        current_claim_revision: int | None = None,
        projection_attempt_id: str | None = None,
    ) -> ProjectionRecord:
        if not input_effect_hash:
            raise ValueError("projection input effect hash is required")
        attempt_id = projection_attempt_id or uuid.uuid4().hex
        if not attempt_id or any(character not in "0123456789abcdef" for character in attempt_id.casefold()):
            raise ValueError("projection attempt id must be a hexadecimal safe segment")
        existing_attempts = self.attempts(claim_uri, source_revision)
        now = utc_now()
        record = ProjectionRecord(
            claim_uri=claim_uri,
            slot_uri=slot_uri,
            source_revision=source_revision,
            projection_revision=projection_revision,
            projection_attempt_id=attempt_id,
            input_effect_hash=input_effect_hash,
            publish_token=uuid.uuid4().hex,
            l0_uri=l0_uri,
            l1_uri=l1_uri,
            l2_uri=l2_uri,
            manifest_uri=manifest_uri,
            relations_uri=relations_uri,
            current_claim_revision=int(current_claim_revision or source_revision),
            status=ProjectionStatus.RUNNING.value,
            attempt_count=len(existing_attempts) + 1,
            created_at=now,
            updated_at=now,
        )
        return self.save(record)

    def save(self, record: ProjectionRecord) -> ProjectionRecord:
        path = self.attempt_path(record.claim_uri, record.source_revision, record.projection_attempt_id)
        if path.exists():
            existing = ProjectionRecord.from_dict(self._read_json(path))
            if self._identity(existing) != self._identity(record):
                raise ProjectionIntegrityError("projection attempt identity changed")
        self._write_json_atomic(path, record.to_dict())
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
        persisted = self.load(
            record.claim_uri,
            record.source_revision,
            projection_attempt_id=record.projection_attempt_id,
        )
        if persisted is None or self._identity(persisted) != self._identity(record):
            raise ProjectionIntegrityError("projection attempt no longer owns its record")
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
        current = self.load_current(record.claim_uri)
        if current is not None and current.projection_attempt_id == record.projection_attempt_id:
            return current
        return self.update(
            record,
            status=ProjectionStatus.FAILED.value,
            failure_reason=str(reason)[:1000],
            retryable=retryable,
            current=False,
        )

    def stale(self, record: ProjectionRecord, reason: str) -> ProjectionRecord:
        current = self.load_current(record.claim_uri)
        if current is not None and current.projection_attempt_id == record.projection_attempt_id:
            return current
        return self.update(
            record,
            status=ProjectionStatus.STALE.value,
            failure_reason=str(reason)[:1000],
            retryable=False,
            current=False,
        )

    def promote(self, record: ProjectionRecord, *, replace_same_effect: bool = False) -> ProjectionRecord:
        """CAS one completed attempt into current; caller must hold ``claim_lock``."""

        current = self.load_current(record.claim_uri)
        if current is not None:
            if current.source_revision > record.source_revision:
                return self.stale(record, "newer projection revision is already current")
            if current.source_revision == record.source_revision:
                if current.input_effect_hash != record.input_effect_hash:
                    raise ProjectionIntegrityError("same projection revision has a different input effect")
                if current.projection_attempt_id == record.projection_attempt_id:
                    return current
                if not replace_same_effect:
                    self.stale(record, "equivalent projection attempt is already current")
                    return current

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
        pointer_core: dict[str, object] = {
            "schema_version": self.POINTER_SCHEMA,
            "claim_uri": completed.claim_uri,
            "slot_uri": completed.slot_uri,
            "source_revision": completed.source_revision,
            "projection_revision": completed.projection_revision,
            "projection_attempt_id": completed.projection_attempt_id,
            "input_effect_hash": completed.input_effect_hash,
            "publish_token": completed.publish_token,
            "record_path": str(self.attempt_path_for(completed)),
            "updated_at": completed.updated_at,
        }
        self._write_json_atomic(
            self.current_path(completed.claim_uri),
            {**pointer_core, "pointer_digest": self._digest(pointer_core)},
        )
        if current is not None and current.projection_attempt_id != completed.projection_attempt_id:
            self.save(replace(current, current=False, updated_at=utc_now()))
        return completed

    def clear_current_if(
        self,
        claim_uri: str,
        source_revision: int,
        *,
        projection_attempt_id: str,
        publish_token: str,
        reason: str,
    ) -> bool:
        pointer = self._read_pointer_optional(claim_uri)
        if pointer is None:
            return False
        if (
            _as_int(pointer.get("source_revision", 0)) != int(source_revision)
            or str(pointer.get("projection_attempt_id", "")) != projection_attempt_id
            or str(pointer.get("publish_token", "")) != publish_token
        ):
            return False
        self.current_path(claim_uri).unlink(missing_ok=True)
        record = self.load(claim_uri, source_revision, projection_attempt_id=projection_attempt_id)
        if record is not None:
            self.save(
                replace(
                    record,
                    status=ProjectionStatus.STALE.value,
                    failure_reason=str(reason)[:1000],
                    retryable=False,
                    current=False,
                    updated_at=utc_now(),
                )
            )
        return True

    def load(
        self,
        claim_uri: str,
        source_revision: int,
        *,
        projection_attempt_id: str | None = None,
    ) -> ProjectionRecord | None:
        if projection_attempt_id is not None:
            path = self.attempt_path(claim_uri, source_revision, projection_attempt_id)
            if not path.exists():
                return None
            record = ProjectionRecord.from_dict(self._read_json(path))
            self._validate_location(record, claim_uri, source_revision, projection_attempt_id)
            return record
        current = self.load_current(claim_uri, source_revision=source_revision)
        if current is not None:
            return current
        attempts = self.attempts(claim_uri, source_revision)
        return max(attempts, key=lambda item: (item.updated_at, item.projection_attempt_id), default=None)

    def attempts(self, claim_uri: str, source_revision: int) -> list[ProjectionRecord]:
        directory = self._revision_dir(claim_uri, source_revision)
        if not directory.exists():
            return []
        result: list[ProjectionRecord] = []
        for path in sorted(directory.glob("attempt-*.json")):
            record = ProjectionRecord.from_dict(self._read_json(path))
            self._validate_location(record, claim_uri, source_revision, record.projection_attempt_id)
            result.append(record)
        return result

    def load_current(self, claim_uri: str, *, source_revision: int | None = None) -> ProjectionRecord | None:
        pointer = self._read_pointer_optional(claim_uri)
        if pointer is None:
            return None
        revision = _as_int(pointer.get("source_revision", 0))
        if source_revision is not None and revision != int(source_revision):
            return None
        attempt_id = str(pointer.get("projection_attempt_id", ""))
        record = self.load(claim_uri, revision, projection_attempt_id=attempt_id)
        if (
            record is None
            or not record.current
            or not record.usable
            or record.input_effect_hash != str(pointer.get("input_effect_hash", ""))
            or record.publish_token != str(pointer.get("publish_token", ""))
            or str(self.attempt_path_for(record)) != str(pointer.get("record_path", ""))
        ):
            raise ProjectionIntegrityError("projection current pointer does not match its attempt record")
        return record

    def record_path(
        self,
        claim_uri: str,
        source_revision: int,
        projection_attempt_id: str | None = None,
    ) -> Path:
        if projection_attempt_id is not None:
            return self.attempt_path(claim_uri, source_revision, projection_attempt_id)
        record = self.load(claim_uri, source_revision)
        if record is None:
            return self._revision_dir(claim_uri, source_revision) / "missing.json"
        return self.attempt_path_for(record)

    def attempt_path(self, claim_uri: str, source_revision: int, projection_attempt_id: str) -> Path:
        return self._revision_dir(claim_uri, source_revision) / f"attempt-{projection_attempt_id}.json"

    def attempt_path_for(self, record: ProjectionRecord) -> Path:
        return self.attempt_path(record.claim_uri, record.source_revision, record.projection_attempt_id)

    def current_path(self, claim_uri: str) -> Path:
        return self._claim_dir(claim_uri) / "current.json"

    @contextmanager
    def claim_lock(self, claim_uri: str) -> Iterator[None]:
        """Serialize publication across processes without serializing unrelated Claims."""

        lock_path = self._claim_dir(claim_uri) / ".projection.lock"
        self._secure_directory(lock_path.parent)
        if not lock_path.exists():
            lock_path.touch(mode=0o600)
        os.chmod(lock_path, 0o600)
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

    def _revision_dir(self, claim_uri: str, source_revision: int) -> Path:
        return self._claim_dir(claim_uri) / "revisions" / f"rev-{int(source_revision)}"

    def _claim_dir(self, claim_uri: str) -> Path:
        digest = hashlib.sha256(claim_uri.encode("utf-8")).hexdigest()
        return self.state_root / digest[:2] / digest

    def _read_pointer_optional(self, claim_uri: str) -> dict[str, object] | None:
        path = self.current_path(claim_uri)
        if not path.exists():
            return None
        pointer = self._read_json(path)
        if str(pointer.get("schema_version", "")) != self.POINTER_SCHEMA:
            raise ProjectionIntegrityError("unsupported projection current pointer schema")
        claimed = str(pointer.get("pointer_digest", ""))
        core = {key: value for key, value in pointer.items() if key != "pointer_digest"}
        if not claimed or claimed != self._digest(core):
            raise ProjectionIntegrityError("projection current pointer digest mismatch")
        return pointer

    def _read_json(self, path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if path.exists():
                quarantine_control_file(
                    self.root,
                    path,
                    kind="projection_record",
                    error=exc,
                    identifiers={"record_id": path.stem},
                )
            raise ProjectionIntegrityError(f"invalid projection state: {path.name}") from exc
        if not isinstance(value, dict):
            quarantine_control_file(
                self.root,
                path,
                kind="projection_record",
                error=ValueError("projection state is not an object"),
                identifiers={"record_id": path.stem},
            )
            raise ProjectionIntegrityError(f"invalid projection state: {path.name}")
        return value

    def _write_json_atomic(self, path: Path, payload: Mapping[str, object]) -> None:
        self._secure_directory(path.parent)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _secure_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = directory
        stop = self.root.parent
        while current != stop and (current == self.root or self.root in current.parents):
            os.chmod(current, 0o700)
            if current == self.root:
                break
            current = current.parent

    def _digest(self, payload: Mapping[str, object]) -> str:
        raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _identity(self, record: ProjectionRecord) -> tuple[object, ...]:
        return (
            record.claim_uri,
            record.slot_uri,
            record.source_revision,
            record.projection_revision,
            record.projection_attempt_id,
            record.input_effect_hash,
            record.publish_token,
        )

    def _validate_location(
        self,
        record: ProjectionRecord,
        claim_uri: str,
        source_revision: int,
        projection_attempt_id: str,
    ) -> None:
        if (
            record.claim_uri != claim_uri
            or record.source_revision != int(source_revision)
            or record.projection_attempt_id != projection_attempt_id
        ):
            raise ProjectionIntegrityError("projection record path identity mismatch")


__all__ = [
    "ProjectionIntegrityError",
    "ProjectionRecord",
    "ProjectionRecordStore",
    "ProjectionStatus",
    "ProjectionStepStatus",
]
