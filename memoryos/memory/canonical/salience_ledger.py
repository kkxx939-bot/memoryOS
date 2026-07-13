"""Create-only salience reservations that make extraction budgets durable."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.time import utc_now
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.salience import (
    EpisodeSalienceGate,
    SalienceDecision,
    SalienceFactor,
)
from memoryos.operations.commit.effect_marker import atomic_create_json

SALIENCE_RESERVATION_SCHEMA_VERSION = "memory_salience_reservation_v1"
SALIENCE_RESERVATION_ANCHOR_SCHEMA_VERSION = "memory_salience_reservation_anchor_v1"


class SalienceLedgerIntegrityError(RuntimeError):
    """A durable budget reservation is missing, corrupt, or cross-boundary."""


@dataclass(frozen=True)
class SalienceReservationResult:
    decision: SalienceDecision
    created: bool
    reservation_digest: str


class DurableSalienceLedger:
    """Reserve at most one extraction budget unit for one immutable task."""

    def __init__(self, root: str | Path, *, tenant_id: str) -> None:
        require_safe_path_segment(tenant_id, "salience ledger tenant_id")
        shared = Path(root)
        self.artifact_root = shared if tenant_id == "default" else shared / "tenants" / tenant_id
        self.root = self.artifact_root / "system" / "salience-reservations"
        self.anchor_root = self.artifact_root / "system" / "salience-reservation-anchors"
        self.tenant_id = tenant_id

    def path(self, task_id: str) -> Path:
        if not task_id:
            raise ValueError("salience reservation task_id is required")
        return self.root / f"{hashlib.sha256(task_id.encode('utf-8')).hexdigest()}.json"

    def anchor_path(self, task_id: str) -> Path:
        if not task_id:
            raise ValueError("salience reservation task_id is required")
        return self.anchor_root / f"{hashlib.sha256(task_id.encode('utf-8')).hexdigest()}.json"

    def reserve(
        self,
        gate: EpisodeSalienceGate,
        episode: Any,
        *,
        task_id: str,
        user_id: str,
        project_id: str,
        existing_memories: Sequence[Any] = (),
        policy_seen_fingerprints: Collection[str] = (),
        prior_episode_counts: Mapping[str, int] | None = None,
        policy_consumed_budget: int = 0,
        max_episode_budget: int = 8,
    ) -> SalienceReservationResult:
        if not user_id or not project_id:
            raise ValueError("durable salience reservation requires user and project identity")
        if str(getattr(episode, "tenant_id", "") or "") != self.tenant_id:
            raise SalienceLedgerIntegrityError("salience episode crosses the ledger tenant boundary")
        with self._scope_lock(user_id, project_id):
            reservation_path = self.path(task_id)
            if reservation_path.is_symlink():
                raise SalienceLedgerIntegrityError("salience reservation cannot be a symbolic link")
            if reservation_path.exists():
                existing = self.load(task_id)
                if existing["user_id"] != user_id or existing["project_id"] != project_id:
                    raise SalienceLedgerIntegrityError("salience task is already reserved across an owner boundary")
                current_fingerprint, current_budget_key, current_scope_keys = gate.fingerprint(episode)
                decision_payload = dict(existing["decision"])
                decision_metadata = dict(decision_payload.get("metadata", {}) or {})
                if (
                    decision_payload.get("episode_fingerprint") != current_fingerprint
                    or decision_metadata.get("budget_key") != current_budget_key
                    or tuple(decision_metadata.get("scope_keys", ()) or ()) != current_scope_keys
                ):
                    raise SalienceLedgerIntegrityError(
                        "salience task reservation is bound to a different semantic episode"
                    )
                return SalienceReservationResult(
                    self._decision(decision_payload),
                    False,
                    str(existing["reservation_digest"]),
                )

            reservations = self._all()
            scoped = [row for row in reservations if row["user_id"] == user_id and row["project_id"] == project_id]
            seen = {
                str(row["decision"].get("episode_fingerprint") or "")
                for row in scoped
                if row["decision"].get("episode_fingerprint")
            }
            seen.update(str(item) for item in policy_seen_fingerprints if item)
            consumed = sum(int(row["decision"].get("budget_cost", 0) or 0) for row in scoped)
            consumed = max(consumed, max(0, int(policy_consumed_budget)))
            decision = gate.evaluate(
                episode,
                existing_memories=existing_memories,
                seen_episode_fingerprints=seen,
                prior_episode_counts=prior_episode_counts,
                consumed_budget=consumed,
                max_episode_budget=max(0, min(8, int(max_episode_budget))),
            )
            decision_payload = self._decision_payload(decision)
            core = {
                "schema_version": SALIENCE_RESERVATION_SCHEMA_VERSION,
                "task_id": task_id,
                "tenant_id": self.tenant_id,
                "user_id": user_id,
                "project_id": project_id,
                "decision": decision_payload,
                "created_at": utc_now(),
            }
            payload = {**core, "reservation_digest": canonical_digest(core)}
            self._ensure_anchor(payload)
            atomic_create_json(
                self.path(task_id),
                payload,
                artifact_root=self.artifact_root,
            )
            stored = self.load(task_id)
            return SalienceReservationResult(
                self._decision(dict(stored["decision"])),
                True,
                str(stored["reservation_digest"]),
            )

    def load(self, task_id: str) -> dict[str, Any]:
        path = self.path(task_id)
        try:
            if path.is_symlink():
                raise OSError("salience reservation cannot be a symbolic link")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SalienceLedgerIntegrityError("salience reservation is unreadable") from exc
        return self._validate(payload, task_id=task_id)

    def validate_all(self) -> dict[str, int]:
        rows = self._all()
        reservation_task_ids = {str(row["task_id"]) for row in rows}
        anchors = 0
        anchor_task_ids: set[str] = set()
        for path in sorted(self.anchor_root.glob("*.json")) if self.anchor_root.exists() else ():
            anchor = self._load_anchor(path)
            task_id = str(anchor["task_id"])
            if path.is_symlink() or path.resolve() != self.anchor_path(task_id).resolve():
                raise SalienceLedgerIntegrityError("salience anchor path is invalid")
            try:
                reservation = self.load(task_id)
            except SalienceLedgerIntegrityError:
                raise
            except FileNotFoundError as exc:
                raise SalienceLedgerIntegrityError("salience reservation referenced by its anchor is missing") from exc
            if anchor["reservation_digest"] != reservation["reservation_digest"]:
                raise SalienceLedgerIntegrityError("salience anchor disagrees with its reservation")
            anchor_task_ids.add(task_id)
            anchors += 1
        if anchor_task_ids != reservation_task_ids:
            missing = sorted(reservation_task_ids - anchor_task_ids)
            detached = sorted(anchor_task_ids - reservation_task_ids)
            raise SalienceLedgerIntegrityError(
                "salience reservation and immutable anchor sets disagree: "
                f"missing_anchors={missing}; detached_anchors={detached}"
            )
        return {"reservations": len(rows), "anchors": anchors}

    def _all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")) if self.root.exists() else ():
            if path.is_symlink():
                raise SalienceLedgerIntegrityError("salience reservation path is invalid")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SalienceLedgerIntegrityError(f"salience reservation is unreadable: {path.name}") from exc
            task_id = str(raw.get("task_id") or "") if isinstance(raw, dict) else ""
            if not task_id or path.is_symlink() or path.resolve() != self.path(task_id).resolve():
                raise SalienceLedgerIntegrityError("salience reservation path is invalid")
            rows.append(self._validate(raw, task_id=task_id))
        return rows

    def _validate(self, payload: object, *, task_id: str) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != SALIENCE_RESERVATION_SCHEMA_VERSION:
            raise SalienceLedgerIntegrityError("salience reservation schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "reservation_digest"}
        if payload.get("reservation_digest") != canonical_digest(core):
            raise SalienceLedgerIntegrityError("salience reservation digest is corrupt")
        if payload.get("tenant_id") != self.tenant_id or payload.get("task_id") != task_id:
            raise SalienceLedgerIntegrityError("salience reservation crosses task or tenant boundary")
        if not payload.get("user_id") or not payload.get("project_id"):
            raise SalienceLedgerIntegrityError("salience reservation owner or project is missing")
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            raise SalienceLedgerIntegrityError("salience reservation decision is invalid")
        reconstructed = self._decision(dict(decision))
        if self._decision_payload(reconstructed) != decision:
            raise SalienceLedgerIntegrityError("salience reservation decision is not canonical")
        return payload

    def _ensure_anchor(self, reservation: dict[str, Any]) -> dict[str, Any]:
        task_id = str(reservation["task_id"])
        core = {
            "schema_version": SALIENCE_RESERVATION_ANCHOR_SCHEMA_VERSION,
            "task_id": task_id,
            "tenant_id": self.tenant_id,
            "reservation_digest": str(reservation["reservation_digest"]),
        }
        anchor = {**core, "anchor_digest": canonical_digest(core)}
        path = self.anchor_path(task_id)
        if path.is_symlink():
            raise SalienceLedgerIntegrityError("salience reservation anchor cannot be a symbolic link")
        if path.exists():
            existing = self._load_anchor(path)
            if existing != anchor:
                raise SalienceLedgerIntegrityError("salience reservation conflicts with its immutable anchor")
            return existing
        atomic_create_json(path, anchor, artifact_root=self.artifact_root)
        return self._load_anchor(path)

    def _load_anchor(self, path: Path) -> dict[str, Any]:
        try:
            if path.is_symlink():
                raise OSError("salience reservation anchor cannot be a symbolic link")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SalienceLedgerIntegrityError("salience reservation anchor is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SALIENCE_RESERVATION_ANCHOR_SCHEMA_VERSION:
            raise SalienceLedgerIntegrityError("salience reservation anchor schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "anchor_digest"}
        if payload.get("anchor_digest") != canonical_digest(core) or payload.get("tenant_id") != self.tenant_id:
            raise SalienceLedgerIntegrityError("salience reservation anchor is corrupt")
        return payload

    @staticmethod
    def _decision_payload(decision: SalienceDecision) -> dict[str, Any]:
        return {
            "salient": decision.salient,
            "reasons": list(decision.reasons),
            "score": decision.score,
            "factors": [
                {"name": item.name, "weight": item.weight, "event_ids": list(item.event_ids)}
                for item in decision.factors
            ],
            "episode_fingerprint": decision.episode_fingerprint,
            "budget_cost": decision.budget_cost,
            "duplicate": decision.duplicate,
            "privacy_risk": decision.privacy_risk,
            "metadata": dict(decision.metadata),
        }

    @staticmethod
    def _decision(payload: dict[str, Any]) -> SalienceDecision:
        required = {
            "salient",
            "reasons",
            "score",
            "factors",
            "episode_fingerprint",
            "budget_cost",
            "duplicate",
            "privacy_risk",
            "metadata",
        }
        if set(payload) != required or not isinstance(payload.get("factors"), list):
            raise SalienceLedgerIntegrityError("salience decision fields are invalid")
        try:
            return SalienceDecision(
                bool(payload["salient"]),
                tuple(str(item) for item in payload["reasons"]),
                score=int(payload["score"]),
                factors=tuple(
                    SalienceFactor(
                        str(item["name"]),
                        int(item["weight"]),
                        tuple(str(value) for value in item.get("event_ids", []) or []),
                    )
                    for item in payload["factors"]
                    if isinstance(item, dict)
                ),
                episode_fingerprint=str(payload["episode_fingerprint"]),
                budget_cost=int(payload["budget_cost"]),
                duplicate=bool(payload["duplicate"]),
                privacy_risk=bool(payload["privacy_risk"]),
                metadata=dict(payload["metadata"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SalienceLedgerIntegrityError("salience decision cannot be reconstructed") from exc

    @contextmanager
    def _scope_lock(self, user_id: str, project_id: str) -> Iterator[None]:
        digest = hashlib.sha256(f"{self.tenant_id}\0{user_id}\0{project_id}".encode()).hexdigest()
        path = self.root / ".locks" / f"{digest}.lock"
        descriptor = open_private_lock(path, root=self.artifact_root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
