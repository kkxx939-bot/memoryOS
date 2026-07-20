"""用于提取重放的无正文耐久显著性预约。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from foundation.clock import utc_now
from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest
from infrastructure.store.filesystem.durable_io import atomic_create_json
from memory.core.formation.salience import EpisodeSalienceGate, SalienceDecision, SalienceFactor

_SCHEMA = "memory_salience_reservation_v2"


class SalienceLedgerIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class SalienceReservationResult:
    decision: SalienceDecision
    created: bool
    reservation_digest: str


class DurableSalienceLedger:
    """只创建一次，并绑定租户、所有者、任务和证据摘要的预约。"""

    def __init__(self, root: str | Path, *, tenant_id: str) -> None:
        require_safe_path_segment(tenant_id, "salience tenant_id")
        shared = Path(root).resolve()
        artifact_root = shared if tenant_id == "default" else shared / "tenants" / tenant_id
        self.root = artifact_root / "system" / "memory-evidence" / "salience-reservations"
        self.artifact_root = artifact_root
        self.tenant_id = tenant_id

    def path(self, task_id: str) -> Path:
        if not task_id:
            raise ValueError("salience task_id is required")
        return self.root / f"{hashlib.sha256(task_id.encode()).hexdigest()}.json"

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
        if str(getattr(episode, "tenant_id", "")) != self.tenant_id or not user_id or not project_id:
            raise SalienceLedgerIntegrityError("salience reservation boundary mismatch")
        path = self.path(task_id)
        if path.is_symlink():
            raise SalienceLedgerIntegrityError("salience reservation cannot be a symlink")
        if path.exists():
            payload = self.load(task_id)
            if payload["user_id"] != user_id or payload["project_id"] != project_id:
                raise SalienceLedgerIntegrityError("salience task crosses owner boundary")
            decision = self._decision(dict(payload["decision"]))
            current, _, _ = gate.fingerprint(episode)
            if decision.episode_fingerprint != current:
                raise SalienceLedgerIntegrityError("salience task references different evidence")
            return SalienceReservationResult(decision, False, str(payload["reservation_digest"]))
        decision = gate.evaluate(
            episode,
            existing_memories=existing_memories,
            seen_episode_fingerprints=policy_seen_fingerprints,
            prior_episode_counts=prior_episode_counts,
            consumed_budget=policy_consumed_budget,
            max_episode_budget=max_episode_budget,
        )
        core = {
            "schema_version": _SCHEMA,
            "task_id": task_id,
            "tenant_id": self.tenant_id,
            "user_id": user_id,
            "project_id": project_id,
            "decision": self._payload(decision),
            "created_at": utc_now(),
        }
        payload = {**core, "reservation_digest": canonical_digest(core)}
        atomic_create_json(path, payload, artifact_root=self.artifact_root)
        stored = self.load(task_id)
        return SalienceReservationResult(self._decision(dict(stored["decision"])), True, str(stored["reservation_digest"]))

    def load(self, task_id: str) -> dict[str, Any]:
        path = self.path(task_id)
        try:
            if path.is_symlink():
                raise OSError("symlink")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SalienceLedgerIntegrityError("salience reservation is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA:
            raise SalienceLedgerIntegrityError("salience reservation schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "reservation_digest"}
        if payload.get("reservation_digest") != canonical_digest(core):
            raise SalienceLedgerIntegrityError("salience reservation digest is corrupt")
        if payload.get("tenant_id") != self.tenant_id or payload.get("task_id") != task_id:
            raise SalienceLedgerIntegrityError("salience reservation identity mismatch")
        self._decision(dict(payload.get("decision", {}) or {}))
        return payload

    @staticmethod
    def _payload(decision: SalienceDecision) -> dict[str, Any]:
        payload = asdict(decision)
        payload["factors"] = [asdict(item) for item in decision.factors]
        return payload

    @staticmethod
    def _decision(payload: dict[str, Any]) -> SalienceDecision:
        try:
            factors = tuple(
                SalienceFactor(str(item["name"]), int(item["weight"]), tuple(str(value) for value in item["event_ids"]))
                for item in payload.get("factors", [])
            )
            return SalienceDecision(
                salient=bool(payload["salient"]),
                reasons=tuple(str(item) for item in payload.get("reasons", [])),
                score=int(payload["score"]),
                factors=factors,
                episode_fingerprint=str(payload["episode_fingerprint"]),
                budget_cost=int(payload["budget_cost"]),
                duplicate=bool(payload.get("duplicate", False)),
                privacy_risk=bool(payload.get("privacy_risk", False)),
                metadata=dict(payload.get("metadata", {}) or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SalienceLedgerIntegrityError("salience decision is invalid") from exc


__all__ = ["DurableSalienceLedger", "SalienceLedgerIntegrityError", "SalienceReservationResult"]
