"""与领域无关的进程就绪状态，以及附着在服务上的就绪保护。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from foundation.clock import utc_now


class RuntimeReadinessState(str, Enum):
    STARTING = "STARTING"
    RECOVERING = "RECOVERING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    NOT_READY = "NOT_READY"
    STOPPING = "STOPPING"


class RuntimeNotReadyError(RuntimeError):
    def __init__(self, state: RuntimeReadinessState, reasons: tuple[str, ...]) -> None:
        self.state = state
        self.reasons = reasons
        super().__init__(
            f"MemoryOS runtime is {state.value}: " + ("; ".join(reasons) if reasons else "startup incomplete")
        )


@dataclass
class RuntimeReadiness:
    state: RuntimeReadinessState = RuntimeReadinessState.STARTING
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def transition(
        self,
        state: RuntimeReadinessState,
        *,
        reasons: tuple[str, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.state = state
            self.reasons = tuple(str(item) for item in reasons if str(item))
            self.details = dict(details or {})
            self.updated_at = utc_now()

    def require_ready(self) -> None:
        with self._lock:
            if self.state != RuntimeReadinessState.READY:
                raise RuntimeNotReadyError(self.state, self.reasons)

    def mark_not_ready(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        self.transition(RuntimeReadinessState.NOT_READY, reasons=(str(reason),), details=details)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "ready": self.state == RuntimeReadinessState.READY,
                "reasons": list(self.reasons),
                "details": dict(self.details),
                "updated_at": self.updated_at,
            }


def readiness_for_source_store(source_store: Any) -> Any | None:
    readiness = getattr(source_store, "readiness", None)
    return readiness if callable(getattr(readiness, "require_ready", None)) else None


def readiness_for_session_service(service: Any) -> Any | None:
    committer = getattr(service, "committer", None)
    committer = getattr(committer, "delegate", committer)
    readiness = readiness_for_source_store(getattr(committer, "source_store", None))
    if readiness is not None:
        return readiness
    return None


def require_source_store_ready(source_store: Any) -> None:
    readiness = readiness_for_source_store(source_store)
    if readiness is not None:
        readiness.require_ready()


def require_session_service_ready(service: Any) -> None:
    readiness = readiness_for_session_service(service)
    if readiness is not None:
        readiness.require_ready()


def session_service_is_ready(service: Any) -> bool:
    readiness = readiness_for_session_service(service)
    if readiness is None:
        return True
    state_obj = getattr(readiness, "state", None)
    return str(getattr(state_obj, "value", state_obj or "")) == "READY"


def require_source_store_recovering(source_store: Any) -> None:
    readiness = readiness_for_source_store(source_store)
    state_obj = getattr(readiness, "state", None)
    state = str(getattr(state_obj, "value", state_obj or "UNBOUND"))
    if readiness is None or state != "RECOVERING":
        raise RuntimeError(f"projection startup entry requires RECOVERING runtime, got {state}")


__all__ = [
    "RuntimeNotReadyError",
    "RuntimeReadiness",
    "RuntimeReadinessState",
    "readiness_for_session_service",
    "readiness_for_source_store",
    "require_session_service_ready",
    "require_source_store_ready",
    "require_source_store_recovering",
    "session_service_is_ready",
]
