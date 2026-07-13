"""Runtime readiness gate for canonical-memory recovery and serving."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memoryos.core.time import utc_now


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
        """Fail the live runtime closed after an authoritative integrity violation."""

        self.transition(
            RuntimeReadinessState.NOT_READY,
            reasons=(str(reason),),
            details=details,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "ready": self.state == RuntimeReadinessState.READY,
                "reasons": list(self.reasons),
                "details": dict(self.details),
                "updated_at": self.updated_at,
            }
