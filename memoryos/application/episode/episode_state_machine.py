from __future__ import annotations

from dataclasses import dataclass


EPISODE_STATE_VERSION = "episode_state_v1"

CREATED = "CREATED"
OBSERVED = "OBSERVED"
RETRIEVED = "RETRIEVED"
PREDICTED = "PREDICTED"
INTERVENTION_SELECTED = "INTERVENTION_SELECTED"
FEEDBACK_PENDING = "FEEDBACK_PENDING"
FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"
LEARNING_QUEUED = "LEARNING_QUEUED"
LEARNING_APPLIED = "LEARNING_APPLIED"
CLOSED = "CLOSED"
FAILED = "FAILED"

VALID_TRANSITIONS = {
    CREATED: {OBSERVED, FAILED},
    OBSERVED: {RETRIEVED, FAILED},
    RETRIEVED: {PREDICTED, FAILED},
    PREDICTED: {INTERVENTION_SELECTED, FAILED},
    INTERVENTION_SELECTED: {FEEDBACK_PENDING, FEEDBACK_RECEIVED, FAILED},
    FEEDBACK_PENDING: {FEEDBACK_RECEIVED, CLOSED, FAILED},
    FEEDBACK_RECEIVED: {LEARNING_QUEUED, FAILED},
    LEARNING_QUEUED: {LEARNING_APPLIED, FAILED},
    LEARNING_APPLIED: {CLOSED, FAILED},
    CLOSED: set(),
    FAILED: set(),
}


@dataclass(frozen=True)
class StateTransition:
    from_state: str
    to_state: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "from": self.from_state,
            "to": self.to_state,
            "reason": self.reason,
        }


def assert_transition(from_state: str, to_state: str) -> None:
    if to_state not in VALID_TRANSITIONS.get(from_state, set()):
        raise ValueError(f"Invalid episode transition: {from_state} -> {to_state}")
