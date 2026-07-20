"""不包含业务语义的跨领域基础工具。"""

from foundation.clock import utc_now
from foundation.ids import new_id, require_safe_path_segment, stable_hash
from foundation.readiness import (
    RuntimeNotReadyError,
    RuntimeReadiness,
    RuntimeReadinessState,
)

__all__ = [
    "RuntimeNotReadyError",
    "RuntimeReadiness",
    "RuntimeReadinessState",
    "new_id",
    "require_safe_path_segment",
    "stable_hash",
    "utc_now",
]
