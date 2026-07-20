"""运行时启动、恢复和停止状态机。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foundation.readiness import RuntimeReadinessState
from runtime.recovery.report import RecoveryReport

if TYPE_CHECKING:
    from runtime.container import RuntimeContainer
    from runtime.recovery.coordinator import RuntimeRecoveryCoordinator


class RuntimeLifecycle:
    """把对象构建与有副作用的启动恢复明确分开。"""

    def __init__(self, recovery: RuntimeRecoveryCoordinator) -> None:
        self.recovery = recovery

    def start(self, runtime: RuntimeContainer) -> RecoveryReport:
        if runtime.readiness.state == RuntimeReadinessState.READY:
            return RecoveryReport(ready=True, details=dict(runtime.readiness.details))
        return self.recovery.recover(runtime)

    def stop(self, runtime: RuntimeContainer) -> None:
        runtime.readiness.transition(RuntimeReadinessState.STOPPING)


__all__ = ["RuntimeLifecycle"]
