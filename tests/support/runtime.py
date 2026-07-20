"""测试中创建并启动真实 Runtime 的唯一辅助入口。"""

from __future__ import annotations

from runtime import RuntimeBuilder, RuntimeConfig, RuntimeContainer, RuntimeDependencies


def build_test_runtime(
    config: RuntimeConfig,
    dependencies: RuntimeDependencies | None = None,
    **dependency_overrides: object,
) -> RuntimeContainer:
    """显式构建并启动 Runtime，同时保留测试所需的依赖替换能力。"""

    if dependencies is not None and dependency_overrides:
        raise ValueError("pass RuntimeDependencies or keyword overrides, not both")
    injected = dependencies or RuntimeDependencies(**dependency_overrides)  # type: ignore[arg-type]
    runtime = RuntimeBuilder(config, injected).build()
    runtime.start()
    return runtime


__all__ = ["build_test_runtime"]
