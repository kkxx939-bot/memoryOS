"""MemoryOS 进程入口。

入口归运行时模块所有，但通过延迟导入把命令解析保留在对外交付层，
避免仅导入 ``runtime`` 时提前加载 CLI、SDK、存储或后台任务。
"""

from __future__ import annotations


def run(argv: list[str] | None = None) -> int:
    """启动 MemoryOS 命令行入口，并返回进程退出码。"""

    from openApi.cli.commands import run as run_cli

    return run_cli(argv)


__all__ = ["run"]
