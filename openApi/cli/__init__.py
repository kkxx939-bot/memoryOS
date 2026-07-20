"""MemoryOS 命令行交付通道的公开入口。"""

from __future__ import annotations


def run(argv: list[str] | None = None) -> int:
    """延迟加载 CLI 命令协议，避免包初始化时启动应用运行时。"""
    from openApi.cli.commands import run as run_commands

    return run_commands(argv)


__all__ = ["run"]
