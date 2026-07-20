"""MemoryOS 所有启动方式共享的进程级配置。

这里只定义数据根目录、运行模式和日志级别。模型、HTTP、MCP、Agent Hook 与
Runtime 的专属参数由对应模块自己的 ``config.py`` 管理。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# 默认数据放在当前用户目录，避免从源码仓库启动时生成运行时文件。
DEFAULT_MEMORY_ROOT = "~/.memoryos"
DEFAULT_LOG_LEVEL = "WARNING"
SUPPORTED_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


class RuntimeMode(str, Enum):
    """MemoryOS 支持的进程运行方式。"""

    LOCAL = "local"
    SERVER = "server"
    REMOTE_CLIENT = "remote_client"


@dataclass(frozen=True)
class MemoryOSConfig:
    """所有进程入口复用的最小公共配置。"""

    root: str
    mode: RuntimeMode | str = RuntimeMode.LOCAL
    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        raw_root = str(self.root).strip()
        if not raw_root or any(marker in raw_root for marker in ("$", "${", "*", "?", "[", "]")):
            raise ValueError("root must be one explicit path without variables or glob syntax")
        object.__setattr__(self, "root", raw_root)

        try:
            mode = self.mode if isinstance(self.mode, RuntimeMode) else RuntimeMode(str(self.mode))
        except ValueError as exc:
            raise ValueError(f"unsupported runtime mode: {self.mode}") from exc
        object.__setattr__(self, "mode", mode)

        log_level = str(self.log_level).strip().upper()
        if log_level not in SUPPORTED_LOG_LEVELS:
            raise ValueError(f"unsupported log level: {self.log_level}")
        object.__setattr__(self, "log_level", log_level)

    @classmethod
    def from_env(
        cls,
        *,
        default_mode: RuntimeMode | str = RuntimeMode.LOCAL,
    ) -> MemoryOSConfig:
        """从环境变量读取一次公共进程配置。"""

        return cls(
            root=os.environ.get("MEMORYOS_ROOT", DEFAULT_MEMORY_ROOT),
            mode=os.environ.get("MEMORYOS_MODE", str(getattr(default_mode, "value", default_mode))),
            log_level=os.environ.get("MEMORYOS_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        )

    @property
    def root_path(self) -> Path:
        """返回经过展开和规范化的数据根目录。"""

        return Path(self.root).expanduser().resolve(strict=False)


__all__ = [
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_MEMORY_ROOT",
    "MemoryOSConfig",
    "RuntimeMode",
]
