"""本地 HTTP 服务监听配置。"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field

from config import MemoryOSConfig, RuntimeMode
from infrastructure.model.config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class HTTPServerConfig(MemoryOSConfig):
    """HTTP 服务只允许使用回环监听地址。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    host: str = "127.0.0.1"
    port: int = 8765

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        host = str(self.host).strip()
        if not host:
            raise ValueError("HTTP host must be non-empty")
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host.casefold() == "localhost"
        if not is_loopback:
            raise ValueError("single-user MemoryOS HTTP may only bind to a loopback host")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("HTTP port must be an integer between 1 and 65535")
        object.__setattr__(self, "host", host)

    @classmethod
    def from_env(
        cls,
        *,
        default_mode: RuntimeMode | str = RuntimeMode.SERVER,
    ) -> HTTPServerConfig:
        """从环境变量组合公共配置和 HTTP 专属配置。"""

        common = MemoryOSConfig.from_env(default_mode=default_mode)
        try:
            port = int(os.environ.get("MEMORYOS_HTTP_PORT", "8765"))
        except (TypeError, ValueError) as exc:
            raise ValueError("MEMORYOS_HTTP_PORT must be an integer") from exc
        return cls(
            root=common.root,
            mode=common.mode,
            log_level=common.log_level,
            model=ModelConfig.from_env(),
            host=os.environ.get("MEMORYOS_HTTP_HOST", "127.0.0.1"),
            port=port,
        )


__all__ = ["HTTPServerConfig"]
