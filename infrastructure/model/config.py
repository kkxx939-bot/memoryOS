"""模型 Provider 与统一 ModelClient 使用的连接配置。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

_CONFIG_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelConfig:
    """模型连接元数据；真实密钥始终保留在环境变量中。"""

    enabled: bool = False
    provider: str = ""
    protocol: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("model enabled must be boolean")
        for field_name in ("provider", "protocol"):
            value = str(getattr(self, field_name) or "").strip()
            object.__setattr__(self, field_name, value)
            if value and not _CONFIG_NAME.fullmatch(value):
                raise ValueError(f"model {field_name} contains unsupported characters")
        model = str(self.model or "").strip()
        object.__setattr__(self, "model", model)
        if model and not _MODEL_NAME.fullmatch(model):
            raise ValueError("model name contains unsupported characters")
        base_url = str(self.base_url or "").strip().rstrip("/")
        object.__setattr__(self, "base_url", base_url)
        api_key_env = str(self.api_key_env or "").strip()
        object.__setattr__(self, "api_key_env", api_key_env)
        if api_key_env and not _ENV_NAME.fullmatch(api_key_env):
            raise ValueError("model api_key_env must be an environment variable name")
        if isinstance(self.timeout_seconds, bool) or not 0 < float(self.timeout_seconds) <= 300:
            raise ValueError("model timeout_seconds must be between zero and 300")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) or not 0 <= self.max_retries <= 10:
            raise ValueError("model max_retries must be between zero and 10")
        if not self.enabled:
            return
        if not self.provider or not self.protocol or not self.model or not base_url:
            raise ValueError("enabled model config requires provider, protocol, model and base_url")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("model base_url must be one credential-free HTTP(S) origin or API prefix")

    @classmethod
    def from_env(cls) -> ModelConfig:
        """仅在显式启用时读取连接元数据，不读取密钥值。"""

        return cls(
            enabled=_env_flag("MEMORYOS_MODEL_ENABLED", default=False),
            provider=os.environ.get("MEMORYOS_MODEL_PROVIDER", ""),
            protocol=os.environ.get("MEMORYOS_MODEL_PROTOCOL", "openai_compatible"),
            model=os.environ.get("MEMORYOS_MODEL_NAME", ""),
            base_url=os.environ.get("MEMORYOS_MODEL_BASE_URL", ""),
            api_key_env=os.environ.get("MEMORYOS_MODEL_API_KEY_ENV", ""),
            timeout_seconds=_env_float("MEMORYOS_MODEL_TIMEOUT_SECONDS", 30.0),
            max_retries=_env_int("MEMORYOS_MODEL_MAX_RETRIES", 2),
        )


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = ["ModelConfig"]
