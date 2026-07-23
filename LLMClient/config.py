"""一级 LLMClient 包所管理的供应商路由配置。"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

_CONFIG_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelConfig:
    """一条模型主路由及其有序回退路由；密钥始终保存在环境变量中。"""

    enabled: bool = False
    provider: str = ""
    protocol: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 30.0
    max_output_tokens: int | None = None
    max_concurrent: int = 16
    max_response_bytes: int = 8 * 1024 * 1024
    native_structured_output: bool = False
    reasoning: bool = False
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, object] = field(default_factory=dict)
    fallbacks: tuple[ModelConfig, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("model enabled must be boolean")
        for field_name in ("provider", "protocol"):
            value = str(getattr(self, field_name) or "").strip().lower()
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

        timeout = _positive_float(self.timeout_seconds, "model timeout_seconds", maximum=600.0)
        object.__setattr__(self, "timeout_seconds", timeout)
        retry_base = _positive_float(
            self.retry_base_delay_seconds,
            "model retry_base_delay_seconds",
            maximum=60.0,
        )
        retry_max = _positive_float(
            self.retry_max_delay_seconds,
            "model retry_max_delay_seconds",
            maximum=300.0,
        )
        if retry_max < retry_base:
            raise ValueError("model retry_max_delay_seconds cannot be below retry_base_delay_seconds")
        object.__setattr__(self, "retry_base_delay_seconds", retry_base)
        object.__setattr__(self, "retry_max_delay_seconds", retry_max)

        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or not 0 <= self.max_retries <= 10
        ):
            raise ValueError("model max_retries must be between zero and 10")
        if self.max_output_tokens is not None and (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("model max_output_tokens must be a positive integer")
        if (
            not isinstance(self.max_concurrent, int)
            or isinstance(self.max_concurrent, bool)
            or not 1 <= self.max_concurrent <= 4096
        ):
            raise ValueError("model max_concurrent must be between one and 4096")
        if (
            not isinstance(self.max_response_bytes, int)
            or isinstance(self.max_response_bytes, bool)
            or not 1024 <= self.max_response_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("model max_response_bytes must be between 1 KiB and 64 MiB")
        if not isinstance(self.native_structured_output, bool):
            raise TypeError("model native_structured_output must be boolean")
        if not isinstance(self.reasoning, bool):
            raise TypeError("model reasoning must be boolean")

        headers = _string_mapping(self.extra_headers, "model extra_headers")
        forbidden_headers = {"authorization", "proxy-authorization"} & {key.casefold() for key in headers}
        if forbidden_headers:
            raise ValueError("model extra_headers cannot contain authorization credentials")
        object.__setattr__(self, "extra_headers", headers)
        if not isinstance(self.extra_body, Mapping):
            raise TypeError("model extra_body must be an object")
        forbidden_body = {"model", "messages", "stream", "tools", "tool_choice", "response_format"}
        overlap = forbidden_body & set(self.extra_body)
        if overlap:
            raise ValueError(f"model extra_body cannot override reserved fields: {sorted(overlap)}")
        extra_body = dict(self.extra_body)
        try:
            json.dumps(extra_body, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("model extra_body must contain JSON-serializable values") from exc
        object.__setattr__(self, "extra_body", extra_body)

        fallbacks = tuple(self.fallbacks)
        if any(not isinstance(item, ModelConfig) for item in fallbacks):
            raise TypeError("model fallbacks must contain ModelConfig values")
        if any(not item.enabled for item in fallbacks):
            raise ValueError("model fallback routes must be enabled")
        object.__setattr__(self, "fallbacks", fallbacks)

        if not self.enabled:
            if fallbacks:
                raise ValueError("disabled model config cannot declare fallback routes")
            return
        if not self.provider or not self.protocol or not self.model:
            raise ValueError("enabled model config requires provider, protocol and model")
        if self.protocol == "openai_compatible" and not base_url:
            raise ValueError("openai-compatible model config requires base_url")
        if base_url:
            self._validate_base_url(base_url)

    @staticmethod
    def _validate_base_url(base_url: str) -> None:
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
        if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
            raise ValueError("remote model base_url must use HTTPS")

    @classmethod
    def from_env(cls) -> ModelConfig:
        """加载连接元数据和环境变量名称，绝不加载密钥值。"""

        fallback_payload = _env_json("MEMORYOS_MODEL_FALLBACKS_JSON", [])
        if not isinstance(fallback_payload, list):
            raise ValueError("MEMORYOS_MODEL_FALLBACKS_JSON must contain an array")
        fallbacks: list[ModelConfig] = []
        for index, item in enumerate(fallback_payload):
            try:
                fallbacks.append(
                    _config_from_mapping(
                        item,
                        f"MEMORYOS_MODEL_FALLBACKS_JSON[{index}]",
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid model fallback at index {index}: {exc}") from exc

        max_output_tokens = _env_optional_int("MEMORYOS_MODEL_MAX_OUTPUT_TOKENS")
        return cls(
            enabled=_env_flag("MEMORYOS_MODEL_ENABLED", default=False),
            provider=os.environ.get("MEMORYOS_MODEL_PROVIDER", ""),
            protocol=os.environ.get("MEMORYOS_MODEL_PROTOCOL", "openai_compatible"),
            model=os.environ.get("MEMORYOS_MODEL_NAME", ""),
            base_url=os.environ.get("MEMORYOS_MODEL_BASE_URL", ""),
            api_key_env=os.environ.get("MEMORYOS_MODEL_API_KEY_ENV", ""),
            timeout_seconds=_env_float("MEMORYOS_MODEL_TIMEOUT_SECONDS", 30.0),
            max_retries=_env_int("MEMORYOS_MODEL_MAX_RETRIES", 2),
            retry_base_delay_seconds=_env_float("MEMORYOS_MODEL_RETRY_BASE_DELAY_SECONDS", 0.5),
            retry_max_delay_seconds=_env_float("MEMORYOS_MODEL_RETRY_MAX_DELAY_SECONDS", 30.0),
            max_output_tokens=max_output_tokens,
            max_concurrent=_env_int("MEMORYOS_MODEL_MAX_CONCURRENT", 16),
            max_response_bytes=_env_int("MEMORYOS_MODEL_MAX_RESPONSE_BYTES", 8 * 1024 * 1024),
            native_structured_output=_env_flag(
                "MEMORYOS_MODEL_NATIVE_STRUCTURED_OUTPUT",
                default=False,
            ),
            reasoning=_env_flag("MEMORYOS_MODEL_REASONING", default=False),
            extra_headers=_env_string_object("MEMORYOS_MODEL_EXTRA_HEADERS_JSON"),
            extra_body=_env_object("MEMORYOS_MODEL_EXTRA_BODY_JSON"),
            fallbacks=tuple(fallbacks),
        )


def _config_from_mapping(value: object, label: str) -> ModelConfig:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    values = dict(value)
    values.setdefault("enabled", True)
    children = values.get("fallbacks", ())
    if not isinstance(children, list | tuple):
        raise ValueError(f"{label}.fallbacks must be an array")
    values["fallbacks"] = tuple(
        _config_from_mapping(child, f"{label}.fallbacks[{index}]") for index, child in enumerate(children)
    )
    return ModelConfig(**values)


def _positive_float(value: object, label: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not 0 < number <= maximum:
        raise ValueError(f"{label} must be between zero and {maximum:g}")
    return number


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"{label} values must be strings")
        result[key] = item
    return result


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


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


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_json(name: str, default: object) -> object:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc


def _env_object(name: str) -> dict[str, object]:
    value = _env_json(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _env_string_object(name: str) -> dict[str, str]:
    return _string_mapping(_env_object(name), name)


__all__ = ["ModelConfig"]
