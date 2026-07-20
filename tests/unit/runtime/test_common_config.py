from __future__ import annotations

from pathlib import Path

import pytest

from agent_hook.config import AgentHookConfig
from config import DEFAULT_MEMORY_ROOT, MemoryOSConfig, RuntimeMode
from infrastructure.model.config import ModelConfig
from openApi.http.config import HTTPServerConfig
from openApi.mcp.config import MCPServerConfig
from runtime.config import RetentionConfig, RetrievalConfig, RuntimeConfig


def _clear_common_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MEMORYOS_ROOT",
        "MEMORYOS_MODE",
        "MEMORYOS_LOG_LEVEL",
        "MEMORYOS_MODEL_ENABLED",
        "MEMORYOS_MODEL_PROVIDER",
        "MEMORYOS_MODEL_PROTOCOL",
        "MEMORYOS_MODEL_NAME",
        "MEMORYOS_MODEL_BASE_URL",
        "MEMORYOS_MODEL_API_KEY_ENV",
        "MEMORYOS_MODEL_TIMEOUT_SECONDS",
        "MEMORYOS_MODEL_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_common_config_owns_shared_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_common_env(monkeypatch)

    config = MemoryOSConfig.from_env()

    assert DEFAULT_MEMORY_ROOT == "~/.memoryos"
    assert config.root == DEFAULT_MEMORY_ROOT
    assert config.mode == RuntimeMode.LOCAL
    assert config.log_level == "WARNING"
    assert not hasattr(config, "model")


def test_common_config_normalizes_shared_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMORYOS_ROOT", str(tmp_path))
    monkeypatch.setenv("MEMORYOS_MODE", "server")
    monkeypatch.setenv("MEMORYOS_LOG_LEVEL", "info")

    config = MemoryOSConfig.from_env()

    assert config.root_path == tmp_path.resolve()
    assert config.mode == RuntimeMode.SERVER
    assert config.log_level == "INFO"


def test_common_config_loads_model_metadata_without_reading_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_common_env(monkeypatch)
    monkeypatch.setenv("MEMORYOS_ROOT", str(tmp_path))
    monkeypatch.setenv("MEMORYOS_MODEL_ENABLED", "true")
    monkeypatch.setenv("MEMORYOS_MODEL_PROVIDER", "local-vllm")
    monkeypatch.setenv("MEMORYOS_MODEL_PROTOCOL", "openai_compatible")
    monkeypatch.setenv("MEMORYOS_MODEL_NAME", "qwen3")
    monkeypatch.setenv("MEMORYOS_MODEL_BASE_URL", "http://127.0.0.1:8000/v1/")
    monkeypatch.setenv("MEMORYOS_MODEL_API_KEY_ENV", "LOCAL_MODEL_TOKEN")
    monkeypatch.setenv("LOCAL_MODEL_TOKEN", "do-not-copy-into-config")

    config = ModelConfig.from_env()

    assert config == ModelConfig(
        enabled=True,
        provider="local-vllm",
        protocol="openai_compatible",
        model="qwen3",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="LOCAL_MODEL_TOKEN",
    )
    assert "do-not-copy-into-config" not in repr(config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"root": ""}, "root must be one explicit path"),
        ({"root": "${MEMORY_ROOT}"}, "root must be one explicit path"),
        ({"root": "/tmp/memory", "mode": "unknown"}, "unsupported runtime mode"),
        ({"root": "/tmp/memory", "log_level": "verbose"}, "unsupported log level"),
    ],
)
def test_common_config_rejects_invalid_shared_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryOSConfig(**kwargs)  # type: ignore[arg-type]


def test_model_config_rejects_credentials_embedded_in_url() -> None:
    with pytest.raises(ValueError, match="credential-free HTTP"):
        ModelConfig(
            enabled=True,
            provider="openai",
            model="gpt-5-mini",
            base_url="https://user:secret@example.com/v1",
        )


def test_module_configs_reuse_common_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMORYOS_ROOT", str(tmp_path))
    monkeypatch.setenv("MEMORYOS_MODE", "server")
    monkeypatch.setenv("MEMORYOS_LOG_LEVEL", "debug")
    monkeypatch.setenv("MEMORYOS_USER_ID", "u1")
    monkeypatch.setenv("MEMORYOS_HTTP_PORT", "9000")
    monkeypatch.setenv("MEMORYOS_MODEL_ENABLED", "true")
    monkeypatch.setenv("MEMORYOS_MODEL_PROVIDER", "local-vllm")
    monkeypatch.setenv("MEMORYOS_MODEL_NAME", "Qwen/Qwen3-8B")
    monkeypatch.setenv("MEMORYOS_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")

    runtime = RuntimeConfig.from_env(default_mode="server")
    mcp = MCPServerConfig.from_env()
    hook = AgentHookConfig.from_env()
    http = HTTPServerConfig.from_env()

    for config in (runtime, mcp, hook, http):
        assert config.root_path == tmp_path.resolve()
        assert config.mode == RuntimeMode.SERVER
        assert config.log_level == "DEBUG"
        assert config.model.enabled is True
        assert config.model.model == "Qwen/Qwen3-8B"
    assert http.port == 9000


def test_http_config_uses_server_mode_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_common_env(monkeypatch)
    monkeypatch.delenv("MEMORYOS_HTTP_PORT", raising=False)

    config = HTTPServerConfig.from_env()

    assert config.mode == RuntimeMode.SERVER


def test_runtime_config_requires_typed_module_configuration(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="retrieval must be a RetrievalConfig"):
        RuntimeConfig(root=str(tmp_path), retrieval={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retention must be a RetentionConfig"):
        RuntimeConfig(root=str(tmp_path), retention={})  # type: ignore[arg-type]

    config = RuntimeConfig(
        root=str(tmp_path),
        retrieval=RetrievalConfig(vectorize_important_session_events=True),
        retention=RetentionConfig(hot_days=1, warm_days=2, cold_days=3, batch_size=32),
    )

    assert config.retrieval.vectorize_important_session_events is True
    assert config.retention.to_mapping()["batch_size"] == 32
