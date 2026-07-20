from __future__ import annotations

import json
from typing import Any

import pytest

from infrastructure.model import (
    ChatMessage,
    ChatRequest,
    ModelClient,
    ModelClientFactory,
    ModelConfigurationError,
    ModelResponse,
    ModelTransportError,
)
from infrastructure.model.config import ModelConfig
from infrastructure.model.providers import OpenAICompatibleProvider
from memory.formation.llm import LLMMemoryExtractorBackend
from runtime.config import RuntimeConfig
from tests.support.runtime import build_test_runtime


def _config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "enabled": True,
        "provider": "test-provider",
        "protocol": "openai_compatible",
        "model": "test-model",
        "base_url": "https://model.example/v1",
        "max_retries": 2,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]


class _RecordingProvider:
    provider_name = "recording"
    is_remote = False

    def __init__(self, *, fail_count: int = 0) -> None:
        self.fail_count = fail_count
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) <= self.fail_count:
            raise ModelTransportError("temporary model failure")
        return ModelResponse(
            text="ok",
            model=str(request.model),
            provider=self.provider_name,
            prompt_version=request.prompt_version,
        )

    def health_check(self) -> dict[str, object]:
        return {"ok": True}


class _HTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_model_client_normalizes_text_and_retries_only_retryable_failures() -> None:
    provider = _RecordingProvider(fail_count=1)
    sleeps: list[float] = []
    client = ModelClient(_config(), provider, sleep=sleeps.append)

    response = client.complete("extract memory")

    assert response.text == "ok"
    assert len(provider.requests) == 2
    assert provider.requests[0].messages == (ChatMessage(role="user", content="extract memory"),)
    assert provider.requests[0].model == "test-model"
    assert sleeps == [0.25]


def test_model_client_preserves_domain_request_metadata_without_sending_policy_to_provider() -> None:
    provider = _RecordingProvider()
    client = ModelClient(_config(), provider, sleep=lambda _seconds: None)
    request = ChatRequest(
        messages=(ChatMessage(role="user", content="hello"),),
        prompt_version="memory-v1",
        metadata={"purpose": "memory_extraction"},
    )

    response = client.complete(request)

    assert response.prompt_version == "memory-v1"
    assert provider.requests[0].metadata == {"purpose": "memory_extraction"}


def test_factory_resolves_secret_only_when_building_provider() -> None:
    config = _config(api_key_env="MODEL_TEST_TOKEN")

    with pytest.raises(ModelConfigurationError, match="MODEL_TEST_TOKEN"):
        ModelClientFactory().create(config, environ={})

    client = ModelClientFactory().create(config, environ={"MODEL_TEST_TOKEN": "secret"})

    assert isinstance(client.provider, OpenAICompatibleProvider)
    assert client.provider.api_key == "secret"
    assert "secret" not in repr(config)


def test_openai_compatible_provider_treats_loopback_service_as_local() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local-vllm",
        base_url="http://127.0.0.1:8000/v1",
        opener=lambda *_args, **_kwargs: None,
    )

    assert provider.is_remote is False


def test_openai_compatible_provider_builds_protocol_request_and_normalizes_response() -> None:
    captured: list[tuple[Any, float]] = []

    def open_request(request, *, timeout):  # noqa: ANN001, ANN202
        captured.append((request, timeout))
        return _HTTPResponse(
            {
                "model": "served-model",
                "choices": [{"message": {"content": "model result"}}],
                "usage": {"total_tokens": 12},
            }
        )

    provider = OpenAICompatibleProvider(
        provider_name="openai-compatible",
        base_url="https://model.example/v1",
        api_key="secret-token",
        timeout_seconds=9,
        opener=open_request,
    )
    request = ChatRequest(
        messages=(ChatMessage(role="user", content="hello"),),
        model="requested-model",
        temperature=0.2,
        metadata={"private_policy": "must-not-leak"},
    )

    response = provider.complete(request)

    wire_request, timeout = captured[0]
    payload = json.loads(wire_request.data.decode())
    assert wire_request.full_url == "https://model.example/v1/chat/completions"
    assert wire_request.headers["Authorization"] == "Bearer secret-token"
    assert timeout == 9
    assert payload == {
        "messages": [{"content": "hello", "role": "user"}],
        "model": "requested-model",
        "temperature": 0.2,
    }
    assert "private_policy" not in wire_request.data.decode()
    assert response.text == "model result"
    assert response.model == "served-model"
    assert response.usage == {"total_tokens": 12}


def test_runtime_injects_configured_client_into_memory_extraction(tmp_path) -> None:
    config = _config()
    client = ModelClient(config, _RecordingProvider(), sleep=lambda _seconds: None)

    container = build_test_runtime(
        RuntimeConfig(root=str(tmp_path), model=config),
        model_client=client,
    )

    assert container.stores.model_client is client
    assert isinstance(container.session.commit_service.memory_planner.extractor, LLMMemoryExtractorBackend)
    assert container.session.commit_service.memory_planner.extractor.provider is client
