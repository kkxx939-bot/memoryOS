from __future__ import annotations

import time
from dataclasses import dataclass

from memoryos.adapters.providers.openai_compatible.http_client import post_json
from memoryos.ports.providers.chat_provider import ChatMessage, ChatRequest, ModelResponse
from memoryos.ports.providers.provider_errors import ProviderBadResponse


@dataclass
class OpenAICompatibleChatProvider:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout: float = 60.0
    temperature: float = 0.0
    retries: int = 2
    backoff_seconds: float = 0.5
    provider_name: str = "openai_compatible"

    def complete(self, request: ChatRequest | str) -> ModelResponse | str:
        if isinstance(request, str):
            prompt_version = None
            messages = [ChatMessage(role="user", content=request)]
            return_text_only = True
        else:
            prompt_version = request.prompt_version
            messages = request.messages
            return_text_only = False
        started = time.monotonic()
        response = post_json(
            base_url=self.base_url,
            endpoint="/chat/completions",
            payload={
                "model": self.model,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "temperature": self.temperature,
            },
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
            provider=self.provider_name,
        )
        choices = response.get("choices") or []
        if not choices:
            raise ProviderBadResponse("Chat completion response has no choices", provider=self.provider_name)
        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content") or first.get("text")
        if not isinstance(content, str) or not content.strip():
            raise ProviderBadResponse("Chat completion response has no text content", provider=self.provider_name)
        if return_text_only:
            return content
        return ModelResponse(
            text=content,
            model=self.model,
            provider=self.provider_name,
            prompt_version=prompt_version,
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            latency_ms=int((time.monotonic() - started) * 1000),
            raw=response,
        )

    def health_check(self) -> dict:
        return {"status": "configured", "provider": self.provider_name, "model": self.model}
