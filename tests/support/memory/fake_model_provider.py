"""记忆提取测试使用的可观测模型 Provider。"""

from __future__ import annotations

from infrastructure.model import ChatRequest


class FakeMemoryModelProvider:
    provider_name = "fake-memory-model"
    model = "fake-memory-model"
    is_remote = False

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, request: ChatRequest | str) -> str:
        self.calls += 1
        if isinstance(request, ChatRequest):
            self.prompts.append("\n".join(message.content for message in request.messages))
        else:
            self.prompts.append(request)
        return self.response

    def health_check(self) -> dict[str, bool]:
        return {"ok": True}


__all__ = ["FakeMemoryModelProvider"]
