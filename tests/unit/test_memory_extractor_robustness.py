from __future__ import annotations

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.extraction.llm_memory_extractor import LLMMemoryExtractor
from memoryos.providers.llm.base import ChatRequest, ModelResponse


class Provider:
    provider_name = "fake"
    model = "fake"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, request: ChatRequest | str) -> ModelResponse | str:
        return self.text

    def health_check(self) -> dict:
        return {"ok": True}


def _archive() -> SessionArchive:
    return SessionArchive(user_id="u1", session_id="s1", archive_uri="memoryos://user/u1/sessions/history/s1")


def test_bad_operation_does_not_block_accepted_operation() -> None:
    extractor = LLMMemoryExtractor(Provider('{"operations":[{"action":"add","content":"likes cool room","confidence":0.9},{"action":"unknown","content":"bad"}]}'))

    result = extractor.extract(_archive())

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1


def test_invalid_confidence_rejects_only_current_operation() -> None:
    extractor = LLMMemoryExtractor(Provider('{"operations":[{"action":"add","content":"ok","confidence":0.5},{"action":"add","content":"bad","confidence":2}]}'))

    result = extractor.extract(_archive())

    assert len(result.accepted) == 1
    assert result.rejected[0]["error"] == "confidence must be between 0 and 1"


def test_update_delete_without_target_go_pending_and_sensitive_is_pending() -> None:
    extractor = LLMMemoryExtractor(
        Provider('{"operations":[{"action":"update","content":"needs target"},{"action":"delete","content":"needs target"},{"action":"add","content":"api_key secret","confidence":0.8}]}')
    )

    result = extractor.extract(_archive())

    assert len(result.pending) == 3
    assert result.rejected == []


def test_non_json_response_rejects_batch_without_raising() -> None:
    result = LLMMemoryExtractor(Provider("not json")).extract(_archive())

    assert result.accepted == []
    assert result.pending == []
    assert result.rejected
