from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive


class MemoryModelProvider(Protocol):
    def complete(self, prompt: str) -> str: ...


class MemoryExtractionPromptBuilder:
    def build(self, archive: SessionArchive, schemas: Sequence[MemoryTypeSchema]) -> str:
        schema_names = ", ".join(schema.memory_type.value for schema in schemas)
        messages = "\n".join(json.dumps(message, ensure_ascii=False, sort_keys=True) for message in archive.messages)
        return (
            "Extract durable memory candidates as JSON. "
            "Return an object with a candidates array. "
            f"Allowed memory_type values: {schema_names}. "
            "Do not output operations or target URIs.\n"
            f"{messages}"
        )


class MemoryExtractionJsonParser:
    VALID_SOURCE_ROLES = {"user", "assistant", "agent", "tool", "unknown"}

    def parse(
        self,
        response: str,
        *,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemoryCandidateDraft]:
        payload = self._load_json(response)
        if isinstance(payload, dict):
            raw_candidates = payload.get("candidates", [])
        else:
            raw_candidates = payload
        if not isinstance(raw_candidates, list):
            raise ValueError("memory extraction candidates must be a list")
        allowed = {schema.memory_type for schema in schemas}
        candidates = []
        adapter_id = adapter_id_from_archive(archive)
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                raise ValueError(f"candidate[{index}] must be an object")
            memory_type = MemoryType(str(raw.get("memory_type", "")))
            if memory_type not in allowed:
                raise ValueError(f"candidate[{index}] memory_type is not allowed: {memory_type.value}")
            role = str(raw.get("source_role", "unknown") or "unknown").lower()
            if role not in self.VALID_SOURCE_ROLES:
                raise ValueError(f"candidate[{index}] source_role is not allowed: {role}")
            fields = raw.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError(f"candidate[{index}] fields must be an object")
            content = str(raw.get("content", raw.get("text", ""))).strip()
            title = str(raw.get("title", content[:64] or memory_type.value)).strip()
            if not content:
                raise ValueError(f"candidate[{index}] content is required")
            candidates.append(
                MemoryCandidateDraft(
                    memory_type=memory_type,
                    title=title,
                    content=content,
                    fields=fields,
                    confidence=float(raw.get("confidence", 0.5)),
                    source_role=role,
                    source_adapter_id=str(raw.get("source_adapter_id") or adapter_id),
                    source_session_id=str(raw.get("source_session_id") or archive.session_id),
                    source_message_ids=[str(item) for item in raw.get("source_message_ids", []) if item],
                    evidence=[item for item in raw.get("evidence", []) if isinstance(item, dict)],
                    suggested_retrieval_views=[],
                    merge_key=str(raw.get("merge_key", "")),
                    reason=str(raw.get("reason", "llm_candidate")),
                )
            )
        return candidates

    def _load_json(self, response: str) -> dict | list:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM memory response is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict | list):
            raise ValueError("LLM memory response must be an object or list")
        return payload


class LLMMemoryExtractorBackend:
    def __init__(
        self,
        provider: MemoryModelProvider,
        prompt_builder: MemoryExtractionPromptBuilder | None = None,
        parser: MemoryExtractionJsonParser | None = None,
    ) -> None:
        self.provider = provider
        self.prompt_builder = prompt_builder or MemoryExtractionPromptBuilder()
        self.parser = parser or MemoryExtractionJsonParser()

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemoryCandidateDraft]:
        prompt = self.prompt_builder.build(archive, schemas)
        response = self.provider.complete(prompt)
        return self.parser.parse(response, archive=archive, schemas=schemas)


@dataclass
class FakeMemoryModelProvider:
    response: str
    prompts: list[str] | None = None

    def complete(self, prompt: str) -> str:
        if self.prompts is not None:
            self.prompts.append(prompt)
        return self.response
