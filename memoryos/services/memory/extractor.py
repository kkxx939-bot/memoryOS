from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memoryos.domain.memory.memory_item import MEMORY_TYPES
from memoryos.ports.providers.chat_provider import ChatProvider, ModelResponse

MEMORY_ACTIONS = {"add", "update", "delete", "ignore"}


@dataclass
class MemoryOperation:
    action: str
    memory_type: str
    title: str
    text: str
    tags: list[str]
    confidence: float = 0.7
    target: str | None = None
    rationale: str = ""
    page_id: int | None = None
    links: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action not in MEMORY_ACTIONS:
            raise ValueError(f"Unknown memory action: {self.action}")
        if self.memory_type not in MEMORY_TYPES:
            known = ", ".join(sorted(MEMORY_TYPES))
            raise ValueError(f"Unknown memory type: {self.memory_type}. Known types: {known}")
        if not isinstance(self.tags, list):
            raise ValueError("Memory operation tags must be a list")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Memory operation confidence must be in [0, 1]")
        if not isinstance(self.links, list):
            raise ValueError("Memory operation links must be a list")


ExtractedMemory = MemoryOperation


@dataclass
class MemoryExtractionResult:
    accepted: list[MemoryOperation] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_model_output: str = ""
    extractor_version: str = "json_llm_extractor_v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [operation_to_dict(operation) for operation in self.accepted],
            "rejected": self.rejected,
            "pending": self.pending,
            "errors": self.errors,
            "raw_model_output": self.raw_model_output,
            "extractor_version": self.extractor_version,
        }


def operation_to_dict(operation: MemoryOperation) -> dict[str, Any]:
    return {
        "action": operation.action,
        "memory_type": operation.memory_type,
        "title": operation.title,
        "text": operation.text,
        "tags": operation.tags,
        "confidence": operation.confidence,
        "target": operation.target,
        "rationale": operation.rationale,
        "page_id": operation.page_id,
        "links": operation.links,
    }


class RuleBasedExtractor:
    """Temporary extractor; replace with an LLM extractor after the schema stabilizes."""

    markers = ("记住：", "记住:", "remember:", "Remember:")
    injected_context_pattern = re.compile(
        r"<personal-memory\b[^>]*>.*?</personal-memory>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def extract(self, messages: list[dict[str, str]]) -> list[MemoryOperation]:
        extracted: list[MemoryOperation] = []
        for message in messages:
            text = self.strip_injected_context(message.get("text", ""))
            for marker in self.markers:
                if marker in text:
                    memory_text = text.split(marker, 1)[1].strip()
                    if memory_text:
                        extracted.append(self._classify(memory_text))
                    break
        return extracted

    def _classify(self, text: str) -> MemoryOperation:
        lowered = text.lower()
        if any(word in text for word in ("喜欢", "偏好", "不喜欢", "希望")):
            memory_type = "preference"
        elif any(word in text for word in ("通常", "经常", "习惯", "一般")):
            memory_type = "habit"
        elif any(word in text for word in ("允许", "必须", "不要", "不能")):
            memory_type = "policy"
        elif "feedback" in lowered or "反馈" in text:
            memory_type = "feedback"
        else:
            memory_type = "event"
        title = text[:24].strip(" ，,。.") or "extracted memory"
        tags = [memory_type]
        if memory_type == "policy":
            tags.append("explicit_user_intent")
        return MemoryOperation(
            action="add",
            memory_type=memory_type,
            title=title,
            text=text,
            tags=tags,
        )

    def strip_injected_context(self, text: str) -> str:
        return self.injected_context_pattern.sub("", text)


class JsonLLMMemoryExtractor:
    """Extractor boundary for real LLMs.

    The provider can be an OpenAI client, a local vLLM client, or any other model adapter.
    This class owns the prompt contract and JSON validation, not the model transport.
    """

    injected_context_pattern = RuleBasedExtractor.injected_context_pattern

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider
        self.last_result = MemoryExtractionResult()

    def extract(self, messages: list[dict[str, str]]) -> list[MemoryOperation]:
        clean_messages = [
            {**message, "text": self.strip_injected_context(message.get("text", ""))}
            for message in messages
        ]
        prompt = self.build_prompt(clean_messages)
        response = self.provider.complete(prompt)
        text = response.text if isinstance(response, ModelResponse) else str(response)
        result = self.parse_response_detailed(text)
        self.last_result = result
        return result.accepted

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        transcript = "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('text', '')}"
            for message in messages
            if message.get("text")
        )
        memory_types = ", ".join(sorted(MEMORY_TYPES))
        return f"""You are the memory extraction layer for a personal memory system.

Extract only durable, useful memories. Do not store injected <personal-memory> context.
Return strict JSON. No markdown. No commentary.

Allowed actions: add, update, delete, ignore.
Allowed memory_type values: {memory_types}.

Schema:
{{
  "operations": [
    {{
      "action": "add",
      "memory_type": "habit",
      "title": "short title",
      "text": "complete memory content",
      "tags": ["short", "labels"],
      "confidence": 0.0,
      "target": null,
      "page_id": 100,
      "links": [
        {{"to": "user/gulf/profile/user-profile.md", "link_type": "related_to", "description": "why linked"}}
      ],
      "rationale": "why this should be stored"
    }}
  ]
}}

Use update only when the input clearly revises an existing memory and target is known.
Use delete when the user explicitly asks to forget/remove a known memory and target is known.
For policy memories, add tag "explicit_user_intent" only when the user explicitly states the rule or permission boundary.
Use ignore for transient chatter, duplicate injected context, or low-value facts.

Transcript:
{transcript}
"""

    def parse_response(self, response: str) -> list[MemoryOperation]:
        return self.parse_response_detailed(response).accepted

    def parse_response_detailed(self, response: str) -> MemoryExtractionResult:
        result = MemoryExtractionResult(raw_model_output=response)
        try:
            payload = self._load_json(response)
        except ValueError as exc:
            result.errors.append(str(exc))
            return result
        if isinstance(payload, dict):
            raw_operations = payload.get("operations", [])
        elif isinstance(payload, list):
            raw_operations = payload
        else:
            raw_operations = []
        if not isinstance(raw_operations, list):
            result.errors.append("LLM memory response must contain an operations list")
            return result
        for index, raw in enumerate(raw_operations):
            if not isinstance(raw, dict):
                result.rejected.append({"index": index, "raw": raw, "error": "operation must be an object"})
                continue
            try:
                operation = self._operation_from_raw(raw)
            except Exception as exc:
                result.rejected.append({"index": index, "raw": raw, "error": str(exc)})
                continue
            if operation.action != "ignore" and operation.confidence < 0.45:
                result.pending.append(
                    {
                        "index": index,
                        "operation": operation_to_dict(operation),
                        "reason": "low_confidence",
                    }
                )
                continue
            result.accepted.append(operation)
        return result

    def _operation_from_raw(self, raw: dict[str, Any]) -> MemoryOperation:
        action = str(raw.get("action", "ignore")).strip()
        memory_type = str(raw.get("memory_type", "event")).strip()
        title = str(raw.get("title", "")).strip()
        text = str(raw.get("text", "")).strip()
        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError("tags must be a list")
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if action == "ignore":
            title = title or "ignored"
            text = text or str(raw.get("rationale", "ignored"))
            tags = tags or ["ignore"]
        if action == "delete":
            title = title or "delete memory"
            text = text or str(raw.get("rationale", "delete requested"))
        if not title or not text:
            raise ValueError("Memory add/update operations require title and text")
        page_id = None
        if raw.get("page_id") is not None:
            try:
                page_id = int(raw["page_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError("page_id must be an integer") from exc
        return MemoryOperation(
            action=action,
            memory_type=memory_type,
            title=title,
            text=text,
            tags=tags,
            confidence=confidence,
            target=raw.get("target"),
            rationale=str(raw.get("rationale", "")),
            page_id=page_id,
            links=raw.get("links", []) if isinstance(raw.get("links", []), list) else [],
        )

    def strip_injected_context(self, text: str) -> str:
        return self.injected_context_pattern.sub("", text)

    def _load_json(self, response: str) -> dict | list:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM memory response is not valid JSON: {exc}") from exc
