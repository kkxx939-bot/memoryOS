from __future__ import annotations

import json
import re
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.ids import stable_hash
from memoryos.memory.extraction.memory_extractor import ExtractionResult, MemoryExtractor
from memoryos.memory.model.memory import Memory, MemoryKind
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.ports.providers.chat_provider import ChatProvider, ModelResponse


class LLMMemoryExtractor(MemoryExtractor):
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def extract(self, session_archive) -> ExtractionResult:
        prompt = self._prompt(session_archive)
        response = self.provider.complete(prompt)
        text = response.text if isinstance(response, ModelResponse) else str(response)
        return self.parse_response(text, user_id=session_archive.user_id, session_id=session_archive.session_id)

    def parse_response(self, response: str, user_id: str, session_id: str = "") -> ExtractionResult:
        result = ExtractionResult(raw_output=response, extractor_version="llm_context_memory_extractor_v1")
        try:
            payload = self._load_json(response)
        except ValueError as exc:
            result.rejected.append({"error": str(exc), "raw": response})
            return result
        if isinstance(payload, dict):
            raw_operations = payload.get("operations", [])
        else:
            raw_operations = payload
        if not isinstance(raw_operations, list):
            result.rejected.append({"error": "operations must be a list", "raw": payload})
            return result
        for index, raw in enumerate(raw_operations):
            if not isinstance(raw, dict):
                result.rejected.append({"index": index, "error": "operation must be an object", "raw": raw})
                continue
            operation = self._operation(raw, user_id=user_id, session_id=session_id)
            if isinstance(operation, dict):
                result.rejected.append({"index": index, **operation})
                continue
            if operation.status == OperationStatus.PENDING:
                result.pending.append(operation)
            else:
                result.accepted.append(operation)
        return result

    def _operation(self, raw: dict[str, Any], user_id: str, session_id: str) -> ContextOperation | dict:
        action_text = str(raw.get("action", "add")).strip()
        action_map = {
            "add": OperationAction.ADD,
            "update": OperationAction.UPDATE,
            "delete": OperationAction.DELETE,
            "ignore": OperationAction.REJECT,
            "confirm": OperationAction.CONFIRM,
            "reject": OperationAction.REJECT,
        }
        action = action_map.get(action_text)
        if action is None:
            return {"error": f"unknown action: {action_text}", "raw": raw}
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            return {"error": "confidence must be numeric", "raw": raw}
        target_uri = raw.get("target_uri") or raw.get("target")
        status = OperationStatus.CANDIDATE
        if action in {OperationAction.UPDATE, OperationAction.DELETE} and not target_uri:
            status = OperationStatus.PENDING
        if self._is_sensitive(raw):
            status = OperationStatus.PENDING
        content = str(raw.get("text", raw.get("content", ""))).strip()
        title = str(raw.get("title", content[:32] or "memory")).strip()
        kind = MemoryKind(str(raw.get("memory_kind", raw.get("kind", MemoryKind.EXPLICIT.value))))
        uri = str(target_uri or f"memoryos://user/{user_id}/memories/{kind.value}/{stable_hash([title, content], 16)}")
        memory = Memory(uri=uri, user_id=user_id, title=title, content=content, kind=kind, confidence=confidence)
        return ContextOperation(
            user_id=user_id,
            context_type=ContextType.MEMORY,
            action=action,
            target_uri=uri,
            payload={"context_object": memory.to_context_object().to_dict(), "content": content},
            evidence=[{"source": "llm_extractor"}],
            confidence=confidence,
            source_session_id=session_id,
            status=status,
        )

    def _prompt(self, session_archive) -> str:
        messages = "\n".join(str(item) for item in getattr(session_archive, "messages", []))
        return f"Extract durable memory operations as strict JSON with operations list.\n{messages}"

    def _is_sensitive(self, raw: dict[str, Any]) -> bool:
        tags = [str(tag).lower() for tag in raw.get("tags", []) if isinstance(tag, str)]
        text = json.dumps(raw, ensure_ascii=False).lower()
        return "sensitive" in tags or "password" in text or "api_key" in text

    def _load_json(self, response: str) -> dict | list:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM memory response is not valid JSON: {exc}") from exc
        if not isinstance(payload, (dict, list)):
            raise ValueError("LLM memory response must be an object or list")
        return payload
