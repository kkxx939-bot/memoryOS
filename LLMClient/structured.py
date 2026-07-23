"""支持分层 JSON 修复和严格校验的结构化模型调用。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Generic, TypeVar, cast

from LLMClient.client import LLMClient
from LLMClient.contracts import (
    ChatMessage,
    ChatRequest,
    ModelResponse,
    ModelStructuredOutputError,
    ResponseFormat,
)
from LLMClient.json_parser import JSONParseMode, parse_json_response
from LLMClient.schema_validation import validate_json_schema

T = TypeVar("T")


@dataclass(frozen=True)
class StructuredResponse(Generic[T]):
    """通过校验的值，以及原始响应和解析审计元数据。"""

    value: T
    response: ModelResponse
    raw_text: str
    parse_mode: JSONParseMode

    @property
    def repaired(self) -> bool:
        return self.parse_mode != "strict"


class StructuredLLMClient:
    """将 Schema 提示和校验逻辑与传输供应商分离。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        allow_json_repair: bool = True,
        validation_retries: int = 1,
    ) -> None:
        if not isinstance(client, LLMClient):
            raise TypeError("client must be LLMClient")
        if not isinstance(allow_json_repair, bool):
            raise TypeError("allow_json_repair must be boolean")
        if (
            not isinstance(validation_retries, int)
            or isinstance(validation_retries, bool)
            or not 0 <= validation_retries <= 5
        ):
            raise ValueError("validation_retries must be between zero and five")
        self.client = client
        self.allow_json_repair = allow_json_repair
        self.validation_retries = validation_retries

    def complete_json(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str = "structured_response",
        validator: Callable[[object], T] | None = None,
    ) -> StructuredResponse[T | object]:
        prepared = self._prepare(request, schema=schema, name=name)
        last_error: Exception | None = None
        for attempt in range(self.validation_retries + 1):
            response = self.client.complete(prepared)
            try:
                return self._validate_response(response, schema=schema, validator=validator)
            except (TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.validation_retries:
                    prepared = self._correction_request(prepared, response, exc)
        raise ModelStructuredOutputError(
            f"model failed structured output validation after {self.validation_retries + 1} attempt(s)"
        ) from last_error

    async def complete_json_async(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str = "structured_response",
        validator: Callable[[object], T] | None = None,
    ) -> StructuredResponse[T | object]:
        prepared = self._prepare(request, schema=schema, name=name)
        last_error: Exception | None = None
        for attempt in range(self.validation_retries + 1):
            response = await self.client.complete_async(prepared)
            try:
                return self._validate_response(response, schema=schema, validator=validator)
            except (TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.validation_retries:
                    prepared = self._correction_request(prepared, response, exc)
        raise ModelStructuredOutputError(
            f"model failed structured output validation after {self.validation_retries + 1} attempt(s)"
        ) from last_error

    def complete_model(
        self,
        request: ChatRequest | str,
        *,
        model_class: type[T],
        name: str | None = None,
    ) -> StructuredResponse[T]:
        schema, validator = _model_contract(model_class)
        result = self.complete_json(
            request,
            schema=schema,
            name=name or model_class.__name__,
            validator=validator,
        )
        return cast(StructuredResponse[T], result)

    async def complete_model_async(
        self,
        request: ChatRequest | str,
        *,
        model_class: type[T],
        name: str | None = None,
    ) -> StructuredResponse[T]:
        schema, validator = _model_contract(model_class)
        result = await self.complete_json_async(
            request,
            schema=schema,
            name=name or model_class.__name__,
            validator=validator,
        )
        return cast(StructuredResponse[T], result)

    def _prepare(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str,
    ) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("structured model prompt cannot be empty")
            request = ChatRequest(messages=(ChatMessage(role="user", content=request),))
        if not isinstance(request, ChatRequest):
            raise TypeError("structured request must be ChatRequest or non-empty text")
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("structured output schema must be a non-empty object")
        try:
            schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2)
        except (TypeError, ValueError) as exc:
            raise ValueError("structured output schema must be JSON serializable") from exc
        instruction = ChatMessage(
            role="system",
            content=(
                "Return exactly one JSON value that satisfies the following JSON Schema. "
                "Do not add Markdown fences, commentary, or fields not declared by the schema.\n"
                f"{schema_text}"
            ),
        )
        response_format = ResponseFormat(name=name, schema=schema, strict=True)
        return replace(
            request,
            messages=(instruction, *request.messages),
            response_format=response_format,
        )

    @staticmethod
    def _correction_request(
        request: ChatRequest,
        response: ModelResponse,
        error: Exception,
    ) -> ChatRequest:
        messages = list(request.messages)
        if response.content:
            messages.append(ChatMessage(role="assistant", content=response.content))
        detail = str(error).replace("\n", " ")[:512]
        messages.append(
            ChatMessage(
                role="user",
                content=(f"The previous JSON response was invalid ({detail}). Return one corrected JSON value only."),
            )
        )
        return replace(request, messages=tuple(messages))

    def _validate_response(
        self,
        response: ModelResponse,
        *,
        schema: Mapping[str, object],
        validator: Callable[[object], T] | None,
    ) -> StructuredResponse[T | object]:
        if response.finish_reason == "length":
            raise ValueError("structured model response was truncated")
        if response.finish_reason in {"content_filter", "safety"}:
            raise ValueError("structured model response was blocked by content safety")
        if not response.content:
            raise ValueError("structured model response has no text content")
        parsed = parse_json_response(response.content, allow_repair=self.allow_json_repair)
        validate_json_schema(parsed.value, schema)
        value: T | object = parsed.value
        if validator is not None:
            value = validator(parsed.value)
        return StructuredResponse(
            value=value,
            response=response,
            raw_text=response.content,
            parse_mode=parsed.mode,
        )


def _model_contract(model_class: type[T]) -> tuple[Mapping[str, object], Callable[[object], T]]:
    schema_builder = getattr(model_class, "model_json_schema", None)
    model_validator = getattr(model_class, "model_validate", None)
    if not callable(schema_builder) or not callable(model_validator):
        raise TypeError("model_class must provide model_json_schema() and model_validate()")
    schema = schema_builder()
    if not isinstance(schema, Mapping):
        raise TypeError("model_json_schema() must return an object")

    def validate(value: object) -> T:
        return cast(T, model_validator(value))

    return schema, validate


__all__ = ["StructuredLLMClient", "StructuredResponse"]
