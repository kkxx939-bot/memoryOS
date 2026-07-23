"""OpenAI 兼容供应商适配器共用的序列化和规范化逻辑。"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from LLMClient.contracts import (
    ChatMessage,
    ChatRequest,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ProviderCapabilities,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from LLMClient.json_parser import parse_json_response

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def build_openai_payload(
    request: ChatRequest,
    *,
    model: str,
    capabilities: ProviderCapabilities,
    configured_reasoning: bool,
    default_max_output_tokens: int | None,
    extra_body: Mapping[str, object],
    stream: bool,
) -> dict[str, object]:
    """把稳定请求转换为通用 OpenAI Chat Completions 格式。"""

    payload: dict[str, object] = {
        "model": model,
        "messages": [_message_payload(message) for message in request.messages],
    }
    reasoning = configured_reasoning or _is_reasoning_model(model) or request.reasoning is not None
    if reasoning:
        if request.reasoning and request.reasoning.effort:
            payload["reasoning_effort"] = request.reasoning.effort
    elif request.temperature is not None:
        payload["temperature"] = float(request.temperature)

    max_tokens = request.max_output_tokens or default_max_output_tokens
    if max_tokens is not None:
        payload["max_completion_tokens" if reasoning else "max_tokens"] = max_tokens
    if request.tools:
        if not capabilities.tools:
            raise ModelResponseError("selected provider does not support tool calls")
        payload["tools"] = [_tool_payload(tool) for tool in request.tools]
        payload["tool_choice"] = request.tool_choice or "auto"
    if request.response_format is not None and capabilities.native_structured_output:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_format.name,
                "strict": request.response_format.strict,
                "schema": dict(request.response_format.schema),
            },
        }
    payload.update(extra_body)
    if stream:
        payload["stream"] = True
    return payload


def normalize_response(
    source: Mapping[str, object],
    *,
    provider: str,
    configured_model: str,
    prompt_version: str | None,
    started: float,
) -> ModelResponse:
    choices = source.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ModelResponseError("model response has no choices")
    first = choices[0]
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ModelResponseError("model response has no assistant message")
    content = _message_text(message.get("content"))
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if not content and not tool_calls:
        raise ModelResponseError("model response has neither content nor tool calls")
    reasoning_content = _optional_text(message.get("reasoning_content", message.get("reasoning")))
    finish_reason = str(first.get("finish_reason") or "stop")
    return ModelResponse(
        content=content,
        model=str(source.get("model") or configured_model),
        provider=provider,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
        usage=normalize_usage(source.get("usage")),
        prompt_version=prompt_version,
        latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        raw=source,
    )


def normalize_stream_chunk(source: Mapping[str, object]) -> tuple[ModelStreamEvent, ...]:
    events: list[ModelStreamEvent] = []
    choices = source.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        first = choices[0]
        delta = first.get("delta")
        if isinstance(delta, Mapping):
            content = _optional_text(delta.get("content"))
            if content:
                events.append(ModelStreamEvent(kind="content_delta", content_delta=content, raw=source))
            reasoning = _optional_text(delta.get("reasoning_content", delta.get("reasoning")))
            if reasoning:
                events.append(ModelStreamEvent(kind="reasoning_delta", reasoning_delta=reasoning, raw=source))
            raw_tool_calls = delta.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for fallback_index, raw_tool_call in enumerate(raw_tool_calls):
                    if not isinstance(raw_tool_call, Mapping):
                        continue
                    function = raw_tool_call.get("function")
                    function = function if isinstance(function, Mapping) else {}
                    raw_index = raw_tool_call.get("index", fallback_index)
                    index = raw_index if isinstance(raw_index, int) else fallback_index
                    events.append(
                        ModelStreamEvent(
                            kind="tool_call_delta",
                            tool_call_index=index,
                            tool_call_id=_optional_text(raw_tool_call.get("id")),
                            tool_name=_optional_text(function.get("name")),
                            tool_arguments_delta=_optional_text(function.get("arguments")),
                            raw=source,
                        )
                    )
        finish_reason = _optional_text(first.get("finish_reason"))
        if finish_reason:
            events.append(ModelStreamEvent(kind="done", finish_reason=finish_reason, raw=source))
    if isinstance(source.get("usage"), Mapping):
        events.append(ModelStreamEvent(kind="usage", usage=normalize_usage(source["usage"]), raw=source))
    return tuple(events)


def normalize_usage(source: object) -> TokenUsage:
    if not isinstance(source, Mapping):
        return TokenUsage()
    prompt_tokens = _non_negative_int(source.get("prompt_tokens", source.get("input_tokens")))
    completion_tokens = _non_negative_int(source.get("completion_tokens", source.get("output_tokens")))
    total_tokens = _non_negative_int(source.get("total_tokens"))
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    prompt_details = source.get("prompt_tokens_details")
    completion_details = source.get("completion_tokens_details")
    cached = _mapping_int(prompt_details, "cached_tokens")
    reasoning = _mapping_int(completion_details, "reasoning_tokens")
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
        details=dict(source),
    )


def object_to_mapping(source: object) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(source, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return dict(result)
    raise ModelResponseError("provider returned a response that cannot be normalized")


def _message_payload(message: ChatMessage) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters),
    }
    if tool.strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _parse_tool_calls(source: object) -> tuple[ToolCall, ...]:
    if source is None:
        return ()
    if not isinstance(source, list):
        raise ModelResponseError("model tool_calls must be an array")
    result: list[ToolCall] = []
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ModelResponseError(f"model tool_calls[{index}] must be an object")
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise ModelResponseError(f"model tool_calls[{index}] has no function")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        elif isinstance(raw_arguments, str):
            try:
                parsed = parse_json_response(raw_arguments).value
            except ValueError as exc:
                raise ModelResponseError(f"model tool_calls[{index}] arguments are not valid JSON") from exc
            if not isinstance(parsed, Mapping):
                raise ModelResponseError(f"model tool_calls[{index}] arguments must be an object")
            arguments = dict(parsed)
        else:
            raise ModelResponseError(f"model tool_calls[{index}] arguments must be JSON text")
        call_id = raw.get("id")
        name = function.get("name")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ModelResponseError(f"model tool_calls[{index}] id must be non-empty")
        if not isinstance(name, str) or not name.strip():
            raise ModelResponseError(f"model tool_calls[{index}] name must be non-empty")
        result.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            )
        )
    return tuple(result)


def _message_text(source: object) -> str | None:
    if source is None:
        return None
    if isinstance(source, str):
        return source or None
    if isinstance(source, list):
        parts: list[str] = []
        for item in source:
            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) or None
    raise ModelResponseError("model response content must be text or text parts")


def _optional_text(source: object) -> str | None:
    return source if isinstance(source, str) and source else None


def _non_negative_int(source: object) -> int:
    return source if isinstance(source, int) and not isinstance(source, bool) and source >= 0 else 0


def _mapping_int(source: object, key: str) -> int:
    return _non_negative_int(source.get(key)) if isinstance(source, Mapping) else 0


def _is_reasoning_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1].casefold()
    return name.startswith(_REASONING_MODEL_PREFIXES)


__all__ = [
    "build_openai_payload",
    "normalize_response",
    "normalize_stream_chunk",
    "normalize_usage",
    "object_to_mapping",
]
