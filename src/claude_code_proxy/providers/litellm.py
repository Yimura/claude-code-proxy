"""LiteLLM provider adapter."""

import asyncio
from copy import deepcopy
import json
import logging
import uuid
from typing import Any

import litellm

from ..config import Settings
from ..domain.models import (
    CompletionRequest, CompletionResponse, ImageBlock, StreamComplete, StreamError,
    StreamStart, TextBlock, TextDelta, TokenUsage, ToolInputDelta, ToolResultBlock,
    ToolUseBlock, ToolUseEnd, ToolUseStart,
)
from .base import ProviderError, protocol_error, stream_error_from_exception
from .usage import normalize_usage

logger = logging.getLogger(__name__)


def clean_gemini_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        if schema.get("type") == "string" and schema.get("format") not in (None, "enum", "date-time"):
            schema.pop("format")
        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema


def parse_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item) if isinstance(item, (dict, list)) else str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", json.dumps(content))
    return str(content)


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, settings: Settings, client=litellm) -> None:
        self._settings = settings
        self._client = client

    def build_request(self, request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": "\n\n".join(block.text for block in request.system)})
        for message in request.messages:
            messages.extend(self._convert_message(message.role, message.content))

        max_tokens = request.max_tokens
        if request.model.startswith(("openai/", "gemini/")):
            max_tokens = min(max_tokens, 16_384)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.model.startswith("anthropic/"):
            if request.thinking is not None:
                payload["thinking"] = request.thinking.model_dump(exclude_none=True)
            if request.output_config is not None:
                payload["output_config"] = request.output_config.model_dump(exclude_none=True)
        elif request.reasoning.enabled and request.reasoning.effort:
            payload["reasoning_effort"] = request.reasoning.effort
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.top_p:
            payload["top_p"] = request.top_p
        if request.top_k:
            payload["top_k"] = request.top_k
        self._add_tools(payload, request)
        self._apply_auth(payload, request.model)
        if request.model.startswith("openai/"):
            self._normalize_openai_messages(payload["messages"])
        return payload

    def _convert_message(self, role, content):
        if len(content) == 1 and isinstance(content[0], TextBlock):
            return [{"role": role, "content": content[0].text}]
        if role == "user" and any(isinstance(block, ToolResultBlock) for block in content):
            parts = []
            for block in content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
                elif isinstance(block, ToolResultBlock):
                    parts.append(f"Tool result for {block.tool_use_id}:\n{parse_tool_result_content(block.content)}")
            return [{"role": role, "content": "\n".join(parts).strip()}]
        blocks = []
        for block in content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                blocks.append({"type": "image", "source": deepcopy(block.source)})
            elif isinstance(block, ToolUseBlock):
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": deepcopy(block.input)})
            elif isinstance(block, ToolResultBlock):
                blocks.append({"type": "tool_result", "tool_use_id": block.tool_use_id, "content": [{"type": "text", "text": parse_tool_result_content(block.content)}]})
        return [{"role": role, "content": blocks}]

    def _add_tools(self, payload, request):
        tools = []
        for tool in request.tools:
            if tool.input_schema is None:
                continue
            schema = deepcopy(tool.input_schema)
            if request.model.startswith("gemini/"):
                schema = clean_gemini_schema(schema)
            tools.append({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": schema}})
        if request.tools:
            payload["tools"] = tools
        choice = request.tool_choice
        if choice is None:
            return
        if choice.type == "auto":
            payload["tool_choice"] = "auto"
        elif choice.type == "any":
            payload["tool_choice"] = "any"
        elif choice.type == "tool" and choice.name in {tool["function"]["name"] for tool in tools}:
            payload["tool_choice"] = {"type": "function", "function": {"name": choice.name}}
        else:
            payload["tool_choice"] = "auto"

    def _apply_auth(self, payload, model):
        if model.startswith("openai/"):
            payload["api_key"] = self._settings.openai_api_key
            if self._settings.openai_base_url:
                payload["api_base"] = self._settings.openai_base_url
        elif model.startswith("gemini/"):
            if self._settings.use_vertex_auth:
                payload.update(vertex_project=self._settings.vertex_project, vertex_location=self._settings.vertex_location, custom_llm_provider="vertex_ai")
            else:
                payload["api_key"] = self._settings.gemini_api_key
        else:
            payload["api_key"] = self._settings.anthropic_api_key

    @staticmethod
    def _normalize_openai_messages(messages):
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                text = []
                for block in content:
                    if block.get("type") == "text":
                        text.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        text.append(parse_tool_result_content(block.get("content", [])))
                    elif block.get("type") == "tool_use":
                        text.append(f"[Tool: {block.get('name', 'unknown')} (ID: {block.get('id', 'unknown')})]\nInput: {json.dumps(block.get('input', {}))}")
                    elif block.get("type") == "image":
                        text.append("[Image content - not displayed in text format]")
                message["content"] = "\n".join(text).strip() or "..."
            elif content is None:
                message["content"] = "..."
            for key in list(message):
                if key not in {"role", "content", "name", "tool_call_id", "tool_calls"}:
                    del message[key]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response = await asyncio.to_thread(self._client.completion, **self.build_request(request, stream=False))
            return self._normalize_response(response, request)
        except ProviderError:
            raise
        except Exception as error:
            raise self._provider_error(error) from error

    def _normalize_response(self, response, request):
        data = response if isinstance(response, dict) else response.model_dump() if hasattr(response, "model_dump") else response
        choices = data.get("choices", []) if isinstance(data, dict) else response.choices
        choice = choices[0] if choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else choice.message
        content_text = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        finish = choice.get("finish_reason", "stop") if isinstance(choice, dict) else getattr(choice, "finish_reason", "stop")
        usage = data.get("usage", {}) if isinstance(data, dict) else response.usage
        blocks = []
        if content_text:
            blocks.append(TextBlock(content_text))
        for call in tool_calls or []:
            function = call.get("function", {}) if isinstance(call, dict) else call.function
            arguments = function.get("arguments", "{}") if isinstance(function, dict) else function.arguments
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
            blocks.append(ToolUseBlock(call.get("id", f"tool_{uuid.uuid4().hex}") if isinstance(call, dict) else getattr(call, "id", f"tool_{uuid.uuid4().hex}"), function.get("name", "") if isinstance(function, dict) else function.name, arguments))
        if not blocks:
            blocks.append(TextBlock(""))
        return CompletionResponse(
            data.get("id", f"msg_{uuid.uuid4().hex}") if isinstance(data, dict) else getattr(response, "id", f"msg_{uuid.uuid4().hex}"),
            request.model, tuple(blocks), {"length": "max_tokens", "tool_calls": "tool_use"}.get(finish, "end_turn"),
            normalize_usage(usage),
        )

    async def stream(self, request: CompletionRequest):
        usage = TokenUsage(0, 0)
        stop_reason = "end_turn"
        finish_seen = False
        slots = set()
        try:
            upstream = await self._client.acompletion(**self.build_request(request, stream=True))
            yield StreamStart()
            iterator = aiter(upstream)
            try:
                async for chunk in iterator:
                    data = chunk if isinstance(chunk, dict) else chunk.model_dump()
                    raw_usage = data.get("usage")
                    if raw_usage is not None:
                        usage = normalize_usage(raw_usage)
                    for choice in data.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield TextDelta(delta["content"])
                        for index, call in enumerate(delta.get("tool_calls") or []):
                            slot = str(call.get("index", index))
                            function = call.get("function") or {}
                            if slot not in slots:
                                slots.add(slot)
                                yield ToolUseStart(slot, call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}", function.get("name", ""))
                            if function.get("arguments"):
                                yield ToolInputDelta(slot, function["arguments"])
                        finish = choice.get("finish_reason")
                        if finish:
                            finish_seen = True
                            stop_reason = {"length": "max_tokens", "tool_calls": "tool_use"}.get(finish, "end_turn")
            finally:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()
            if not finish_seen:
                yield protocol_error(
                    "LiteLLM stream ended without finish_reason", provider="litellm"
                )
                return
            for slot in sorted(slots):
                yield ToolUseEnd(slot)
            yield StreamComplete(stop_reason, usage)
        except Exception as error:
            provider_error = self._provider_error(error)
            yield stream_error_from_exception(
                provider_error, provider="litellm", expose_message=True
            )

    async def count_tokens(self, request: CompletionRequest) -> int:
        payload = self.build_request(request, stream=False)
        counter = getattr(self._client, "token_counter", None)
        if counter is None:
            return 1000
        arguments = {"model": payload["model"], "messages": payload["messages"]}
        if request.model.startswith("openai/") and self._settings.openai_base_url:
            arguments["api_base"] = self._settings.openai_base_url
        return await asyncio.to_thread(counter, **arguments)

    @staticmethod
    def _provider_error(error):
        return ProviderError(str(getattr(error, "message", error)), provider="litellm", status_code=getattr(error, "status_code", 500) or 500)
