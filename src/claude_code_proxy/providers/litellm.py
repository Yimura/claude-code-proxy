"""LiteLLM provider adapter."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import os
import uuid
from typing import Any

# LiteLLM loads model metadata during import; prefer its bundled map by default.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from ..config import Settings
from ..domain.models import (
    CompletionRequest, CompletionResponse, ImageBlock, StreamComplete, StreamError,
    StreamStart, TextBlock, TextDelta, TokenUsage, ToolInputDelta, ToolResultBlock,
    ToolUseBlock, ToolUseEnd, ToolUseStart,
)
from ..performance import (
    ProviderTelemetry,
    ReasoningContinuation,
    notify_telemetry,
)
from ..failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
    unexpected_failure_diagnostic,
)
from .base import (
    ProviderError,
    protocol_error,
    public_error,
    scalar_provider_code,
    stream_error_from_exception,
)
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


@dataclass
class _LiteLLMStreamState:
    usage: TokenUsage = field(default_factory=lambda: TokenUsage.unavailable())
    stop_reason: str = "end_turn"
    finish_seen: bool = False
    slots: set[str] = field(default_factory=set)
    terminal_error: StreamError | None = None

    def feed(self, chunk) -> tuple:
        data = chunk if isinstance(chunk, dict) else chunk.model_dump()
        raw_usage = data.get("usage")
        if raw_usage is not None:
            self.usage = normalize_usage(raw_usage)

        events = []
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                events.append(TextDelta(delta["content"]))
            for index, call in enumerate(delta.get("tool_calls") or []):
                slot = str(call.get("index", index))
                function = call.get("function") or {}
                if slot not in self.slots:
                    self.slots.add(slot)
                    events.append(
                        ToolUseStart(
                            slot,
                            call.get("id")
                            or f"toolu_{uuid.uuid4().hex[:24]}",
                            function.get("name", ""),
                        )
                    )
                if function.get("arguments"):
                    events.append(ToolInputDelta(slot, function["arguments"]))
            finish = choice.get("finish_reason")
            if finish:
                self.finish_seen = True
                self.stop_reason = {
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }.get(finish, "end_turn")
        return tuple(events)

    def completion_events(self) -> tuple:
        events = [ToolUseEnd(slot) for slot in sorted(self.slots)]
        events.append(StreamComplete(self.stop_reason, self.usage))
        return tuple(events)


def _report_reasoning_continuation(
    request: CompletionRequest, telemetry: ProviderTelemetry | None
) -> None:
    state: ReasoningContinuation = (
        "unavailable" if request.reasoning.enabled else "not_applicable"
    )
    notify_telemetry(telemetry, "set_reasoning_continuation", state)


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

    async def complete(
        self,
        request: CompletionRequest,
        telemetry: ProviderTelemetry | None = None,
    ) -> CompletionResponse:
        _report_reasoning_continuation(request, telemetry)
        try:
            payload = self.build_request(request, stream=False)
        except Exception as error:
            raise self._translation_error(
                error, "request_translation_failed"
            ) from error

        try:
            response = await asyncio.to_thread(self._client.completion, **payload)
        except ProviderError:
            raise
        except Exception as error:
            raise self._provider_error(
                error,
                stage=FailureStage.REQUEST,
                code="provider_invocation_failed",
            ) from error

        try:
            return self._normalize_response(response, request)
        except Exception as error:
            raise self._translation_error(
                error, "response_translation_failed"
            ) from error

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
            request.response_model, tuple(blocks), {"length": "max_tokens", "tool_calls": "tool_use"}.get(finish, "end_turn"),
            normalize_usage(usage),
        )

    async def stream(self, request: CompletionRequest, telemetry: ProviderTelemetry | None = None):
        _report_reasoning_continuation(request, telemetry)
        try:
            payload = self.build_request(request, stream=True)
        except Exception as error:
            error = self._translation_error(error, "request_translation_failed")
            yield stream_error_from_exception(error, provider=self.name)
            return

        try:
            upstream = await self._client.acompletion(**payload)
        except ProviderError as error:
            yield stream_error_from_exception(error, provider=self.name)
            return
        except Exception as error:
            error = self._provider_error(
                error,
                stage=FailureStage.REQUEST,
                code="provider_invocation_failed",
            )
            yield stream_error_from_exception(error, provider=self.name)
            return

        try:
            iterator = aiter(upstream)
        except Exception as error:
            error = self._translation_error(
                error, "stream_initialization_failed"
            )
            yield stream_error_from_exception(error, provider=self.name)
            return

        state = _LiteLLMStreamState()
        external_exit = False
        try:
            yield StreamStart()
            try:
                async for chunk in iterator:
                    try:
                        events = state.feed(chunk)
                    except Exception as error:
                        error = self._translation_error(
                            error, "stream_chunk_translation_failed"
                        )
                        state.terminal_error = stream_error_from_exception(
                            error, provider=self.name
                        )
                        break
                    for event in events:
                        yield event
            except ProviderError as error:
                state.terminal_error = stream_error_from_exception(
                    error, provider=self.name
                )
            except Exception as error:
                error = self._provider_error(
                    error,
                    stage=FailureStage.STREAM,
                    code="stream_failed",
                )
                state.terminal_error = stream_error_from_exception(
                    error, provider=self.name
                )
        except (GeneratorExit, asyncio.CancelledError):
            external_exit = True
            raise
        finally:
            await self._close_stream_iterator(iterator, state, external_exit)

        if state.terminal_error is not None:
            yield state.terminal_error
            return
        if not state.finish_seen:
            yield protocol_error("missing_finish_reason", provider=self.name)
            return
        for event in state.completion_events():
            yield event

    async def _close_stream_iterator(
        self,
        iterator,
        state: _LiteLLMStreamState,
        external_exit: bool,
    ) -> None:
        try:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except Exception as error:
            if external_exit or state.terminal_error is not None:
                return
            error = self._provider_error(
                error,
                stage=FailureStage.STREAM,
                code="stream_cleanup_failed",
            )
            state.terminal_error = stream_error_from_exception(
                error, provider=self.name
            )

    async def count_tokens(
        self,
        request: CompletionRequest,
        telemetry: ProviderTelemetry | None = None,
    ) -> int:
        try:
            payload = self.build_request(request, stream=False)
        except Exception as error:
            raise self._translation_error(
                error, "token_count_request_translation_failed"
            ) from error

        counter = getattr(self._client, "token_counter", None)
        if counter is None:
            return 1000
        arguments = {"model": payload["model"], "messages": payload["messages"]}
        if request.model.startswith("openai/") and self._settings.openai_base_url:
            arguments["api_base"] = self._settings.openai_base_url
        try:
            return await asyncio.to_thread(counter, **arguments)
        except ProviderError:
            raise
        except Exception as error:
            raise self._provider_error(
                error,
                stage=FailureStage.REQUEST,
                code="token_count_failed",
            ) from error

    @staticmethod
    def _translation_error(error: Exception, code: str) -> ProviderError:
        return ProviderError(
            "Internal server error",
            provider="litellm",
            status_code=500,
            diagnostic=unexpected_failure_diagnostic(
                error,
                category=FailureCategory.TRANSLATION,
                stage=FailureStage.PROVIDER_TRANSLATION,
                code=code,
            ),
        )

    @staticmethod
    def _provider_error(
        error: Exception,
        *,
        stage: FailureStage,
        code: str,
    ) -> ProviderError:
        status_code = getattr(error, "status_code", None)
        if not isinstance(status_code, int) or isinstance(status_code, bool):
            status_code = None

        category = FailureCategory.INTERNAL
        local_code = code
        if isinstance(error, litellm.AuthenticationError):
            category = FailureCategory.AUTHENTICATION
            status_code = status_code or 401
            local_code = "authentication_error"
        elif isinstance(error, litellm.PermissionDeniedError):
            category = FailureCategory.AUTHENTICATION
            status_code = status_code or 403
            local_code = "permission_denied"
        elif isinstance(error, litellm.Timeout):
            category = FailureCategory.TRANSPORT
            status_code = 504
            local_code = "timeout"
        elif isinstance(error, litellm.APIConnectionError):
            category = FailureCategory.TRANSPORT
            status_code = 503
            local_code = "transport_error"
        elif isinstance(error, litellm.APIError) or status_code is not None:
            category = FailureCategory.UPSTREAM_HTTP
            status_code = status_code or 502
            local_code = "upstream_http_error"
        else:
            status_code = 500

        _, message = public_error(status_code)
        if category == FailureCategory.INTERNAL:
            diagnostic = unexpected_failure_diagnostic(
                error,
                stage=stage,
                code=local_code,
            )
        else:
            diagnostic = FailureDiagnostic(
                category,
                stage,
                local_code,
                scalar_provider_code(getattr(error, "code", None)),
            )
        return ProviderError(
            message,
            provider="litellm",
            status_code=status_code,
            diagnostic=diagnostic,
        )
