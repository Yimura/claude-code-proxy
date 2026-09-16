"""Translation between external Anthropic schemas and domain models."""

from contextlib import suppress
from copy import deepcopy
from typing import Any
from ..domain.models import CompletionRequest, ImageBlock, Message, TextBlock, ToolChoice, ToolDefinition, ToolResultBlock, ToolUseBlock
from ..reasoning import ReasoningPolicy
from .schemas import ContentBlockImage, ContentBlockText, ContentBlockToolResult, ContentBlockToolUse, MessagesRequest


def normalize_request(request: MessagesRequest) -> CompletionRequest:
    return CompletionRequest(
        original_model=request.model,
        model=request.model,
        max_tokens=request.max_tokens,
        messages=tuple(Message(role=message.role, content=_normalize_content(message.content)) for message in request.messages),
        reasoning=ReasoningPolicy(None, None),
        system=_normalize_system(request.system),
        tools=tuple(ToolDefinition(name=tool.name, description=tool.description or "", input_schema=deepcopy(tool.input_schema)) for tool in request.tools or []),
        tool_choice=_normalize_tool_choice(request.tool_choice),
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=tuple(request.stop_sequences or ()),
        metadata=deepcopy(request.metadata),
        thinking=request.thinking,
        output_config=request.output_config,
    )


def _normalize_content(content: str | list[Any]):
    if isinstance(content, str):
        return (TextBlock(content),)
    blocks = []
    for block in content:
        if isinstance(block, ContentBlockText):
            blocks.append(TextBlock(block.text))
        elif isinstance(block, ContentBlockImage):
            blocks.append(ImageBlock(deepcopy(block.source)))
        elif isinstance(block, ContentBlockToolUse):
            blocks.append(ToolUseBlock(block.id, block.name, deepcopy(block.input)))
        elif isinstance(block, ContentBlockToolResult):
            blocks.append(ToolResultBlock(block.tool_use_id, deepcopy(block.content)))
        else:
            raise TypeError(f"Unsupported content block: {type(block).__name__}")
    return tuple(blocks)


def _normalize_system(system):
    if system is None:
        return ()
    if isinstance(system, str):
        return (TextBlock(system),)
    return tuple(TextBlock(block.text) for block in system)


def _normalize_tool_choice(choice: dict[str, Any] | None) -> ToolChoice | None:
    if choice is None:
        return None
    return ToolChoice(type=choice.get("type", "auto"), name=choice.get("name"), disable_parallel_tool_use=choice.get("disable_parallel_tool_use"))


import asyncio
import json
import uuid

from ..domain.models import (
    CompletionResponse,
    StreamComplete,
    StreamError,
    StreamStart,
    TextDelta,
    TokenUsage,
    ToolInputDelta,
    ToolUseEnd,
    ToolUseStart,
)
from .schemas import (
    ContentBlockText as ApiTextBlock,
    ContentBlockToolUse as ApiToolUseBlock,
    MessagesResponse,
    Usage,
)


def to_api_response(response: CompletionResponse) -> MessagesResponse:
    content = []
    for block in response.content:
        if isinstance(block, TextBlock):
            content.append(ApiTextBlock(type="text", text=block.text))
        elif isinstance(block, ToolUseBlock):
            content.append(
                ApiToolUseBlock(
                    type="tool_use",
                    id=block.id,
                    name=block.name,
                    input=block.input,
                )
            )
    return MessagesResponse(
        id=response.id,
        model=response.model,
        content=content,
        stop_reason=response.stop_reason,
        usage=_api_usage(response.usage),
    )


async def serialize_stream(
    request: CompletionRequest, events, *, heartbeat_interval: float = 15.0
):
    state = _AnthropicStreamState(request)
    for frame in state.start():
        yield frame

    iterator = aiter(events)
    pending_event = None
    try:
        pending_event = asyncio.ensure_future(anext(iterator))
        while True:
            done, _ = await asyncio.wait(
                {pending_event}, timeout=heartbeat_interval
            )
            if not done:
                yield _sse("ping", {"type": "ping"})
                continue
            completed_event = pending_event
            pending_event = None
            try:
                event = completed_event.result()
            except StopAsyncIteration:
                break
            try:
                frames = state.consume(event)
            except ValueError as error:
                frames = state.error(StreamError(diagnostic=str(error)))
                for frame in frames:
                    yield frame
                return
            for frame in frames:
                yield frame
            if isinstance(event, (StreamComplete, StreamError)):
                return
            pending_event = asyncio.ensure_future(anext(iterator))

        for frame in state.error(
            StreamError(diagnostic="stream ended without terminal outcome")
        ):
            yield frame
    finally:
        try:
            if pending_event is not None:
                if not pending_event.done():
                    pending_event.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending_event
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()


class _AnthropicStreamState:
    def __init__(self, request: CompletionRequest) -> None:
        self.request = request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.text_open = True
        self.tool_indices: dict[str, int] = {}
        self.open_tools: set[str] = set()
        self.next_index = 1

    def start(self) -> list[str]:
        message = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        }
        return [
            _sse("message_start", {"type": "message_start", "message": message}),
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse("ping", {"type": "ping"}),
        ]

    def consume(self, event) -> list[str]:
        if isinstance(event, (StreamStart,)):
            return []
        if isinstance(event, TextDelta):
            return self._text_delta(event)
        if isinstance(event, ToolUseStart):
            return self._tool_start(event)
        if isinstance(event, ToolInputDelta):
            return self._tool_delta(event)
        if isinstance(event, ToolUseEnd):
            return self._tool_end(event)
        if isinstance(event, StreamComplete):
            return self.finish(event)
        if isinstance(event, StreamError):
            return self.error(event)
        raise TypeError(f"Unsupported stream event: {type(event).__name__}")

    def _text_delta(self, event: TextDelta) -> list[str]:
        if not self.text_open:
            raise ValueError("text delta received after text block closed")
        return [
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": event.text},
                },
            )
        ]

    def _tool_start(self, event: ToolUseStart) -> list[str]:
        if event.slot in self.tool_indices:
            raise ValueError(f"duplicate tool start for slot {event.slot}")
        frames = self._close_text()
        index = self.next_index
        self.next_index += 1
        self.tool_indices[event.slot] = index
        self.open_tools.add(event.slot)
        frames.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": event.id,
                        "name": event.name,
                        "input": {},
                    },
                },
            )
        )
        return frames

    def _tool_delta(self, event: ToolInputDelta) -> list[str]:
        if event.slot not in self.open_tools:
            raise ValueError(f"tool delta received for closed slot {event.slot}")
        index = self.tool_indices[event.slot]
        return [
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": event.partial_json,
                    },
                },
            )
        ]

    def _tool_end(self, event: ToolUseEnd) -> list[str]:
        if event.slot not in self.open_tools:
            raise ValueError(f"tool end received for closed slot {event.slot}")
        self.open_tools.remove(event.slot)
        return [self._block_stop(self.tool_indices[event.slot])]

    def finish(self, event: StreamComplete) -> list[str]:
        if self.open_tools:
            raise ValueError("stream completed with open tool blocks")
        frames = self._close_text()
        frames.extend(
            [
                _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": event.stop_reason,
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": event.usage.output_tokens},
                    },
                ),
                _sse("message_stop", {"type": "message_stop"}),
                "data: [DONE]\n\n",
            ]
        )
        return frames

    def error(self, event: StreamError) -> list[str]:
        return [
            _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": event.error_type,
                        "message": event.message,
                    },
                },
            )
        ]

    def _close_text(self) -> list[str]:
        if not self.text_open:
            return []
        self.text_open = False
        return [self._block_stop(0)]

    @staticmethod
    def _block_stop(index: int) -> str:
        return _sse(
            "content_block_stop", {"type": "content_block_stop", "index": index}
        )


def _api_usage(usage: TokenUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
