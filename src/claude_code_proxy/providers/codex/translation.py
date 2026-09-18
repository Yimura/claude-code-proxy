"""Translation between normalized domain models and Codex Responses API."""

from dataclasses import dataclass, field
import json
import uuid
from typing import Any

from ...domain.models import (
    CompletionRequest,
    CompletionResponse,
    RedactedThinking,
    RedactedThinkingBlock,
    StreamComplete,
    StreamError,
    StreamEvent,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolInputDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseEnd,
    ToolUseStart,
)
from ..usage import normalize_usage
from .identity import CodexIdentity
from .reasoning import decode_reasoning, encode_reasoning


def content_to_text(content: Any) -> str:
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


def build_request(
    request: CompletionRequest,
    identity: CodexIdentity | None = None,
) -> dict[str, Any]:
    resolved_identity = identity or CodexIdentity.from_client(
        request.client_identity
    )
    body: dict[str, Any] = {
        "model": request.model.removeprefix("openai/"),
        "input": _convert_messages(request),
        "store": False,
        "stream": True,
        "prompt_cache_key": resolved_identity.session_id,
        "client_metadata": resolved_identity.client_metadata(),
    }
    if request.reasoning.enabled and request.reasoning.effort:
        body["reasoning"] = {"effort": request.reasoning.effort}
    if request.reasoning.enabled:
        body["include"] = ["reasoning.encrypted_content"]
    if request.system:
        body["instructions"] = "\n\n".join(block.text for block in request.system)
    tools = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "strict": False,
            "parameters": tool.input_schema,
        }
        for tool in request.tools
        if tool.input_schema is not None
    ]
    if tools:
        body["tools"] = tools
    if request.tool_choice is not None:
        body["tool_choice"] = _convert_tool_choice(request.tool_choice, tools)
    return body


def _convert_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    items = []
    for message in request.messages:
        text_parts = []
        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
                continue
            if text_parts:
                items.append({"role": message.role, "content": "\n".join(text_parts)})
                text_parts = []
            if isinstance(block, RedactedThinkingBlock):
                reasoning = decode_reasoning(block.data)
                if reasoning is not None:
                    items.append(reasoning)
            elif isinstance(block, ToolUseBlock):
                items.append({"type": "function_call", "call_id": block.id, "name": block.name, "arguments": json.dumps(block.input)})
            elif isinstance(block, ToolResultBlock):
                items.append({"type": "function_call_output", "call_id": block.tool_use_id, "output": content_to_text(block.content)})
        if text_parts:
            items.append({"role": message.role, "content": "\n".join(text_parts)})
    return items


def _convert_tool_choice(choice, tools):
    if choice.type == "any":
        return "required"
    if choice.type == "tool" and choice.name in {tool["name"] for tool in tools}:
        return {"type": "function", "name": choice.name}
    return "auto"


@dataclass
class CodexEventTranslator:
    tool_slots: dict[int, str] = field(default_factory=dict)
    tool_arguments: dict[int, str] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    stop_reason: str = "end_turn"
    completed: bool = False

    def feed(self, event_type: str, data: dict[str, Any]) -> tuple[StreamEvent, ...]:
        if event_type == "response.output_text.delta":
            text = data.get("delta", "")
            return (TextDelta(text),) if text else ()
        if event_type == "response.content_part.delta":
            delta = data.get("delta", {})
            text = delta.get("text", "") if isinstance(delta, dict) else str(delta)
            return (TextDelta(text),) if text else ()
        if event_type == "response.output_item.added":
            return self._tool_start(data)
        if event_type == "response.output_item.done":
            return self._completed_item(data)
        if event_type == "response.function_call_arguments.delta":
            return self._tool_delta(data)
        if event_type == "response.function_call_arguments.done":
            return self._tool_end(data)
        if event_type in {"response.completed", "response.incomplete"}:
            self._record_completion(data)
            return ()
        if event_type == "response.failed":
            response = data.get("response", data)
            error = response.get("error") or {}
            message = error.get("message") or "Codex request failed"
            code = error.get("code") or "unknown_error"
            return (
                StreamError(
                    message=message,
                    provider="codex",
                    diagnostic=f"{code}: {message}",
                ),
            )
        return ()

    def finish(self) -> StreamComplete:
        return StreamComplete(self.stop_reason, self.usage)

    def _completed_item(self, data):
        item = data.get("item", {})
        if item.get("type") != "reasoning":
            return ()
        encrypted_content = item.get("encrypted_content")
        summary = item.get("summary", [])
        if not isinstance(encrypted_content, str) or not encrypted_content:
            return ()
        if not isinstance(summary, list) or not all(
            isinstance(part, dict) for part in summary
        ):
            return ()
        return (
            RedactedThinking(encode_reasoning(encrypted_content, summary)),
        )

    def _tool_start(self, data):
        item = data.get("item", {})
        if item.get("type") != "function_call":
            return ()
        output_index = data.get("output_index", len(self.tool_slots))
        slot = str(output_index)
        self.tool_slots[output_index] = slot
        self.tool_arguments[output_index] = ""
        tool_id = item.get("call_id") or item.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
        return (ToolUseStart(slot, tool_id, item.get("name", "")),)

    def _tool_delta(self, data):
        output_index = data.get("output_index", 0)
        slot = self.tool_slots.get(output_index, str(output_index))
        delta = data.get("delta", "")
        if not delta:
            return ()
        self.tool_arguments[output_index] = self.tool_arguments.get(output_index, "") + delta
        return (ToolInputDelta(slot, delta),)

    def _tool_end(self, data):
        output_index = data.get("output_index", 0)
        slot = self.tool_slots.get(output_index, str(output_index))
        arguments = data.get("arguments", "")
        events = []
        if arguments and not self.tool_arguments.get(output_index):
            self.tool_arguments[output_index] = arguments
            events.append(ToolInputDelta(slot, arguments))
        events.append(ToolUseEnd(slot))
        return tuple(events)

    def _record_completion(self, data):
        self.completed = True
        response = data if "usage" in data else data.get("response", data)
        self.usage = normalize_usage(response.get("usage"))
        if response.get("status") == "incomplete":
            details = response.get("incomplete_details", {})
            if details.get("reason") == "max_output_tokens":
                self.stop_reason = "max_tokens"
        if self.tool_slots and self.stop_reason == "end_turn":
            self.stop_reason = "tool_use"


def response_from_events(request: CompletionRequest, events: list[StreamEvent]) -> CompletionResponse:
    text_segments: list[list[str]] = []
    tools: dict[str, dict[str, Any]] = {}
    order: list[tuple[str, int | str | RedactedThinkingBlock]] = []
    complete = StreamComplete("end_turn", TokenUsage(0, 0))
    for event in events:
        if isinstance(event, TextDelta):
            if not order or order[-1][0] != "text":
                text_segments.append([])
                order.append(("text", len(text_segments) - 1))
            text_segments[int(order[-1][1])].append(event.text)
        elif isinstance(event, RedactedThinking):
            order.append(
                ("redacted_thinking", RedactedThinkingBlock(event.data))
            )
        elif isinstance(event, ToolUseStart):
            tools[event.slot] = {
                "id": event.id,
                "name": event.name,
                "arguments": "",
            }
            order.append(("tool", event.slot))
        elif isinstance(event, ToolInputDelta):
            tools.setdefault(
                event.slot, {"id": "", "name": "", "arguments": ""}
            )["arguments"] += event.partial_json
        elif isinstance(event, StreamComplete):
            complete = event
    blocks = []
    for kind, value in order:
        if kind == "text":
            blocks.append(TextBlock("".join(text_segments[int(value)])))
        elif kind == "redacted_thinking":
            blocks.append(value)
        elif kind == "tool":
            tool = tools[str(value)]
            try:
                arguments = json.loads(tool["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"raw": tool["arguments"]}
            blocks.append(
                ToolUseBlock(tool["id"], tool["name"], arguments)
            )
    if not blocks:
        blocks.append(TextBlock(""))
    stop_reason = (
        "tool_use"
        if tools and complete.stop_reason == "end_turn"
        else complete.stop_reason
    )
    return CompletionResponse(
        f"msg_{uuid.uuid4().hex[:24]}",
        request.response_model,
        tuple(blocks),
        stop_reason,
        complete.usage,
    )
