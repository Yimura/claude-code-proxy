"""Anthropic-compatible external request and response schemas."""

from typing import Any, Literal
from pydantic import BaseModel, model_serializer
from ..reasoning import OutputConfig, ThinkingConfig


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Any


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[
        ContentBlockText
        | ContentBlockImage
        | ContentBlockRedactedThinking
        | ContentBlockToolUse
        | ContentBlockToolResult
    ]


class Tool(BaseModel):
    name: str
    type: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    model_config = {"extra": "allow"}


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    output_config: OutputConfig | None = None


class TokenCountRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class OutputTokensDetails(BaseModel):
    thinking_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens_details: OutputTokensDetails | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        if self.output_tokens_details is None:
            data.pop("output_tokens_details", None)
        return data


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[
        ContentBlockText | ContentBlockRedactedThinking | ContentBlockToolUse
    ]
    type: Literal["message"] = "message"
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = None
    stop_sequence: str | None = None
    usage: Usage
