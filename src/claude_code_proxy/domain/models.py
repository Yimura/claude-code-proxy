"""Immutable provider-neutral request, response, and stream models."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ..reasoning import OutputConfig, ReasoningPolicy, ThinkingConfig


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ImageBlock:
    source: dict[str, Any]


@dataclass(frozen=True)
class RedactedThinkingBlock:
    data: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: Any


ContentBlock: TypeAlias = (
    TextBlock | ImageBlock | RedactedThinkingBlock | ToolUseBlock | ToolResultBlock
)
ResponseBlock: TypeAlias = TextBlock | RedactedThinkingBlock | ToolUseBlock


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolChoice:
    type: Literal["auto", "any", "tool"] = "auto"
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


@dataclass(frozen=True)
class ClientIdentity:
    session_id: str | None = field(default=None, repr=False)
    agent_id: str | None = field(default=None, repr=False)
    parent_agent_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class CompletionRequest:
    original_model: str
    model: str
    response_model: str
    max_tokens: int
    messages: tuple[Message, ...]
    reasoning: ReasoningPolicy
    context_window: int | None = None
    client_identity: ClientIdentity = field(default_factory=ClientIdentity)
    system: tuple[TextBlock, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    thinking: ThinkingConfig | None = None
    output_config: OutputConfig | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    thinking_tokens: int | None = None


@dataclass(frozen=True)
class CompletionResponse:
    id: str
    model: str
    content: tuple[ResponseBlock, ...]
    stop_reason: str | None
    usage: TokenUsage


@dataclass(frozen=True)
class StreamStart:
    input_tokens: int = 0


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class RedactedThinking:
    data: str


@dataclass(frozen=True)
class ToolUseStart:
    slot: str
    id: str
    name: str


@dataclass(frozen=True)
class ToolInputDelta:
    slot: str
    partial_json: str


@dataclass(frozen=True)
class ToolUseEnd:
    slot: str


@dataclass(frozen=True)
class StreamComplete:
    stop_reason: str
    usage: TokenUsage


@dataclass(frozen=True)
class StreamError:
    error_type: str = "api_error"
    message: str = "Internal server error"
    status_code: int | None = None
    retryable: bool = True
    provider: str | None = None
    diagnostic: str | None = None


StreamEvent: TypeAlias = (
    StreamStart
    | TextDelta
    | RedactedThinking
    | ToolUseStart
    | ToolInputDelta
    | ToolUseEnd
    | StreamComplete
    | StreamError
)
