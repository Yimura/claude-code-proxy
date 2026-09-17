"""Reconcile recognized Claude Code identity metadata after model mapping."""

import re

from .domain.models import Message, TextBlock

CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_CODE_HARNESS = (
    "You are running inside Claude Code, Anthropic's coding-agent CLI harness."
)
MODEL_IDENTITY = re.compile(
    r"You are powered by the model named [^\n]+?\. "
    r"The exact model ID is (?P<model_id>[^\s.]+)\."
)


def reconcile_system_identity(
    system: tuple[TextBlock, ...],
    original_model: str,
    resolved_model: str,
    mapped: bool,
) -> tuple[TextBlock, ...]:
    """Correct recognized model metadata while preserving other system text."""
    if not mapped or not system:
        return system
    if not _contains_original_identity(system, original_model):
        return system

    model_identity = _resolved_identity(resolved_model)
    reconciled = tuple(
        _reconcile_block(block, original_model, model_identity) for block in system
    )
    return system if reconciled == system else reconciled


def reconcile_message_identities(
    messages: tuple[Message, ...],
    original_model: str,
    resolved_model: str,
    mapped: bool,
) -> tuple[Message, ...]:
    """Correct recognized model metadata in system-role messages."""
    if not mapped or not messages:
        return messages
    text_blocks = tuple(
        block
        for message in messages
        if message.role == "system"
        for block in message.content
        if isinstance(block, TextBlock)
    )
    if not _contains_original_identity(text_blocks, original_model):
        return messages

    model_identity = _resolved_identity(resolved_model)
    reconciled = tuple(
        _reconcile_message(message, original_model, model_identity)
        for message in messages
    )
    return messages if reconciled == messages else reconciled


def _contains_original_identity(
    system: tuple[TextBlock, ...], original_model: str
) -> bool:
    return any(
        match.group("model_id") == original_model
        for block in system
        for match in MODEL_IDENTITY.finditer(block.text)
    )


def _resolved_identity(resolved_model: str) -> str:
    qualifier = ""
    if not resolved_model.startswith("anthropic/"):
        qualifier = ", not an Anthropic Claude model"
    return f"The model generating this response is {resolved_model}{qualifier}."


def _reconcile_block(
    block: TextBlock, original_model: str, model_identity: str
) -> TextBlock:
    text = block.text.replace(CLAUDE_CODE_IDENTITY, CLAUDE_CODE_HARNESS)

    def replace_identity(match: re.Match[str]) -> str:
        if match.group("model_id") != original_model:
            return match.group(0)
        return model_identity

    text = MODEL_IDENTITY.sub(replace_identity, text)
    return block if text == block.text else TextBlock(text)


def _reconcile_message(
    message: Message, original_model: str, model_identity: str
) -> Message:
    if message.role != "system":
        return message
    content = tuple(
        _reconcile_block(block, original_model, model_identity)
        if isinstance(block, TextBlock)
        else block
        for block in message.content
    )
    return message if content == message.content else Message(message.role, content)
