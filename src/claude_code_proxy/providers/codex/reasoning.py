"""Opaque Codex reasoning state carried through Anthropic messages."""

import base64
import binascii
import json
from typing import Any

PREFIX = "codex-reasoning-v1:"


def encode_reasoning(
    encrypted_content: str, summary: list[dict[str, Any]]
) -> str:
    payload = json.dumps(
        {
            "encrypted_content": encrypted_content,
            "summary": summary,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{PREFIX}{encoded}"


def decode_reasoning(data: str) -> dict[str, Any] | None:
    if not data.startswith(PREFIX):
        return None
    encoded = data.removeprefix(PREFIX)
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = json.loads(
            base64.b64decode(
                encoded + padding, altchars=b"-_", validate=True
            ).decode("utf-8")
        )
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    encrypted_content = payload.get("encrypted_content")
    summary = payload.get("summary")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None
    if not isinstance(summary, list) or not all(
        isinstance(item, dict) for item in summary
    ):
        return None
    return {
        "type": "reasoning",
        "summary": summary,
        "encrypted_content": encrypted_content,
    }
