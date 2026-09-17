import base64
import json

import pytest

from claude_code_proxy.providers.codex.reasoning import (
    decode_reasoning,
    encode_reasoning,
)


def encoded_payload(payload) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return "codex-reasoning-v1:" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def test_reasoning_envelope_round_trips_deterministically():
    summary = [{"type": "summary_text", "text": "checked"}]

    encoded = encode_reasoning("encrypted-state", summary)

    assert encoded.startswith("codex-reasoning-v1:")
    assert decode_reasoning(encoded) == {
        "type": "reasoning",
        "summary": summary,
        "encrypted_content": "encrypted-state",
    }
    assert encode_reasoning("encrypted-state", summary) == encoded


@pytest.mark.parametrize(
    "data",
    [
        "anthropic-ciphertext",
        "codex-reasoning-v2:abc",
        "codex-reasoning-v1:not-base64!",
        "codex-reasoning-v1:e30",
        encoded_payload([]),
        encoded_payload({"encrypted_content": "", "summary": []}),
        encoded_payload({"encrypted_content": 1, "summary": []}),
        encoded_payload({"encrypted_content": "encrypted", "summary": {}}),
        encoded_payload({"encrypted_content": "encrypted", "summary": ["text"]}),
    ],
)
def test_invalid_or_foreign_envelope_is_ignored(data):
    assert decode_reasoning(data) is None


def test_invalid_utf8_envelope_is_ignored():
    data = "codex-reasoning-v1:" + base64.urlsafe_b64encode(b"\xff").decode().rstrip("=")

    assert decode_reasoning(data) is None
