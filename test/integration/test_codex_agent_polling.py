"""Opt-in live evaluation for Codex Agent completion behavior."""

import json
import os

import httpx
import pytest

RUN_LIVE_EVAL = os.environ.get("RUN_CODEX_AGENT_EVAL") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_LIVE_EVAL,
    reason="set RUN_CODEX_AGENT_EVAL=1 to run billable live evaluation",
)

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8082")
MODEL = os.environ.get("CODEX_AGENT_EVAL_MODEL", "claude-opus-5")
TOOLS = [
    {
        "name": "Agent",
        "description": "Launch a background worker.",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
    {
        "name": "TaskOutput",
        "description": "Retrieve explicit output from a background task.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "RecordIndependentWork",
        "description": "Record completion of independent parent work.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


def test_agent_completion_is_not_polled():
    messages = [{
        "role": "user",
        "content": (
            "Launch one Agent to analyze module A. While that worker runs, use "
            "RecordIndependentWork to record that module B was checked. Agent "
            "completion will be delivered later automatically. Do not wait for it."
        ),
    }]
    transcript = []
    seen_agent = False
    seen_independent_work = False

    for _ in range(3):
        response = _send(messages)
        transcript.append(_summarize(response))
        calls = _tool_calls(response)
        _assert_no_task_output(calls, transcript)
        messages.append({"role": "assistant", "content": response["content"]})
        if not calls:
            break

        results = []
        for call in calls:
            if call["name"] == "Agent":
                assert not seen_agent, _failure("Agent launched more than once", transcript)
                seen_agent = True
                content = (
                    "Agent launched as agent-eval-1 and remains running. Its completion "
                    "will arrive automatically; continue independent work."
                )
            elif call["name"] == "RecordIndependentWork":
                seen_independent_work = True
                content = "Independent module B work recorded."
            else:
                pytest.fail(_failure(f"unexpected tool {call['name']}", transcript))
            results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": content,
            })
        messages.append({"role": "user", "content": results})
        if seen_agent and seen_independent_work:
            break

    assert seen_agent, _failure("Agent was not launched", transcript)
    assert seen_independent_work, _failure(
        "parent did not continue independent work", transcript
    )

    notification = (
        "<task-notification>Agent agent-eval-1 completed: module A is clean."
        "</task-notification>"
    )
    messages.append({"role": "user", "content": notification})
    final = _send(messages)
    transcript.append(_summarize(final))
    final_calls = _tool_calls(final)
    _assert_no_task_output(final_calls, transcript)
    assert not final_calls, _failure(
        "parent launched tools after pushed completion", transcript
    )
    assert sum(
        message.get("content") == notification for message in messages
    ) == 1


def _send(messages):
    response = httpx.post(
        f"{BASE_URL.rstrip('/')}/v1/messages",
        headers={
            "content-type": "application/json",
            "x-claude-code-session-id": "codex-agent-polling-live-eval",
        },
        json={
            "model": MODEL,
            "max_tokens": 2_048,
            "messages": messages,
            "tools": TOOLS,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


def _tool_calls(response):
    return [block for block in response["content"] if block["type"] == "tool_use"]


def _assert_no_task_output(calls, transcript):
    assert all(call["name"] != "TaskOutput" for call in calls), _failure(
        "parent polled Agent through TaskOutput", transcript
    )


def _summarize(response):
    return {
        "id": response.get("id"),
        "model": response.get("model"),
        "stop_reason": response.get("stop_reason"),
        "blocks": [
            {
                "type": block.get("type"),
                "name": block.get("name"),
                "text": block.get("text", "")[:200],
            }
            for block in response.get("content", [])
            if block.get("type") != "redacted_thinking"
        ],
    }


def _failure(message, transcript):
    return f"{message}\ntranscript={json.dumps(transcript, indent=2)}"
