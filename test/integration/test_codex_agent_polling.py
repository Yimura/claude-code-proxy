"""Opt-in live evaluation for Codex Agent orchestration behavior."""

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
AGENT_TOOL = {
    "name": "Agent",
    "description": "Launch a background worker.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    },
}
TASK_OUTPUT_TOOL = {
    "name": "TaskOutput",
    "description": "Retrieve explicit output from a background task.",
    "input_schema": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}
SEND_MESSAGE_TOOL = {
    "name": "SendMessage",
    "description": "Send follow-up work to an existing worker.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["to", "message"],
    },
}
RECORD_INDEPENDENT_WORK_TOOL = {
    "name": "RecordIndependentWork",
    "description": "Record completion of independent parent work.",
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}
RECORD_DECISION_TOOL = {
    "name": "RecordDecision",
    "description": "Record completed direct analysis.",
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}
SEARCH_REPOSITORY_TOOL = {
    "name": "SearchRepository",
    "description": "Search local repository evidence.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
SEARCH_WEB_TOOL = {
    "name": "SearchWeb",
    "description": (
        "Search external documentation only when local evidence cannot answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
TOOLS = [
    AGENT_TOOL,
    TASK_OUTPUT_TOOL,
    SEND_MESSAGE_TOOL,
    RECORD_INDEPENDENT_WORK_TOOL,
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
                assert not seen_agent, _failure(
                    "Agent launched more than once", transcript
                )
                seen_agent = True
                content = (
                    "Agent launched as agent-eval-1 and remains running. Its "
                    "completion will arrive automatically; continue independent work."
                )
            elif call["name"] == "RecordIndependentWork":
                seen_independent_work = True
                content = "Independent module B work recorded."
            else:
                pytest.fail(_failure(f"unexpected tool {call['name']}", transcript))
            results.append(_tool_result(call, content))
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


def test_related_follow_up_reuses_existing_worker():
    messages = [{
        "role": "user",
        "content": (
            "Use Agent delegation where useful to implement parser validation, "
            "its focused tests, and resulting fixes. These changes share one "
            "subsystem and one file set."
        ),
    }]
    first = _send(messages, tools=[AGENT_TOOL])
    first_calls = _tool_calls(first)
    agent_calls = [call for call in first_calls if call["name"] == "Agent"]
    transcript = [_summarize(first)]

    assert len(agent_calls) == 1, _failure(
        "related work was not grouped under one owner", transcript
    )

    messages.extend([
        {"role": "assistant", "content": first["content"]},
        {
            "role": "user",
            "content": [
                _tool_result(
                    agent_calls[0],
                    "Worker agent-eval-1 completed the coherent parser batch.",
                )
            ],
        },
        {
            "role": "user",
            "content": (
                "Review found one related parser edge case. Address it using "
                "the existing worker, which can be resumed through SendMessage."
            ),
        },
    ])

    follow_up = _send(messages, tools=[AGENT_TOOL, SEND_MESSAGE_TOOL])
    transcript.append(_summarize(follow_up))
    follow_up_calls = _tool_calls(follow_up)
    assert any(call["name"] == "SendMessage" for call in follow_up_calls), _failure(
        "follow-up did not reuse existing worker", transcript
    )
    assert all(call["name"] != "Agent" for call in follow_up_calls), _failure(
        "follow-up launched a replacement worker", transcript
    )


def test_subagent_works_directly_without_nested_agent():
    response = _send(
        [{
            "role": "user",
            "content": (
                "Inspect the assigned parser behavior directly and record your "
                "decision. No recursive delegation was authorized."
            ),
        }],
        tools=[AGENT_TOOL, RECORD_DECISION_TOOL],
        agent_id="agent-eval-child",
        parent_agent_id="agent-eval-parent",
    )
    calls = _tool_calls(response)
    transcript = [_summarize(response)]

    assert all(call["name"] != "Agent" for call in calls), _failure(
        "subagent recursively delegated without authorization", transcript
    )
    assert any(call["name"] == "RecordDecision" for call in calls), _failure(
        "subagent did not perform assigned work directly", transcript
    )


def test_discovery_stops_when_local_evidence_is_sufficient():
    tools = [SEARCH_REPOSITORY_TOOL, SEARCH_WEB_TOOL, RECORD_DECISION_TOOL]
    messages = [{
        "role": "user",
        "content": (
            "Determine where parser timeout is configured and what test proves "
            "a change. Use available evidence, then record the decision."
        ),
    }]
    transcript = []
    discovery_turns = 0
    repository_calls = 0
    web_calls = 0
    evidence_delivered = False
    decision_seen = False

    for _ in range(8):
        response = _send(messages, tools=tools)
        transcript.append(_summarize(response))
        calls = _tool_calls(response)
        messages.append({"role": "assistant", "content": response["content"]})
        decision_calls = [
            call for call in calls if call["name"] == "RecordDecision"
        ]
        if decision_calls:
            assert evidence_delivered, _failure(
                "model decided before reading available evidence", transcript
            )
            decision_seen = True
            break
        if not calls:
            pytest.fail(_failure("model stopped without recording a decision", transcript))

        discovery_turns += 1
        results = []
        for call in calls:
            if call["name"] == "SearchRepository":
                repository_calls += 1
                evidence_delivered = True
                content = (
                    "Current behavior: timeout is 30 seconds. Change location: "
                    "src/parser.py:42. Constraint: preserve cancellation. "
                    "Verification: test_parser_timeout in test_parser.py."
                )
            elif call["name"] == "SearchWeb":
                web_calls += 1
                content = (
                    "External search is unnecessary; repository owns this setting."
                )
            else:
                pytest.fail(_failure(f"unexpected tool {call['name']}", transcript))
            results.append(_tool_result(call, content))
        messages.append({"role": "user", "content": results})

    assert decision_seen, _failure("model never recorded decision", transcript)
    assert repository_calls >= 1, _failure(
        "model did not read required local evidence", transcript
    )
    assert discovery_turns < 8, _failure(
        "model exhausted discovery checkpoint without acting", transcript
    )
    assert web_calls <= 1, _failure(
        "model performed unnecessary web sweep", transcript
    )


def _send(messages, *, tools=TOOLS, agent_id=None, parent_agent_id=None):
    headers = {
        "content-type": "application/json",
        "x-claude-code-session-id": "codex-agent-orchestration-live-eval",
    }
    if agent_id is not None:
        headers["x-claude-code-agent-id"] = agent_id
    if parent_agent_id is not None:
        headers["x-claude-code-parent-agent-id"] = parent_agent_id

    response = httpx.post(
        f"{BASE_URL.rstrip('/')}/v1/messages",
        headers=headers,
        json={
            "model": MODEL,
            "max_tokens": 2_048,
            "messages": messages,
            "tools": tools,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


def _tool_calls(response):
    return [block for block in response["content"] if block["type"] == "tool_use"]


def _tool_result(call, content):
    return {
        "type": "tool_result",
        "tool_use_id": call["id"],
        "content": content,
    }


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
