"""
Codex subscription provider.

Translates Anthropic API format to OpenAI Responses API and calls
chatgpt.com/backend-api/codex/responses directly, bypassing LiteLLM.
"""

import json
import uuid
import time
import logging
import sqlite3
import os
from typing import Dict, Any, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_URL = "https://auth.openai.com/oauth/token"
CODEX_USER_AGENT = "opencode/latest/2.0.3/cli"
SESSION_ID = uuid.uuid4().hex

# ---------- auth ----------

_token_cache: Dict[str, Any] = {
    "access": None,
    "expires": 0,
    "account_id": None,
}


def _read_credential(data_dir: str) -> Dict[str, Any]:
    db_path = os.path.join(data_dir, "opencode.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM credential "
                "WHERE integration_id = 'openai' AND active = 1 LIMIT 1"
            ).fetchone()
            if row:
                cred = json.loads(row[0])
                meta = cred.get("metadata") or {}
                return {
                    "access": cred.get("access", ""),
                    "refresh": cred.get("refresh", ""),
                    "expires": cred.get("expires", 0),
                    "account_id": meta.get("accountID", ""),
                    "source": "db",
                }
        finally:
            conn.close()

    json_path = os.path.join(data_dir, "auth.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        oa = data.get("openai", {})
        return {
            "access": oa.get("access", ""),
            "refresh": oa.get("refresh", ""),
            "expires": oa.get("expires", 0),
            "account_id": oa.get("accountId", ""),
            "source": "json",
        }

    raise RuntimeError(
        f"No opencode auth found in {data_dir}. Run 'opencode auth login'."
    )


def _refresh_token(refresh: str) -> tuple:
    resp = httpx.post(
        CODEX_AUTH_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": CODEX_CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Codex token refresh failed: {resp.status_code} {resp.text}")
    body = resp.json()
    access = body["access_token"]
    expires_ms = int(time.time() * 1000) + body.get("expires_in", 864000) * 1000
    new_refresh = body.get("refresh_token")
    return access, expires_ms, new_refresh


def _save_refreshed(access: str, expires: int, new_refresh: Optional[str], cred: dict, data_dir: str):
    if cred["source"] == "db":
        db_path = os.path.join(data_dir, "opencode.db")
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT id, value FROM credential "
                "WHERE integration_id = 'openai' AND active = 1 LIMIT 1"
            ).fetchone()
            if row:
                val = json.loads(row[1])
                val["access"] = access
                val["expires"] = expires
                if new_refresh:
                    val["refresh"] = new_refresh
                conn.execute(
                    "UPDATE credential SET value = ?, time_updated = ? WHERE id = ?",
                    (json.dumps(val), int(time.time() * 1000), row[0]),
                )
                conn.commit()
        finally:
            conn.close()
    else:
        path = os.path.join(data_dir, "auth.json")
        with open(path) as f:
            data = json.load(f)
        data["openai"]["access"] = access
        data["openai"]["expires"] = expires
        if new_refresh:
            data["openai"]["refresh"] = new_refresh
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def get_auth(data_dir: str) -> tuple:
    global _token_cache
    now = int(time.time() * 1000)

    if _token_cache["access"] and _token_cache["expires"] > now + 60_000:
        return _token_cache["access"], _token_cache["account_id"]

    cred = _read_credential(data_dir)

    if cred["expires"] <= now + 60_000:
        if not cred["refresh"]:
            raise RuntimeError("Codex token expired, no refresh token. Run 'opencode auth login'.")
        logger.info("Refreshing codex token...")
        access, expires, new_refresh = _refresh_token(cred["refresh"])
        _save_refreshed(access, expires, new_refresh, cred, data_dir)
        cred["access"] = access
        cred["expires"] = expires
        logger.info("Codex token refreshed")

    _token_cache = {
        "access": cred["access"],
        "expires": cred["expires"],
        "account_id": cred["account_id"],
    }
    return cred["access"], cred["account_id"]


# ---------- request conversion (anthropic → responses api) ----------

def _extract_system(system) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n\n".join(parts)
    return str(system)


def _convert_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _convert_messages(messages) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for msg in messages:
        content = msg.content

        if isinstance(content, str):
            items.append({"role": msg.role, "content": content})
            continue

        if not isinstance(content, list):
            items.append({"role": msg.role, "content": str(content)})
            continue

        text_parts = []
        for block in content:
            block_type = block.type if hasattr(block, "type") else block.get("type", "")

            if block_type == "text":
                text_parts.append(block.text if hasattr(block, "text") else block.get("text", ""))

            elif block_type == "tool_use":
                if text_parts:
                    items.append({"role": msg.role, "content": "\n".join(text_parts)})
                    text_parts = []
                bid = block.id if hasattr(block, "id") else block.get("id", "")
                name = block.name if hasattr(block, "name") else block.get("name", "")
                inp = block.input if hasattr(block, "input") else block.get("input", {})
                items.append({
                    "type": "function_call",
                    "call_id": bid,
                    "name": name,
                    "arguments": json.dumps(inp) if isinstance(inp, dict) else str(inp),
                })

            elif block_type == "tool_result":
                if text_parts:
                    items.append({"role": msg.role, "content": "\n".join(text_parts)})
                    text_parts = []
                tool_use_id = (
                    block.tool_use_id if hasattr(block, "tool_use_id")
                    else block.get("tool_use_id", "")
                )
                result_content = block.content if hasattr(block, "content") else block.get("content", "")
                if not isinstance(result_content, str):
                    from server import parse_tool_result_content
                    result_content = parse_tool_result_content(result_content)
                items.append({
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": result_content,
                })

        if text_parts:
            items.append({"role": msg.role, "content": "\n".join(text_parts)})

    return items


def _convert_tools(tools) -> List[Dict[str, Any]]:
    if not tools:
        return []
    result = []
    for tool in tools:
        t = tool.dict() if hasattr(tool, "dict") else dict(tool)
        if t.get("input_schema") is None:
            continue
        result.append({
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["input_schema"],
        })
    return result


def build_request(request) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": request.model.removeprefix("openai/"),
        "input": _convert_messages(request.messages),
        "store": False,
        "stream": True,
    }

    instructions = _extract_system(request.system)
    if instructions:
        body["instructions"] = instructions

    tools = _convert_tools(request.tools)
    if tools:
        body["tools"] = tools

    # Codex endpoint does not support temperature

    if request.tool_choice:
        tc = request.tool_choice if isinstance(request.tool_choice, dict) else request.tool_choice.dict()
        choice_type = tc.get("type", "auto")
        if choice_type == "auto":
            body["tool_choice"] = "auto"
        elif choice_type == "any":
            body["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in tc:
            tool_names = {t["name"] for t in tools}
            if tc["name"] in tool_names:
                body["tool_choice"] = {"type": "function", "name": tc["name"]}
            else:
                body["tool_choice"] = "auto"

    return body


def _build_headers(access_token: str, account_id: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": CODEX_USER_AGENT,
        "originator": "opencode",
        "x-codex-beta-features": "remote_compaction_v2",
        "chatgpt-account-id": account_id,
        "session-id": SESSION_ID,
    }


# ---------- response conversion (responses api sse → anthropic sse) ----------

def _make_message_start(request) -> str:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    data = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    return f"event: message_start\ndata: {json.dumps(data)}\n\n"


async def stream_completion(request, data_dir: str):
    access_token, account_id = get_auth(data_dir)
    headers = _build_headers(access_token, account_id)
    body = build_request(request)

    yield _make_message_start(request)
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    anthropic_index = 0
    text_block_open = True
    tool_blocks: Dict[int, int] = {}
    input_tokens = 0
    output_tokens = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        async with client.stream("POST", CODEX_RESPONSES_URL, headers=headers, json=body) as resp:
            if resp.status_code != 200:
                error_body = ""
                async for chunk in resp.aiter_text():
                    error_body += chunk
                error_msg = f"Codex API error {resp.status_code}: {error_body[:500]}"
                logger.error(error_msg)
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': error_msg}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            event_type = None
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue

                if not line.startswith("data: "):
                    continue

                raw = line[6:]
                if raw == "[DONE]":
                    break

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event_type == "response.output_text.delta":
                    delta_text = data.get("delta", "")
                    if delta_text and text_block_open:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

                elif event_type == "response.content_part.delta":
                    delta_obj = data.get("delta", {})
                    if isinstance(delta_obj, dict):
                        delta_text = delta_obj.get("text", "")
                    else:
                        delta_text = str(delta_obj)
                    if delta_text and text_block_open:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        if text_block_open:
                            text_block_open = False
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        anthropic_index += 1
                        out_idx = data.get("output_index", anthropic_index)
                        tool_blocks[out_idx] = anthropic_index

                        tool_id = item.get("call_id") or item.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                        name = item.get("name", "")
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"

                elif event_type == "response.function_call_arguments.delta":
                    out_idx = data.get("output_index", 0)
                    a_idx = tool_blocks.get(out_idx, anthropic_index)
                    delta = data.get("delta", "")
                    if delta:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': a_idx, 'delta': {'type': 'input_json_delta', 'partial_json': delta}})}\n\n"

                elif event_type == "response.function_call_arguments.done":
                    out_idx = data.get("output_index", 0)
                    a_idx = tool_blocks.get(out_idx, anthropic_index)
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': a_idx})}\n\n"

                elif event_type == "response.output_item.done":
                    item = data.get("item", {})
                    if item.get("type") == "message" and text_block_open:
                        text_block_open = False
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                elif event_type == "response.completed":
                    response_obj = data if "usage" in data else data.get("response", data)
                    usage = response_obj.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                    status = response_obj.get("status", "completed")
                    stop_reason = "end_turn"
                    if status == "incomplete":
                        incomplete = response_obj.get("incomplete_details", {})
                        if incomplete.get("reason") == "max_output_tokens":
                            stop_reason = "max_tokens"
                    if tool_blocks:
                        stop_reason = "tool_use"

    if text_block_open:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    yield "data: [DONE]\n\n"


async def complete(request, data_dir: str):
    """Non-streaming: buffer the SSE stream and return a MessagesResponse."""
    from server import MessagesResponse, Usage

    content_blocks: List[Dict[str, Any]] = []
    current_text = ""
    current_tools: Dict[int, Dict[str, Any]] = {}
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"

    access_token, account_id = get_auth(data_dir)
    headers = _build_headers(access_token, account_id)
    body = build_request(request)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        async with client.stream("POST", CODEX_RESPONSES_URL, headers=headers, json=body) as resp:
            if resp.status_code != 200:
                error_body = ""
                async for chunk in resp.aiter_text():
                    error_body += chunk
                raise RuntimeError(f"Codex API error {resp.status_code}: {error_body[:500]}")

            event_type = None
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event_type in ("response.output_text.delta", "response.content_part.delta"):
                    if event_type == "response.content_part.delta":
                        delta_obj = data.get("delta", {})
                        current_text += delta_obj.get("text", "") if isinstance(delta_obj, dict) else str(delta_obj)
                    else:
                        current_text += data.get("delta", "")

                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        out_idx = data.get("output_index", len(current_tools))
                        current_tools[out_idx] = {
                            "type": "tool_use",
                            "id": item.get("call_id") or item.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": item.get("name", ""),
                            "input": {},
                            "_args": "",
                        }

                elif event_type == "response.function_call_arguments.delta":
                    out_idx = data.get("output_index", 0)
                    if out_idx in current_tools:
                        current_tools[out_idx]["_args"] += data.get("delta", "")

                elif event_type == "response.function_call_arguments.done":
                    out_idx = data.get("output_index", 0)
                    if out_idx in current_tools:
                        args_str = data.get("arguments", current_tools[out_idx].get("_args", "{}"))
                        try:
                            current_tools[out_idx]["input"] = json.loads(args_str)
                        except json.JSONDecodeError:
                            current_tools[out_idx]["input"] = {"raw": args_str}

                elif event_type == "response.completed":
                    response_obj = data if "usage" in data else data.get("response", data)
                    usage = response_obj.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    status = response_obj.get("status", "completed")
                    if status == "incomplete":
                        incomplete = response_obj.get("incomplete_details", {})
                        if incomplete.get("reason") == "max_output_tokens":
                            stop_reason = "max_tokens"

    if current_text:
        content_blocks.append({"type": "text", "text": current_text})

    for _idx in sorted(current_tools.keys()):
        tool = current_tools[_idx]
        content_blocks.append({
            "type": "tool_use",
            "id": tool["id"],
            "name": tool["name"],
            "input": tool["input"],
        })

    if current_tools and stop_reason == "end_turn":
        stop_reason = "tool_use"

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=request.model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
