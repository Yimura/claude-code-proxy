"""Codex subscription HTTP provider."""

import uuid

import httpx

from ...domain.models import CompletionRequest, StreamError, StreamStart
from ..base import ProviderError
from .auth import CodexAuth
from .translation import CodexEventTranslator, build_request, response_from_events

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_USER_AGENT = "opencode/latest/2.0.3/cli"


class CodexProvider:
    name = "codex"

    def __init__(self, auth: CodexAuth, client_factory=httpx.AsyncClient, token_counter=None) -> None:
        self._auth = auth
        self._client_factory = client_factory
        self._token_counter = token_counter
        self._session_id = uuid.uuid4().hex

    async def complete(self, request: CompletionRequest):
        events = [event async for event in self.stream(request)]
        error = next((event for event in events if isinstance(event, StreamError)), None)
        if error:
            raise ProviderError(error.message, provider=self.name)
        return response_from_events(request, events)

    async def stream(self, request: CompletionRequest):
        try:
            access_token, account_id = self._auth.get_auth()
            headers = self._build_headers(access_token, account_id)
            translator = CodexEventTranslator()
            yield StreamStart()
            async with self._client_factory(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                async with client.stream("POST", CODEX_RESPONSES_URL, headers=headers, json=build_request(request)) as response:
                    if response.status_code != 200:
                        body = "".join([chunk async for chunk in response.aiter_text()])
                        raise ProviderError(f"Codex API error {response.status_code}: {body[:500]}", provider=self.name, status_code=response.status_code)
                    async for event_type, data in self._response_events(response):
                        for event in translator.feed(event_type, data):
                            yield event
            yield translator.finish()
        except ProviderError as error:
            yield StreamError(str(error))
        except Exception as error:
            yield StreamError(str(error))

    async def count_tokens(self, request: CompletionRequest) -> int:
        if self._token_counter is None:
            return 1000
        return await self._token_counter(request)

    async def _response_events(self, response):
        event_type = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                event_type = line[7:].strip()
                continue
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                import json
                data = json.loads(raw)
            except ValueError:
                continue
            yield event_type or "", data

    def _build_headers(self, access_token: str, account_id: str):
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": CODEX_USER_AGENT,
            "originator": "opencode",
            "x-codex-beta-features": "remote_compaction_v2",
            "chatgpt-account-id": account_id,
            "session-id": self._session_id,
        }
