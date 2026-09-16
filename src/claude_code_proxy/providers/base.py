"""Shared provider contract and errors."""

from collections.abc import AsyncIterator
from typing import Protocol
from ..domain.models import CompletionRequest, CompletionResponse, StreamEvent


class ProviderError(Exception):
    def __init__(self, message: str, *, provider: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class UnsupportedOperationError(ProviderError):
    pass


class Provider(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
    async def count_tokens(self, request: CompletionRequest) -> int: ...
