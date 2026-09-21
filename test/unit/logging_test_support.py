from claude_code_proxy.logging import (
    RequestLogContext,
    SessionIdentity,
)


def make_context(identity: SessionIdentity | None = None) -> RequestLogContext:
    return RequestLogContext(
        session=identity
        or SessionIdentity("abcdef123456", "[session abcdef123456]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude-sonnet",
        upstream_model="openai/gpt-5.6-sol",
        provider="fake",
        effort="high",
    )
