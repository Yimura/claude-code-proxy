import pytest

from claude_code_proxy.public_identity import PublicIdentity


def test_public_identity_is_stable_scoped_and_opaque() -> None:
    identity = PublicIdentity(secret=b"test-secret")

    session = identity.public_id("  raw-session  ")
    agent = identity.public_agent_id("raw-session", " raw-agent ")

    assert session == identity.public_id("raw-session")
    assert agent == identity.public_agent_id(" raw-session ", "raw-agent")
    assert session != agent
    assert "raw-session" not in session
    assert "raw-agent" not in agent


def test_public_identity_repr_does_not_expose_secret() -> None:
    identity = PublicIdentity(secret=b"highly-sensitive-secret")

    rendered = repr(identity)

    assert "highly-sensitive-secret" not in rendered
    assert "secret" not in rendered.lower()


def test_public_identity_requires_bytes_secret() -> None:
    with pytest.raises(TypeError, match="bytes"):
        PublicIdentity(secret="not-bytes")  # type: ignore[arg-type]
