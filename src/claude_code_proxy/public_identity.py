"""Process-scoped opaque public identity derivation."""

import hashlib
import hmac
import secrets


class PublicIdentity:
    """Derive stable opaque identifiers without retaining source identifiers."""

    __slots__ = ("__secret",)

    def __init__(self, *, secret: bytes | None = None) -> None:
        if secret is not None and not isinstance(secret, bytes):
            raise TypeError("identity secret must be bytes")
        self.__secret = secrets.token_bytes(32) if secret is None else secret

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def public_id(self, identifier: str) -> str:
        normalized = identifier.strip()
        return hmac.new(
            self.__secret,
            normalized.encode(errors="surrogatepass"),
            hashlib.sha256,
        ).hexdigest()

    def public_agent_id(self, session_id: str, agent_id: str) -> str:
        normalized_session = session_id.strip()
        normalized_agent = agent_id.strip()
        return self.public_id(
            f"agent:{normalized_session}\0{normalized_agent}"
        )
