"""Map Claude client identity to Codex session and thread identity."""

from dataclasses import dataclass
import json
import uuid

from ...domain.models import ClientIdentity

_THREAD_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/Yimura/claude-code-proxy/codex-thread",
)


def _thread_id(session_id: str, agent_id: str | None) -> str:
    identity = ["root"] if agent_id is None else ["agent", agent_id.strip()]
    name = json.dumps(
        [session_id.strip(), identity],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(_THREAD_NAMESPACE, name))


@dataclass(frozen=True)
class CodexIdentity:
    session_id: str
    thread_id: str
    parent_thread_id: str | None

    @classmethod
    def from_client(cls, identity: ClientIdentity) -> "CodexIdentity":
        session_id = identity.session_id or str(uuid.uuid4())
        agent_id = identity.agent_id
        parent_agent_id = identity.parent_agent_id if agent_id else None
        return cls(
            session_id=session_id,
            thread_id=_thread_id(session_id, agent_id),
            parent_thread_id=(
                _thread_id(session_id, parent_agent_id)
                if parent_agent_id is not None
                else None
            ),
        )

    def turn_metadata(self) -> dict[str, str]:
        metadata = {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
        }
        if self.parent_thread_id is not None:
            metadata["parent_thread_id"] = self.parent_thread_id
        return metadata

    def client_metadata(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "x-codex-turn-metadata": json.dumps(
                self.turn_metadata(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
