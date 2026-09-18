import uuid

from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.providers.codex.identity import CodexIdentity


def test_root_and_child_share_session_but_use_distinct_threads():
    root = CodexIdentity.from_client(ClientIdentity("session"))
    child = CodexIdentity.from_client(ClientIdentity("session", "agent"))

    assert root.session_id == child.session_id == "session"
    assert root.thread_id != child.thread_id
    assert uuid.UUID(root.thread_id)
    assert uuid.UUID(child.thread_id)


def test_resumed_agent_reuses_thread_across_instances():
    identity = ClientIdentity("session", "agent", "parent")

    first = CodexIdentity.from_client(identity)
    second = CodexIdentity.from_client(identity)

    assert first == second
    assert first.parent_thread_id is not None
    assert first.parent_thread_id != first.thread_id


def test_equal_agent_names_are_scoped_to_root_session():
    first = CodexIdentity.from_client(ClientIdentity("first", "worker"))
    second = CodexIdentity.from_client(ClientIdentity("second", "worker"))

    assert first.thread_id != second.thread_id


def test_root_sentinel_cannot_collide_with_agent_name():
    root = CodexIdentity.from_client(ClientIdentity("session"))
    named_root = CodexIdentity.from_client(ClientIdentity("session", "root"))

    assert root.thread_id != named_root.thread_id


def test_missing_session_generates_distinct_request_scoped_identity():
    first = CodexIdentity.from_client(ClientIdentity(None, "agent"))
    second = CodexIdentity.from_client(ClientIdentity(None, "agent"))

    assert first.session_id != second.session_id
    assert first.thread_id != second.thread_id
