from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import threading

import pytest

from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionResponse,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolInputDelta,
    ToolUseStart,
)
from claude_code_proxy.event_journal import EventJournal
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.observability import (
    AmbiguousSessionId,
    InvalidSessionFilter,
    ObservationHandle,
    SessionMetadata,
    SessionRegistry,
)


class Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 1, tzinfo=UTC)
        self.monotonic = 100.0

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def advance(self, seconds: float = 1.0) -> None:
        self.wall += timedelta(seconds=seconds)
        self.monotonic += seconds


def metadata(
    session_id: str | None = "sensitive-session",
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    **changes,
) -> SessionMetadata:
    values = {
        "client_identity": ClientIdentity(
            session_id,
            agent_id,
            parent_agent_id,
        ),
        "client_model": "claude-opus",
        "upstream_model": "openai/gpt-5.6-sol",
        "provider": "openai",
        "transport": "codex",
        "effort": "high",
        "context_window": 1_000_000,
    }
    values.update(changes)
    return SessionMetadata(**values)


def registry(
    clock: Clock,
    inactive_limit: int = 10,
    events: EventJournal | None = None,
) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
        events=events,
    )


def test_constructor_enforces_signed_64_inactive_limit() -> None:
    maximum = 2**63 - 1

    assert SessionRegistry(maximum).inactive_limit == maximum
    with pytest.raises(ValueError, match="inactive_limit"):
        SessionRegistry(maximum + 1)


def test_constructor_rejects_negative_inactive_limit() -> None:
    with pytest.raises(ValueError, match="inactive_limit"):
        SessionRegistry(-1)


def test_observation_handle_preserves_all_legacy_positional_fields() -> None:
    required = ("key", "request", "public", 1.0, False, False)
    handles = [
        ObservationHandle(*required),
        ObservationHandle(*required, "agent-key"),
        ObservationHandle(*required, "agent-key", "agent-public"),
        ObservationHandle(
            *required, "agent-key", "agent-public", "parent-public"
        ),
        ObservationHandle(
            *required,
            "agent-key",
            "agent-public",
            "parent-public",
            True,
        ),
    ]

    assert handles[0].operation == "messages"
    assert handles[1].agent_key == "agent-key"
    assert handles[2].agent_public_id == "agent-public"
    assert handles[3].parent_agent_public_id == "parent-public"
    assert handles[4].agent_is_new is True
    assert all(handle.operation == "messages" for handle in handles)


def test_observation_handle_accepts_explicit_operation_keyword() -> None:
    handle = ObservationHandle(
        "key",
        "request",
        "public",
        1.0,
        False,
        False,
        operation="count_tokens",
    )

    assert handle.operation == "count_tokens"


def test_agents_share_root_aggregate_but_keep_independent_counts() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata(agent_id="agent-one"))
    second = sessions.begin(metadata(agent_id="agent-two"))

    snapshot = sessions.snapshots()[0]

    assert snapshot.requests == 2
    assert snapshot.active_requests == 2
    assert len(snapshot.agents) == 2
    assert {agent.requests for agent in snapshot.agents} == {1}
    assert {agent.active_requests for agent in snapshot.agents} == {1}
    assert first.agent_public_id != second.agent_public_id


def test_resumed_agent_reuses_public_identity_and_record() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata(agent_id="agent-one"))
    sessions.finish(first, "completed")
    second = sessions.begin(metadata(agent_id="agent-one"))

    snapshot = sessions.snapshots()[0]

    assert first.agent_public_id == second.agent_public_id
    assert len(snapshot.agents) == 1
    assert snapshot.agents[0].requests == 2
    assert snapshot.agents[0].active_requests == 1


def test_agent_public_identity_is_scoped_to_root_session() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata("session-one", "shared-name"))
    second = sessions.begin(metadata("session-two", "shared-name"))

    assert first.agent_public_id != second.agent_public_id


def test_nested_agent_retains_safe_parent_reference() -> None:
    clock = Clock()
    sessions = registry(clock)
    nested = sessions.begin(
        metadata("session", "nested-agent", "parent-agent")
    )

    agent = sessions.snapshots()[0].agents[0]

    assert agent.id == nested.agent_public_id
    assert agent.parent_id == nested.parent_agent_public_id
    assert agent.id != agent.parent_id


def test_main_request_does_not_create_agent_snapshot() -> None:
    clock = Clock()
    sessions = registry(clock)

    sessions.begin(metadata())

    assert sessions.snapshots()[0].agents == ()


def test_finishing_one_agent_keeps_other_agent_and_root_active() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata(agent_id="first"))
    sessions.begin(metadata(agent_id="second"))

    sessions.finish(first, "completed")

    snapshot = sessions.snapshots()[0]
    states = {agent.id: agent.state for agent in snapshot.agents}
    assert snapshot.state == "active"
    assert states[first.agent_public_id] == "idle"
    assert "active" in states.values()


def test_request_scoped_agents_do_not_share_records() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata(None, "agent"))
    second = sessions.begin(metadata(None, "agent"))

    assert len(sessions.snapshots()) == 2
    assert first.agent_public_id != second.agent_public_id

def test_public_id_is_stable_and_raw_id_is_not_exposed() -> None:
    clock = Clock()
    sessions = registry(clock)

    session_metadata = metadata("  sensitive-session  ")
    handle = sessions.begin(session_metadata)
    snapshot = sessions.snapshots()[0]

    assert sessions.inactive_limit == 10
    assert handle.public_id == sessions.public_id("sensitive-session")
    assert handle.public_id == sessions.public_id("  sensitive-session  ")
    assert len(snapshot.id) == 64
    assert snapshot.id == snapshot.id.lower()
    assert snapshot.id != "sensitive-session"
    assert "sensitive-session" not in repr(session_metadata)
    assert "sensitive-session" not in repr(snapshot)
    assert "sensitive-session" not in repr(sessions.snapshots())


def test_logical_session_reuses_row_and_missing_ids_are_request_scoped() -> None:
    clock = Clock()
    sessions = registry(clock)

    first = sessions.begin(metadata(" shared "))
    second = sessions.begin(metadata("shared"))
    missing = sessions.begin(metadata(None))
    blank = sessions.begin(metadata("   "))

    assert first.is_new is True
    assert second.is_new is False
    assert first.key == second.key
    assert first.public_id == second.public_id
    assert missing.is_new is True
    assert blank.is_new is True
    assert missing.request_scoped is True
    assert blank.request_scoped is True
    assert missing.key != blank.key
    assert missing.public_id != blank.public_id
    assert sorted(snapshot.requests for snapshot in sessions.snapshots()) == [1, 1, 2]


def test_overlapping_requests_keep_session_active_and_track_durations() -> None:
    clock = Clock()
    sessions = registry(clock)

    first = sessions.begin(metadata())
    clock.advance(2)
    second = sessions.begin(metadata())
    assert sessions.snapshots()[0].elapsed_seconds == 2

    clock.advance(3)
    sessions.finish(first, "completed")
    active = sessions.snapshots()[0]
    assert active.state == "active"
    assert active.active_requests == 1
    assert active.requests == 2
    assert active.elapsed_seconds == 3
    assert active.last_result == "completed"

    clock.advance(4)
    sessions.finish(second, "failed")
    failed = sessions.snapshots()[0]
    assert failed.state == "failed"
    assert failed.active_requests == 0
    assert failed.requests == 2
    assert failed.elapsed_seconds == 7
    assert failed.last_result == "failed"


def test_completed_session_is_idle_and_negative_duration_is_clamped() -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata())
    clock.monotonic -= 10

    sessions.finish(handle, "completed")

    snapshot = sessions.snapshots()[0]
    assert snapshot.state == "idle"
    assert snapshot.elapsed_seconds == 0


def test_finish_is_idempotent_for_finished_and_unknown_handles() -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata())
    clock.advance(4)
    sessions.finish(handle, "completed")
    finished = sessions.snapshots()[0]

    clock.advance(20)
    sessions.finish(handle, "failed")
    sessions.finish(replace(handle, request_id="unknown"), "failed")

    assert sessions.snapshots()[0] == finished


def test_public_metadata_escapes_unpaired_surrogates_without_changing_unicode() -> None:
    clock = Clock()
    sessions = registry(clock)
    raw = "ordinary-界-\ud800-\udfff"

    handle = sessions.begin(
        SessionMetadata(
            client_identity=ClientIdentity("raw-\ud800-session"),
            client_model=raw,
            upstream_model=f"openai/{raw}",
            provider=raw,
            transport=raw,
            effort=raw,
            context_window=1,
        )
    )

    snapshot = sessions.snapshots()[0]
    expected = "ordinary-界-\\ud800-\\udfff"
    assert snapshot.client_model == expected
    assert snapshot.model == expected
    assert snapshot.provider == expected
    assert snapshot.transport == expected
    assert snapshot.effort == expected
    assert handle.public_id == sessions.public_id("raw-\ud800-session")
    assert "raw-\ud800-session" not in repr(sessions)
    for value in (
        snapshot.client_model,
        snapshot.model,
        snapshot.provider,
        snapshot.transport,
        snapshot.effort,
    ):
        value.encode("utf-8", errors="strict")


def test_later_begin_refreshes_metadata() -> None:
    clock = Clock()
    sessions = registry(clock)
    sessions.begin(metadata())
    clock.advance()

    sessions.begin(
        SessionMetadata(
            client_identity=ClientIdentity("sensitive-session"),
            client_model="claude-sonnet",
            upstream_model="vertex/claude-sonnet-5",
            provider="vertex",
            transport="litellm",
            effort="medium",
            context_window=None,
        )
    )

    snapshot = sessions.snapshots()[0]
    assert snapshot.client_model == "claude-sonnet"
    assert snapshot.model == "claude-sonnet-5"
    assert snapshot.provider == "vertex"
    assert snapshot.transport == "litellm"
    assert snapshot.effort == "medium"
    assert snapshot.context_window is None


def test_snapshots_sort_by_last_seen_descending() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata("first"))
    clock.advance()
    second = sessions.begin(metadata("second"))
    clock.advance()
    sessions.begin(metadata("first"))

    assert [item.id for item in sessions.snapshots()] == [
        first.public_id,
        second.public_id,
    ]


def test_retention_evicts_oldest_inactive_and_never_active() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=1)
    active = sessions.begin(metadata("active"))

    clock.advance()
    old_inactive = sessions.begin(metadata("old-inactive"))
    sessions.finish(old_inactive, "completed")
    clock.advance()
    new_inactive = sessions.begin(metadata("new-inactive"))
    sessions.finish(new_inactive, "completed")

    retained_ids = {item.id for item in sessions.snapshots()}
    assert retained_ids == {active.public_id, new_inactive.public_id}

    clock.advance()
    sessions.finish(active, "completed")

    assert [item.id for item in sessions.snapshots()] == [active.public_id]


def test_zero_retention_removes_row_when_it_becomes_inactive() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=0)
    handle = sessions.begin(metadata())

    assert sessions.counts() == (1, 1)
    sessions.finish(handle, "completed")
    assert sessions.snapshots() == []
    assert sessions.counts() == (0, 0)


def make_filter_registry(clock: Clock) -> tuple[SessionRegistry, dict[str, str]]:
    sessions = registry(clock)
    active = sessions.begin(metadata("active-row"))

    clock.advance()
    idle = sessions.begin(
        SessionMetadata(
            client_identity=ClientIdentity("idle-row"),
            client_model="Claude-Sonnet",
            upstream_model="vertex/Claude-Sonnet-5",
            provider="Vertex",
            transport="LiteLLM",
            effort="Medium",
            context_window=None,
        )
    )
    sessions.finish(idle, "completed")

    clock.advance()
    failed = sessions.begin(
        SessionMetadata(
            client_identity=ClientIdentity("failed-row"),
            client_model="claude-haiku",
            upstream_model="gemini/gemini-pro",
            provider="Gemini",
            transport="LiteLLM",
            effort="Low",
            context_window=200_000,
        )
    )
    sessions.finish(failed, "failed")
    return sessions, {
        "active": active.public_id,
        "idle": idle.public_id,
        "failed": failed.public_id,
    }


@pytest.mark.parametrize(
    ("filters", "expected_name"),
    [
        ({"state": ["ACTIVE"]}, "active"),
        ({"provider": ["VERTEX"]}, "idle"),
        ({"transport": ["CODEX"]}, "active"),
        ({"model": ["CLAUDE-SONNET-5"]}, "idle"),
        ({"effort": ["low"]}, "failed"),
    ],
)
def test_filters_exact_fields_case_insensitively(filters, expected_name) -> None:
    clock = Clock()
    sessions, ids = make_filter_registry(clock)

    assert [item.id for item in sessions.snapshots(filters)] == [ids[expected_name]]


def test_id_filter_uses_case_insensitive_public_id_prefix() -> None:
    clock = Clock()
    sessions, ids = make_filter_registry(clock)
    prefix = ids["idle"][:12].upper()

    assert [item.id for item in sessions.snapshots({"id": [prefix]})] == [
        ids["idle"]
    ]


def test_session_id_filter_hashes_raw_id_for_exact_lookup() -> None:
    clock = Clock()
    sessions, ids = make_filter_registry(clock)

    assert [
        item.id
        for item in sessions.snapshots({"session_id": ["  idle-row  "]})
    ] == [ids["idle"]]
    assert sessions.snapshots({"session_id": ["missing-row"]}) == []
    assert sessions.snapshots({"session_id": ["IDLE-ROW"]}) == []


def test_session_id_filter_combines_with_public_filters() -> None:
    clock = Clock()
    sessions, ids = make_filter_registry(clock)

    assert [
        item.id
        for item in sessions.snapshots(
            {
                "session_id": ["idle-row"],
                "effort": ["medium"],
            }
        )
    ] == [ids["idle"]]
    assert sessions.snapshots(
        {
            "session_id": ["idle-row"],
            "effort": ["low"],
        }
    ) == []


def test_filter_values_or_within_key_and_across_keys() -> None:
    clock = Clock()
    sessions, ids = make_filter_registry(clock)

    state_matches = sessions.snapshots({"state": ["idle", "FAILED"]})
    combined = sessions.snapshots({
        "state": ["idle", "failed"],
        "transport": ["litellm"],
        "effort": ["medium"],
    })

    assert {item.id for item in state_matches} == {ids["idle"], ids["failed"]}
    assert [item.id for item in combined] == [ids["idle"]]


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"unknown": ["value"]},
        {"": ["value"]},
        {"   ": ["value"]},
        {"state": []},
        {"state": "active"},
        {"session_id": ["   "]},
    ],
)
def test_invalid_filters_are_rejected(filters) -> None:
    clock = Clock()
    sessions, _ = make_filter_registry(clock)

    with pytest.raises(InvalidSessionFilter):
        sessions.snapshots(filters)


def test_unmatched_id_prefix_contributes_no_rows() -> None:
    clock = Clock()
    sessions, _ = make_filter_registry(clock)

    assert sessions.snapshots({"id": ["not-a-public-id"]}) == []


def test_ambiguous_id_prefix_is_rejected() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=20)
    prefixes: dict[str, str] = {}
    collision = None
    for index in range(17):
        handle = sessions.begin(metadata(f"session-{index}"))
        prefix = handle.public_id[0]
        if prefix in prefixes:
            collision = prefix
            break
        prefixes[prefix] = handle.public_id

    assert collision is not None
    with pytest.raises(AmbiguousSessionId, match=collision):
        sessions.snapshots({"id": [collision.upper()]})


def test_snapshot_strips_only_first_model_prefix() -> None:
    clock = Clock()
    sessions = registry(clock)
    prefixed = sessions.begin(metadata("prefixed"))
    unprefixed = sessions.begin(
        replace(
            metadata("unprefixed"),
            upstream_model="claude-opus-5",
        )
    )
    nested = sessions.begin(
        replace(
            metadata("nested"),
            upstream_model="gateway/openai/gpt-5.6-sol",
        )
    )

    models = {item.id: item.model for item in sessions.snapshots()}
    assert models[prefixed.public_id] == "gpt-5.6-sol"
    assert models[unprefixed.public_id] == "claude-opus-5"
    assert models[nested.public_id] == "openai/gpt-5.6-sol"


def test_counts_return_active_and_total_retained_rows() -> None:
    clock = Clock()
    sessions = registry(clock)
    active = sessions.begin(metadata("active"))
    inactive = sessions.begin(metadata("inactive"))
    sessions.finish(inactive, "completed")

    assert sessions.counts() == (1, 2)
    sessions.finish(active, "completed")
    assert sessions.counts() == (0, 2)


def test_concurrent_begin_reuses_one_logical_row() -> None:
    clock = Clock()
    sessions = registry(clock)

    with ThreadPoolExecutor(max_workers=8) as pool:
        handles = list(pool.map(sessions.begin, [metadata()] * 32))

    snapshot = sessions.snapshots()[0]
    assert sum(handle.is_new for handle in handles) == 1
    assert snapshot.active_requests == 32
    assert snapshot.requests == 32
    assert sessions.counts() == (1, 1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda handle: sessions.finish(handle, "completed"), handles))

    assert sessions.snapshots()[0].state == "idle"
    assert sessions.counts() == (0, 1)


def test_retention_uses_true_lru_when_wall_timestamps_are_equal() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=2)
    first = sessions.begin(metadata("first"))
    sessions.finish(first, "completed")
    second = sessions.begin(metadata("second"))
    sessions.finish(second, "completed")

    refreshed_first = sessions.begin(metadata("first"))
    sessions.finish(refreshed_first, "completed")
    third = sessions.begin(metadata("third"))
    sessions.finish(third, "completed")

    retained = {item.id for item in sessions.snapshots()}
    assert retained == {first.public_id, third.public_id}
    assert second.public_id not in retained


def test_retention_uses_true_lru_when_wall_clock_moves_backward() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=2)
    first = sessions.begin(metadata("first"))
    sessions.finish(first, "completed")

    clock.advance(10)
    second = sessions.begin(metadata("second"))
    sessions.finish(second, "completed")

    clock.wall -= timedelta(seconds=20)
    clock.monotonic += 1
    refreshed_first = sessions.begin(metadata("first"))
    sessions.finish(refreshed_first, "completed")
    clock.monotonic += 1
    third = sessions.begin(metadata("third"))
    sessions.finish(third, "completed")

    retained = {item.id for item in sessions.snapshots()}
    assert retained == {first.public_id, third.public_id}
    assert second.public_id not in retained


class RecordingRegistryLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def __enter__(self) -> "RecordingRegistryLock":
        self._lock.acquire()
        self._owner = threading.get_ident()
        return self

    def __exit__(self, *args: object) -> None:
        self._owner = None
        self._lock.release()

    def owned_by_current_thread(self) -> bool:
        return self._owner == threading.get_ident()


class LockRecordingClock:
    def __init__(self, registry_lock: RecordingRegistryLock) -> None:
        self.registry_lock = registry_lock
        self.samples: list[tuple[str, bool]] = []
        self.wall = datetime(2026, 1, 1, tzinfo=UTC)

    def wall_now(self) -> datetime:
        self.samples.append(("wall", self.registry_lock.owned_by_current_thread()))
        return self.wall

    def monotonic_now(self) -> float:
        self.samples.append(("monotonic", self.registry_lock.owned_by_current_thread()))
        return 100.0


def test_concurrent_begin_samples_clocks_while_holding_registry_lock() -> None:
    recording_lock = RecordingRegistryLock()
    clock = LockRecordingClock(recording_lock)
    sessions = SessionRegistry(
        inactive_limit=10,
        secret=b"test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
    )
    sessions._lock = recording_lock
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def observe(client_model: str) -> None:
        try:
            start.wait(timeout=2)
            sessions.begin(replace(metadata(), client_model=client_model))
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=observe, args=(f"model-{index}",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    start.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    snapshot = sessions.snapshots()[0]
    assert snapshot.requests == 2
    assert snapshot.first_seen <= snapshot.last_seen
    assert {kind for kind, _ in clock.samples} == {"wall", "monotonic"}
    assert all(lock_held for _, lock_held in clock.samples)


def test_last_seen_does_not_decrease_when_wall_clock_moves_backward() -> None:
    clock = Clock()
    sessions = registry(clock)
    sessions.begin(metadata())
    initial = sessions.snapshots()[0]

    clock.wall -= timedelta(seconds=30)
    clock.monotonic += 1
    sessions.begin(metadata())

    updated = sessions.snapshots()[0]
    assert updated.first_seen == initial.first_seen
    assert updated.last_seen == initial.last_seen
    assert updated.first_seen <= updated.last_seen


@pytest.mark.asyncio
async def test_performance_begin_and_finish_publish_safe_shared_snapshots() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)

    handle = sessions.begin(metadata("raw-session-secret"))
    capture = sessions.performance_snapshots()
    replay = events.subscribe(after=0)

    assert capture.cursor == 1
    assert capture.sessions[0].session.id == handle.public_id
    assert capture.sessions[0].performance.session_id == handle.public_id
    assert replay.replay[0].type == "request_started"
    assert replay.replay[0].session_id == handle.public_id
    assert replay.replay[0].request_id == handle.request_id
    assert "raw-session-secret" not in repr(capture)
    assert "raw-session-secret" not in repr(replay.replay)

    terminal = sessions.finish(handle, "completed")
    repeated = sessions.finish(handle, "failed")

    assert terminal is not None
    assert terminal.outcome == "completed"
    assert repeated is None
    assert events.current_sequence == 2
    assert (await replay.receive(1)).type == "completed"
    replay.close()


def test_performance_overlap_keeps_base_and_reducer_concurrency_coherent() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata())
    second = sessions.begin(metadata())

    active = sessions.performance_snapshots().sessions[0]

    assert active.session.active_requests == 2
    assert active.session.state == "active"
    assert active.performance.current_concurrency == 2
    assert active.performance.peak_concurrency == 2
    assert {item.id for item in active.performance.active_requests} == {
        first.request_id,
        second.request_id,
    }

    sessions.finish(first, "completed")
    one_left = sessions.performance_snapshots().sessions[0]
    assert one_left.session.active_requests == 1
    assert one_left.performance.current_concurrency == 1


@pytest.mark.parametrize(
    ("outcome", "base_result", "base_state"),
    [
        ("completed", "completed", "idle"),
        ("failed", "failed", "failed"),
        ("cancelled", "failed", "failed"),
        ("client_disconnected", "failed", "failed"),
    ],
)
def test_detailed_outcomes_map_to_existing_base_contract(
    outcome: str,
    base_result: str,
    base_state: str,
) -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata())

    terminal = sessions.finish(handle, outcome)  # type: ignore[arg-type]
    view = sessions.performance_snapshots().sessions[0]

    assert terminal is not None
    assert terminal.outcome == outcome
    assert view.performance.outcomes == {outcome: 1}
    assert view.session.last_result == base_result
    assert view.session.state == base_state


def test_sequence_exhaustion_keeps_begin_and_finish_state_atomic() -> None:
    clock = Clock()
    begin_events = EventJournal()
    begin_events._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]
    begin_sessions = registry(clock, events=begin_events)

    with pytest.raises(ValueError, match="sequence"):
        begin_sessions.begin(metadata())
    assert begin_sessions.snapshots() == []

    finish_events = EventJournal()
    finish_sessions = registry(clock, events=finish_events)
    handle = finish_sessions.begin(metadata())
    before = finish_sessions.performance_snapshots().sessions
    finish_events._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        finish_sessions.finish(handle, "completed")

    assert finish_sessions.performance_snapshots().sessions == before
    assert finish_sessions.snapshots()[0].state == "active"


@pytest.mark.asyncio
async def test_progress_is_rate_bounded_but_first_output_is_immediate() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    subscription = events.subscribe(after=1)

    observer.stream_event(TextDelta("content-secret"))
    for _ in range(100):
        observer.stream_event(ToolInputDelta("slot-secret", "input-secret"))

    assert (await subscription.receive(1)).type == "first_output"
    assert await subscription.receive(0) is None

    clock.advance(0.25)
    observer.stream_event(ToolInputDelta("slot-secret", "input-secret"))
    progress = await subscription.receive(1)

    assert progress.type == "progress"
    assert "content-secret" not in repr(progress)
    assert "slot-secret" not in repr(progress)
    assert "input-secret" not in repr(progress)
    subscription.close()


@pytest.mark.asyncio
async def test_first_tool_start_publishes_first_output_then_tool_use_safely() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    subscription = events.subscribe(after=1)

    observer.stream_event(
        ToolUseStart("slot-secret", "tool-id-secret", "tool-name-secret")
    )

    published = [
        await subscription.receive(1),
        await subscription.receive(1),
    ]
    assert [item.type for item in published] == ["first_output", "tool_use"]
    rendered = repr(published)
    for secret in ("slot-secret", "tool-id-secret", "tool-name-secret"):
        assert secret not in rendered
    subscription.close()


def test_observer_records_retry_usage_count_tokens_and_reasoning() -> None:
    clock = Clock()
    sessions = registry(clock)
    messages_handle = sessions.begin(metadata())
    messages = sessions.observer(messages_handle)

    messages.upstream_started()
    clock.advance(0.5)
    messages.upstream_finished()
    messages.mark_retries_supported()
    messages.record_retry()
    messages.set_reasoning_continuation("restored")
    messages.response(
        CompletionResponse(
            "response-secret",
            "model-secret",
            (TextBlock("answer-secret"),),
            "end_turn",
            TokenUsage(7, 5),
        )
    )
    sessions.finish(messages_handle, "completed")

    count_handle = sessions.begin(metadata(), operation="count_tokens")
    sessions.observer(count_handle).count_tokens(13)
    sessions.finish(count_handle, "completed")
    snapshot = sessions.performance_snapshots().sessions[0].performance
    count_snapshot, message_snapshot = snapshot.recent_requests

    assert message_snapshot.retries.value == 1
    assert message_snapshot.reasoning_continuation == "restored"
    assert message_snapshot.input_tokens.value == 7
    assert message_snapshot.output_tokens.value == 5
    assert message_snapshot.upstream_duration.value == 0.5
    assert count_snapshot.input_tokens.value == 13
    assert count_snapshot.output_tokens.status == "not_applicable"
    assert "answer-secret" not in repr(snapshot)
    assert "response-secret" not in repr(snapshot)


@pytest.mark.parametrize("value", [True, -1, MAX_CONTROL_INTEGER + 1])
def test_count_tokens_rejects_invalid_values_without_mutation(value: object) -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata(), operation="count_tokens")
    before = sessions.performance_snapshots().sessions

    with pytest.raises(ValueError, match="count_tokens"):
        sessions.observer(handle).count_tokens(value)  # type: ignore[arg-type]

    assert sessions.performance_snapshots().sessions == before


def test_capture_reuses_snapshot_filters_retention_and_latest_twenty() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=1)
    active = sessions.begin(metadata("active"))
    evicted = sessions.begin(metadata("evicted"))
    sessions.finish(evicted, "completed")
    for index in range(25):
        handle = sessions.begin(metadata("history"), operation="count_tokens")
        sessions.observer(handle).count_tokens(index)
        sessions.finish(handle, "completed")

    history = sessions.performance_snapshots({"session_id": ["history"]})
    active_capture = sessions.performance_snapshots({"state": ["ACTIVE"]})

    assert len(history.sessions[0].performance.recent_requests) == 20
    assert history.sessions[0].performance.input_tokens.value == sum(range(25))
    assert active_capture.sessions[0].session.id == active.public_id
    assert sessions.performance_snapshots(
        {"session_id": ["evicted"]}
    ).sessions == ()


@pytest.mark.asyncio
async def test_performance_subscription_capture_replay_reset_and_live_delivery() -> None:
    clock = Clock()
    events = EventJournal(capacity=2)
    sessions = registry(clock, events=events)
    first = sessions.begin(metadata("first"))

    fresh = sessions.subscribe_performance(None, after=None)
    assert fresh.initial is not None
    assert fresh.initial.cursor == 1

    retained = sessions.subscribe_performance(None, after=0)
    assert retained.initial is None
    assert [event.sequence for event in retained.subscription.replay] == [1]

    sessions.finish(first, "completed")
    second = sessions.begin(metadata("second"))
    sessions.finish(second, "completed")
    stale = sessions.subscribe_performance(None, after=1)
    assert stale.subscription.reset_required is True
    assert stale.initial is not None
    assert stale.initial.cursor == 4

    live = sessions.begin(metadata("live"))
    delivered = await fresh.subscription.receive(1)
    assert delivered.request_id == first.request_id
    while delivered.request_id != live.request_id:
        delivered = await fresh.subscription.receive(1)
    assert delivered.type == "request_started"

    for item in (fresh, retained, stale):
        item.subscription.close()


def test_begin_uses_supplied_clocks_and_rejects_invalid_values_atomically() -> None:
    clock = Clock()
    sessions = registry(clock)
    supplied_wall = datetime(2026, 2, 3, tzinfo=UTC)

    handle = sessions.begin(
        metadata("supplied"),
        started_at=supplied_wall,
        started_monotonic=7.5,
    )
    started = sessions.performance_snapshots(
        {"session_id": ["supplied"]}
    ).sessions[0].performance.active_requests[0]

    assert handle.operation == "messages"
    assert handle.started_monotonic == 7.5
    assert started.started_at == supplied_wall
    before = sessions.performance_snapshots().sessions

    with pytest.raises(ValueError, match="started_at"):
        sessions.begin(
            metadata("invalid-wall"),
            started_at=datetime(2026, 2, 3),
        )
    with pytest.raises(ValueError, match="started_monotonic"):
        sessions.begin(
            metadata("invalid-monotonic"),
            started_monotonic=float("nan"),
        )

    assert sessions.performance_snapshots().sessions == before


@pytest.mark.asyncio
async def test_retry_publication_is_immediate_and_safe() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    subscription = events.subscribe(after=1)

    observer.mark_retries_supported()
    assert await subscription.receive(0) is None
    observer.record_retry()
    retry = await subscription.receive(1)

    assert retry.type == "retry"
    assert retry.request.retries.value == 1
    subscription.close()


def test_performance_capture_rejects_ambiguous_public_id_prefix() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=20)
    prefixes: dict[str, str] = {}
    collision = None
    for index in range(17):
        handle = sessions.begin(metadata(f"capture-session-{index}"))
        prefix = handle.public_id[0]
        if prefix in prefixes:
            collision = prefix
            break
        prefixes[prefix] = handle.public_id

    assert collision is not None
    with pytest.raises(AmbiguousSessionId, match=collision):
        sessions.performance_snapshots({"id": [collision.upper()]})


def test_finish_rejects_unsafe_finite_clock_without_registry_mutation() -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata())
    before = sessions.performance_snapshots().sessions
    clock.monotonic = 1e308

    with pytest.raises(ValueError, match="finished_monotonic"):
        sessions.finish(handle, "completed")

    clock.monotonic = 100.0
    assert sessions.performance_snapshots().sessions == before
    assert sessions.snapshots()[0].state == "active"


@pytest.mark.asyncio
async def test_first_tool_start_commits_two_final_sequences_as_one_batch() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    events._sequence = MAX_CONTROL_INTEGER - 2  # type: ignore[attr-defined]
    subscription = events.subscribe(after=MAX_CONTROL_INTEGER - 2)

    observer.stream_event(ToolUseStart("slot", "tool-id", "tool-name"))

    published = subscription.replay + (
        await subscription.receive(1),
        await subscription.receive(1),
    )
    assert [item.type for item in published] == ["first_output", "tool_use"]
    assert [item.sequence for item in published] == [
        MAX_CONTROL_INTEGER - 1,
        MAX_CONTROL_INTEGER,
    ]
    subscription.close()


def test_first_tool_start_capacity_failure_changes_no_reducer_state() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    before = sessions.performance_snapshots().sessions
    retained_before = tuple(events._events)  # type: ignore[attr-defined]
    events._sequence = MAX_CONTROL_INTEGER - 1  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        observer.stream_event(ToolUseStart("slot", "tool-id", "tool-name"))

    events._sequence = 1  # type: ignore[attr-defined]
    assert sessions.performance_snapshots().sessions == before
    assert tuple(events._events) == retained_before  # type: ignore[attr-defined]


def test_retry_capacity_failure_changes_no_reducer_or_progress_state() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    observer.mark_retries_supported()
    before = sessions.performance_snapshots().sessions
    progress_before = dict(
        sessions._records[handle.key].progress_at  # type: ignore[attr-defined]
    )
    events._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        observer.record_retry()

    events._sequence = 1  # type: ignore[attr-defined]
    assert sessions.performance_snapshots().sessions == before
    assert (  # type: ignore[attr-defined]
        sessions._records[handle.key].progress_at == progress_before
    )


@pytest.mark.asyncio
async def test_failed_initial_capture_unregisters_performance_subscription() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, inactive_limit=20, events=events)
    prefixes: dict[str, str] = {}
    collision = None
    for index in range(17):
        handle = sessions.begin(metadata(f"subscribe-session-{index}"))
        prefix = handle.public_id[0]
        if prefix in prefixes:
            collision = prefix
            break
        prefixes[prefix] = handle.public_id

    assert collision is not None
    before = events.subscriber_count
    with pytest.raises(AmbiguousSessionId, match=collision):
        sessions.subscribe_performance({"id": [collision]}, after=None)

    assert events.subscriber_count == before


def _prepare_observer_clock_case(case, sessions, handle, observer) -> None:
    if case == "upstream_finished":
        observer.upstream_started()
    if case == "record_retry":
        observer.mark_retries_supported()


def _invoke_observer_clock_case(case, observer) -> None:
    if case == "upstream_started":
        observer.upstream_started()
    elif case == "upstream_finished":
        observer.upstream_finished()
    elif case == "stream_event":
        observer.stream_event(TextDelta("clock-secret"))
    elif case == "response":
        observer.response(
            CompletionResponse(
                "response-secret",
                "model-secret",
                (TextBlock("clock-secret"),),
                "end_turn",
                TokenUsage(7, 5),
            )
        )
    elif case == "count_tokens":
        observer.count_tokens(7)
    elif case == "record_retry":
        observer.record_retry()
    else:
        observer.set_reasoning_continuation("restored")


@pytest.mark.parametrize(
    "case",
    [
        "upstream_started",
        "upstream_finished",
        "stream_event",
        "response",
        "count_tokens",
        "record_retry",
        "set_reasoning_continuation",
    ],
)
@pytest.mark.parametrize("invalid_clock", ["monotonic", "wall"])
def test_observer_clock_failure_preserves_all_registry_state(
    case: str, invalid_clock: str
) -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    _prepare_observer_clock_case(case, sessions, handle, observer)
    before = sessions.performance_snapshots().sessions
    cursor_before = events.current_sequence
    progress_before = dict(
        sessions._records[handle.key].progress_at  # type: ignore[attr-defined]
    )
    valid_wall = clock.wall
    valid_monotonic = clock.monotonic
    if invalid_clock == "monotonic":
        clock.monotonic = 1e308
    else:
        clock.wall = datetime(2026, 1, 1)

    with pytest.raises(ValueError):
        _invoke_observer_clock_case(case, observer)

    clock.wall = valid_wall
    clock.monotonic = valid_monotonic
    assert sessions.performance_snapshots().sessions == before
    assert events.current_sequence == cursor_before
    assert (  # type: ignore[attr-defined]
        sessions._records[handle.key].progress_at == progress_before
    )


@pytest.mark.parametrize("case", ["count_tokens", "set_reasoning_continuation"])
def test_coalesced_observer_update_validates_clock_before_mutation(case: str) -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    record = sessions._records[handle.key]  # type: ignore[attr-defined]
    record.progress_at[handle.request_id] = 1e308
    before = sessions.performance_snapshots().sessions
    cursor_before = events.current_sequence
    progress_before = dict(record.progress_at)
    clock.monotonic = 1e308

    with pytest.raises(ValueError):
        _invoke_observer_clock_case(case, observer)

    clock.monotonic = 100.0
    assert sessions.performance_snapshots().sessions == before
    assert events.current_sequence == cursor_before
    assert record.progress_at == progress_before


def test_retry_unsafe_monotonic_preserves_observed_zero_and_cursor() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    observer.mark_retries_supported()
    before = sessions.performance_snapshots().sessions[0].performance
    cursor_before = events.current_sequence
    assert before.active_requests[0].retries.value == 0
    clock.monotonic = 1e308

    with pytest.raises(ValueError):
        observer.record_retry()

    clock.monotonic = 100.0
    after = sessions.performance_snapshots().sessions[0].performance
    assert after.active_requests[0].retries.value == 0
    assert events.current_sequence == cursor_before


def test_performance_capture_rejects_naive_wall_clock() -> None:
    clock = Clock()
    sessions = registry(clock)
    sessions.begin(metadata())
    clock.wall = datetime(2026, 1, 1)

    with pytest.raises(ValueError, match="wall"):
        sessions.performance_snapshots()
