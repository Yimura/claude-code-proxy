from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import threading

import pytest

from claude_code_proxy.observability import (
    AmbiguousSessionId,
    InvalidSessionFilter,
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


def metadata(client_session_id: str | None = "sensitive-session") -> SessionMetadata:
    return SessionMetadata(
        client_session_id=client_session_id,
        client_model="claude-opus",
        upstream_model="openai/gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
    )


def registry(clock: Clock, inactive_limit: int = 10) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
    )


def test_constructor_enforces_signed_64_inactive_limit() -> None:
    maximum = 2**63 - 1

    assert SessionRegistry(maximum).inactive_limit == maximum
    with pytest.raises(ValueError, match="inactive_limit"):
        SessionRegistry(maximum + 1)


def test_constructor_rejects_negative_inactive_limit() -> None:
    with pytest.raises(ValueError, match="inactive_limit"):
        SessionRegistry(-1)


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
            client_session_id="raw-\ud800-session",
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
            client_session_id="sensitive-session",
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
            client_session_id="idle-row",
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
            client_session_id="failed-row",
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
