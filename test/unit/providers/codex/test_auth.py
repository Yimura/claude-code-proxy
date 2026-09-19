import asyncio
import base64
import json
import sqlite3
import threading

import pytest

from claude_code_proxy.providers.codex.auth import CodexAccountIdentity, CodexAuth

NOW = 1_700_000_000


def access_token(email: str, *, verified: bool = True) -> str:
    profile = {"email": email, "email_verified": verified}
    payload = json.dumps({"https://api.openai.com/profile": profile}).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"e30.{encoded}.signature"


def write_json(path, expires, access="access", refresh="refresh", account_id="account"):
    path.write_text(
        json.dumps(
            {
                "openai": {
                    "access": access,
                    "refresh": refresh,
                    "expires": expires,
                    "accountId": account_id,
                }
            }
        )
    )


def write_database(
    path,
    expires,
    *,
    access="db-access",
    account_id="db-account",
):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE credential (id INTEGER, integration_id TEXT, active INTEGER, value TEXT, time_updated INTEGER)"
        )
        connection.execute(
            "INSERT INTO credential VALUES (1, 'openai', 1, ?, 0)",
            (
                json.dumps(
                    {
                        "access": access,
                        "refresh": "db-refresh",
                        "expires": expires,
                        "metadata": {"accountID": account_id},
                    }
                ),
            ),
        )


class Response:
    status_code = 200
    text = "ok"

    def json(self):
        return {
            "access_token": "new-access",
            "expires_in": 3600,
            "refresh_token": "new-refresh",
        }


class Client:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or Response()

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


async def wait_until(predicate, timeout=1):
    async def wait_for_predicate():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_predicate(), timeout=timeout)


async def test_json_initialize_returns_safe_account_identity(tmp_path):
    write_json(
        tmp_path / "auth.json",
        (NOW + 3600) * 1000,
        access=access_token("jane.doe@crimson7.io"),
        account_id="account-123",
    )
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    identity = await auth.initialize()

    assert identity == CodexAccountIdentity(
        account_id="account-123",
        masked_email="j***@crimson7.io",
        source="auth.json",
    )
    assert "jane.doe" not in repr(identity)


async def test_initialize_ignores_oversized_profile_email(tmp_path):
    write_json(
        tmp_path / "auth.json",
        (NOW + 3600) * 1000,
        access=access_token(f"a@{'x' * 400}"),
    )
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    identity = await auth.initialize()

    assert identity.masked_email is None


async def test_database_credential_takes_precedence(tmp_path):
    write_json(tmp_path / "auth.json", (NOW + 3600) * 1000)
    write_database(
        tmp_path / "opencode.db",
        (NOW + 3600) * 1000,
        access=access_token("database.owner@crimson7.io"),
    )
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    identity = await auth.initialize()

    assert identity == CodexAccountIdentity(
        account_id="db-account",
        masked_email="d***@crimson7.io",
        source="opencode.db",
    )
    assert await auth.get_auth() == (
        access_token("database.owner@crimson7.io"),
        "db-account",
    )


async def test_database_identity_uses_only_active_account(tmp_path):
    path = tmp_path / "opencode.db"
    write_database(
        path,
        (NOW + 3600) * 1000,
        access=access_token("active.owner@crimson7.io"),
        account_id="active-account",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO credential VALUES (2, 'openai', 0, ?, 0)",
            (
                json.dumps({
                    "access": access_token("inactive.owner@crimson7.io"),
                    "refresh": "inactive-refresh",
                    "expires": (NOW + 3600) * 1000,
                    "metadata": {"accountID": "inactive-account"},
                }),
            ),
        )
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    identity = await auth.initialize()

    assert identity.account_id == "active-account"
    assert identity.masked_email == "a***@crimson7.io"


@pytest.mark.parametrize(
    "token",
    [
        access_token("owner@crimson7.io", verified=False),
        "not-a-jwt",
    ],
)
async def test_initialize_falls_back_when_profile_email_is_not_trusted(
    tmp_path, token
):
    write_json(tmp_path / "auth.json", (NOW + 3600) * 1000, access=token)
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    identity = await auth.initialize()

    assert identity.masked_email is None
    assert identity.account_id == "account"


async def test_refreshed_access_token_supplies_startup_identity(tmp_path):
    class RefreshedResponse(Response):
        def json(self):
            return {
                "access_token": access_token("rotated.owner@crimson7.io"),
                "expires_in": 3600,
                "refresh_token": "new-refresh",
            }

    write_json(tmp_path / "auth.json", NOW * 1000)
    auth = CodexAuth(tmp_path, Client(RefreshedResponse()), lambda: NOW)

    identity = await auth.initialize()

    assert identity == CodexAccountIdentity(
        account_id="account",
        masked_email="r***@crimson7.io",
        source="auth.json",
    )
    assert "new-refresh" not in repr(identity)


async def test_json_fallback_and_cache_avoid_second_read(tmp_path):
    write_json(tmp_path / "auth.json", (NOW + 3600) * 1000)
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    await auth.initialize()
    (tmp_path / "auth.json").unlink()

    assert await auth.get_auth() == ("access", "account")


async def test_missing_credentials_names_login_command(tmp_path):
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    with pytest.raises(RuntimeError, match="opencode auth login"):
        await auth.initialize()


async def test_expiring_json_token_refreshes_and_persists_rotation(tmp_path):
    path = tmp_path / "auth.json"
    write_json(path, NOW * 1000)
    client = Client()
    auth = CodexAuth(tmp_path, client, lambda: NOW)

    await auth.initialize()

    assert await auth.get_auth() == ("new-access", "account")
    stored = json.loads(path.read_text())["openai"]
    assert stored["access"] == "new-access"
    assert stored["refresh"] == "new-refresh"
    assert "refresh" not in str(client.calls[0][0])


async def test_expired_token_without_refresh_requires_login(tmp_path):
    write_json(tmp_path / "auth.json", NOW * 1000, refresh="")
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    with pytest.raises(RuntimeError, match="no refresh token"):
        await auth.initialize()


async def test_refresh_failure_does_not_expose_response_tokens(tmp_path):
    class FailedResponse:
        status_code = 401
        text = "sample-access-token sample-refresh-token"

    write_json(tmp_path / "auth.json", NOW * 1000)
    auth = CodexAuth(tmp_path, Client(FailedResponse()), lambda: NOW)

    with pytest.raises(RuntimeError, match="Codex token refresh failed") as exc_info:
        await auth.initialize()

    message = str(exc_info.value)
    assert "sample-access-token" not in message
    assert "sample-refresh-token" not in message


async def test_invalid_refreshed_credential_is_not_persisted(tmp_path):
    class InvalidResponse(Response):
        def json(self):
            return {"access_token": "", "expires_in": 3600}

    path = tmp_path / "auth.json"
    write_json(path, NOW * 1000, access="original-access")
    auth = CodexAuth(tmp_path, Client(InvalidResponse()), lambda: NOW)

    with pytest.raises(RuntimeError, match="Invalid opencode credential"):
        await auth.initialize()

    assert json.loads(path.read_text())["openai"]["access"] == "original-access"


@pytest.mark.parametrize(
    "access, account_id, expires",
    [
        ("", "account", (NOW + 3600) * 1000),
        ("sample-access-token", "", (NOW + 3600) * 1000),
        ("sample-access-token", "account", "not-a-number"),
        ("sample-access-token", "account", float("nan")),
    ],
)
async def test_malformed_json_credential_requires_login_without_exposing_tokens(
    tmp_path, access, account_id, expires
):
    write_json(
        tmp_path / "auth.json",
        expires,
        access=access,
        refresh="sample-refresh-token",
        account_id=account_id,
    )
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    with pytest.raises(RuntimeError, match="Invalid opencode credential") as exc_info:
        await auth.initialize()

    message = str(exc_info.value)
    assert "sample-access-token" not in message
    assert "sample-refresh-token" not in message


async def test_initialize_does_not_block_event_loop(tmp_path, monkeypatch):
    loop = asyncio.get_running_loop()
    worker_started = asyncio.Event()
    release_worker = threading.Event()
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    def blocking_load(rejected_access):
        loop.call_soon_threadsafe(worker_started.set)
        if not release_worker.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        return {
            "access": "access",
            "refresh": "refresh",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
            "source": "json",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    initialize = asyncio.create_task(auth.initialize())

    try:
        await asyncio.wait_for(worker_started.wait(), timeout=1)
        assert not initialize.done()
    finally:
        release_worker.set()

    await asyncio.wait_for(initialize, timeout=1)


async def test_concurrent_callers_share_one_credential_load(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    def blocking_load(rejected_access):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        return {
            "access": "shared-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    first = asyncio.create_task(auth.get_auth())
    second = asyncio.create_task(auth.get_auth())

    try:
        await wait_until(started.is_set)
        assert calls == 1
    finally:
        release.set()

    assert await asyncio.gather(first, second) == [
        ("shared-access", "account"),
        ("shared-access", "account"),
    ]
    assert calls == 1


async def test_cancelled_waiter_does_not_cancel_shared_load(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    def blocking_load(rejected_access):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        return {
            "access": "shared-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    cancelled = asyncio.create_task(auth.get_auth())
    survivor = asyncio.create_task(auth.get_auth())
    await wait_until(started.is_set)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()

    assert await survivor == ("shared-access", "account")
    assert calls == 1


async def test_failed_shared_refresh_never_returns_stale_credentials(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    calls = 0
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "expired-access",
        "expires": NOW * 1000,
        "account_id": "account",
    }

    def failing_load(rejected_access):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(auth, "_load_credential", failing_load)
    first = asyncio.create_task(auth.get_auth())
    second = asyncio.create_task(auth.get_auth())

    await wait_until(started.is_set)
    assert calls == 1
    release.set()

    for waiter in (first, second):
        with pytest.raises(RuntimeError, match="refresh failed"):
            await waiter
    assert calls == 1

    def successful_load(rejected_access):
        nonlocal calls
        calls += 1
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", successful_load)

    assert await auth.get_auth() == ("recovered-access", "account")
    assert calls == 2


async def test_all_cancelled_waiters_leave_completed_operation_available(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    calls = 0
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    def blocking_load(rejected_access):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        return {
            "access": "completed-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    first = asyncio.create_task(auth.get_auth())
    second = asyncio.create_task(auth.get_auth())
    await wait_until(started.is_set)
    operation = auth._inflight

    first.cancel()
    second.cancel()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert operation is not None
    assert not operation.cancelled()

    release.set()
    await wait_until(lambda: operation.done() and auth._inflight is None)

    assert operation.exception() is None
    assert await auth.get_auth() == ("completed-access", "account")
    assert calls == 1


async def test_all_cancelled_waiters_consume_failure_and_allow_retry(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    calls = 0
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    def failing_load(rejected_access):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(auth, "_load_credential", failing_load)
    exception_consumed = asyncio.Event()
    consume_task_exception = auth._consume_task_exception

    def record_exception_consumption(task):
        consume_task_exception(task)
        exception_consumed.set()

    monkeypatch.setattr(
        auth, "_consume_task_exception", record_exception_consumption
    )
    first = asyncio.create_task(auth.get_auth())
    second = asyncio.create_task(auth.get_auth())
    await wait_until(started.is_set)
    operation = auth._inflight
    assert operation is not None
    first.cancel()
    second.cancel()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    release.set()
    await wait_until(
        lambda: operation.done()
        and auth._inflight is None
        and exception_consumed.is_set()
    )

    def successful_load(rejected_access):
        nonlocal calls
        calls += 1
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", successful_load)

    assert await auth.get_auth() == ("recovered-access", "account")
    assert calls == 2


async def test_recover_rejected_adopts_newer_cached_token(tmp_path, monkeypatch):
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "new-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def unexpected_load(rejected_access):
        raise AssertionError("fresh replacement cache must avoid credential I/O")

    monkeypatch.setattr(auth, "_load_credential", unexpected_load)

    assert await auth.recover_rejected("old-access") == ("new-access", "account")


async def test_recover_rejected_reloads_external_rotation(tmp_path):
    path = tmp_path / "auth.json"
    write_json(path, (NOW + 3600) * 1000, access="old-access")
    client = Client()
    auth = CodexAuth(tmp_path, client, lambda: NOW)
    await auth.initialize()
    write_json(path, (NOW + 3600) * 1000, access="external-access")

    assert await auth.recover_rejected("old-access") == (
        "external-access",
        "account",
    )
    assert client.calls == []


async def test_recover_rejected_refreshes_unchanged_stored_token(tmp_path):
    path = tmp_path / "auth.json"
    write_json(path, (NOW + 3600) * 1000, access="old-access")
    client = Client()
    auth = CodexAuth(tmp_path, client, lambda: NOW)
    await auth.initialize()

    assert await auth.recover_rejected("old-access") == ("new-access", "account")
    assert len(client.calls) == 1


async def test_recover_rejected_forces_once_after_joining_ordinary_load(
    tmp_path, monkeypatch
):
    ordinary_started = threading.Event()
    release_ordinary = threading.Event()
    forced_started = threading.Event()
    release_forced = threading.Event()
    follow_up_requested = asyncio.Event()
    release_follow_up = asyncio.Event()
    get_started = asyncio.Event()
    credential_requests = 0
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    credential_task = auth._credential_task

    def blocking_load(rejected_access):
        calls.append(rejected_access)
        if rejected_access is None:
            ordinary_started.set()
            if not release_ordinary.wait(timeout=1):
                raise TimeoutError("test did not release ordinary worker")
        else:
            forced_started.set()
            if not release_forced.wait(timeout=1):
                raise TimeoutError("test did not release forced worker")
        return {
            "access": (
                "rejected-access"
                if rejected_access is None
                else "recovered-access"
            ),
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    async def controlled_credential_task(rejected_access):
        nonlocal credential_requests
        if rejected_access is not None:
            credential_requests += 1
            if credential_requests == 2:
                follow_up_requested.set()
                await release_follow_up.wait()
        return await credential_task(rejected_access)

    async def get_during_handoff():
        get_started.set()
        return await auth.get_auth()

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    monkeypatch.setattr(auth, "_credential_task", controlled_credential_task)
    ordinary = asyncio.create_task(auth.get_auth())
    await wait_until(ordinary_started.is_set)
    first_recovery = asyncio.create_task(
        auth.recover_rejected("rejected-access")
    )
    assert calls == [None]

    release_ordinary.set()
    assert await ordinary == ("rejected-access", "account")
    await follow_up_requested.wait()
    concurrent_get = asyncio.create_task(get_during_handoff())
    await get_started.wait()

    assert not concurrent_get.done()
    assert calls == [None]
    release_follow_up.set()
    await wait_until(forced_started.is_set)
    second_recovery = asyncio.create_task(
        auth.recover_rejected("rejected-access")
    )
    release_forced.set()

    assert await asyncio.gather(
        first_recovery, second_recovery, concurrent_get
    ) == [
        ("recovered-access", "account"),
        ("recovered-access", "account"),
        ("recovered-access", "account"),
    ]
    assert calls == [None, "rejected-access"]


async def test_recover_rejected_stops_after_forced_result_repeats_token(
    tmp_path, monkeypatch
):
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    def repeated_load(rejected_access):
        calls.append(rejected_access)
        return {
            "access": "rejected-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", repeated_load)

    assert await auth.recover_rejected("rejected-access") == (
        "rejected-access",
        "account",
    )
    assert await auth.get_auth() == ("rejected-access", "account")
    assert await auth.get_auth() == ("rejected-access", "account")
    assert calls == ["rejected-access"]


async def test_get_auth_joins_active_rejected_token_recovery(tmp_path, monkeypatch):
    recovery_started = threading.Event()
    release_recovery = threading.Event()
    get_started = asyncio.Event()
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "old-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def blocking_load(rejected_access):
        calls.append(rejected_access)
        recovery_started.set()
        if not release_recovery.wait(timeout=1):
            raise TimeoutError("test did not release credential worker")
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    async def get_during_recovery():
        get_started.set()
        return await auth.get_auth()

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    recovery = asyncio.create_task(auth.recover_rejected("old-access"))
    await wait_until(recovery_started.is_set)
    async with auth._operation_lock:
        assert auth._inflight_rejected == "old-access"
    concurrent_get = asyncio.create_task(get_during_recovery())
    await get_started.wait()

    assert not concurrent_get.done()
    assert calls == ["old-access"]
    release_recovery.set()

    assert await asyncio.gather(recovery, concurrent_get) == [
        ("recovered-access", "account"),
        ("recovered-access", "account"),
    ]
    assert calls == ["old-access"]


async def test_cancelled_recovery_waiter_does_not_cancel_shared_recovery(
    tmp_path, monkeypatch
):
    recovery_started = threading.Event()
    release_recovery = threading.Event()
    calls = 0
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "old-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def blocking_load(rejected_access):
        nonlocal calls
        calls += 1
        recovery_started.set()
        if not release_recovery.wait(timeout=1):
            raise TimeoutError("test did not release recovery worker")
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    cancelled = asyncio.create_task(auth.recover_rejected("old-access"))
    survivor = asyncio.create_task(auth.recover_rejected("old-access"))
    await wait_until(recovery_started.is_set)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not survivor.done()
    release_recovery.set()

    assert await survivor == ("recovered-access", "account")
    assert calls == 1


async def test_old_token_recovery_does_not_block_newer_cached_token(
    tmp_path, monkeypatch
):
    recovery_started = threading.Event()
    release_recovery = threading.Event()
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "old-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def blocking_load(rejected_access):
        recovery_started.set()
        if not release_recovery.wait(timeout=1):
            raise TimeoutError("test did not release recovery worker")
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    recovery = asyncio.create_task(auth.recover_rejected("old-access"))
    await wait_until(recovery_started.is_set)
    async with auth._operation_lock:
        auth._cache = {
            "access": "newer-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    get_auth = asyncio.create_task(auth.get_auth())
    await wait_until(get_auth.done)
    assert await get_auth == ("newer-access", "account")

    release_recovery.set()
    assert await recovery == ("newer-access", "account")
    assert await auth.get_auth() == ("newer-access", "account")


async def test_failed_recovery_marks_cached_token_for_get_auth_retry(
    tmp_path, monkeypatch
):
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "rejected-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def failing_load(rejected_access):
        calls.append(rejected_access)
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(auth, "_load_credential", failing_load)
    with pytest.raises(RuntimeError, match="refresh failed"):
        await auth.recover_rejected("rejected-access")

    def successful_load(rejected_access):
        calls.append(rejected_access)
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", successful_load)

    assert await auth.get_auth() == ("recovered-access", "account")
    assert calls == ["rejected-access", "rejected-access"]


async def test_repeated_failed_recovery_never_returns_rejected_cache(
    tmp_path, monkeypatch
):
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "rejected-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def failing_load(rejected_access):
        calls.append(rejected_access)
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(auth, "_load_credential", failing_load)
    with pytest.raises(RuntimeError, match="refresh failed"):
        await auth.recover_rejected("rejected-access")
    with pytest.raises(RuntimeError, match="refresh failed"):
        await auth.get_auth()

    assert calls == ["rejected-access", "rejected-access"]


async def test_newer_cache_obsoletes_failed_rejection_marker(
    tmp_path, monkeypatch
):
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "rejected-access",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def failing_load(rejected_access):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(auth, "_load_credential", failing_load)
    with pytest.raises(RuntimeError, match="refresh failed"):
        await auth.recover_rejected("rejected-access")

    async with auth._operation_lock:
        auth._cache = {
            "access": "newer-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    assert await auth.get_auth() == ("newer-access", "account")
    async with auth._operation_lock:
        assert "rejected-access" not in auth._rejected_accesses


async def test_different_rejection_joins_active_cached_token_recovery(
    tmp_path, monkeypatch
):
    recovery_started = threading.Event()
    release_recovery = threading.Event()
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "cached-rejected",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def blocking_load(rejected_access):
        calls.append(rejected_access)
        recovery_started.set()
        if not release_recovery.wait(timeout=1):
            raise TimeoutError("test did not release recovery worker")
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", blocking_load)
    first = asyncio.create_task(auth.recover_rejected("cached-rejected"))
    await wait_until(recovery_started.is_set)
    different = asyncio.create_task(auth.recover_rejected("other-rejected"))
    await asyncio.sleep(0)

    assert not different.done()
    assert calls == ["cached-rejected"]
    release_recovery.set()

    assert await asyncio.gather(first, different) == [
        ("recovered-access", "account"),
        ("recovered-access", "account"),
    ]
    assert calls == ["cached-rejected"]


async def test_different_rejection_retries_known_rejected_cached_token(
    tmp_path, monkeypatch
):
    calls = []
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    auth._cache = {
        "access": "cached-rejected",
        "expires": (NOW + 3600) * 1000,
        "account_id": "account",
    }

    def failing_load(rejected_access):
        calls.append(rejected_access)
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(auth, "_load_credential", failing_load)
    with pytest.raises(RuntimeError, match="refresh failed"):
        await auth.recover_rejected("cached-rejected")

    def successful_load(rejected_access):
        calls.append(rejected_access)
        return {
            "access": "recovered-access",
            "expires": (NOW + 3600) * 1000,
            "account_id": "account",
        }

    monkeypatch.setattr(auth, "_load_credential", successful_load)

    assert await auth.recover_rejected("other-rejected") == (
        "recovered-access",
        "account",
    )
    assert calls == ["cached-rejected", "cached-rejected"]
