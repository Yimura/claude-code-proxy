import asyncio
import json
import sqlite3
import threading

import pytest

from claude_code_proxy.providers.codex.auth import CodexAuth

NOW = 1_700_000_000


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


def write_database(path, expires):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE credential (id INTEGER, integration_id TEXT, active INTEGER, value TEXT, time_updated INTEGER)"
        )
        connection.execute(
            "INSERT INTO credential VALUES (1, 'openai', 1, ?, 0)",
            (
                json.dumps(
                    {
                        "access": "db-access",
                        "refresh": "db-refresh",
                        "expires": expires,
                        "metadata": {"accountID": "db-account"},
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


async def test_database_credential_takes_precedence(tmp_path):
    write_json(tmp_path / "auth.json", (NOW + 3600) * 1000)
    write_database(tmp_path / "opencode.db", (NOW + 3600) * 1000)
    auth = CodexAuth(tmp_path, clock=lambda: NOW)

    await auth.initialize()

    assert await auth.get_auth() == ("db-access", "db-account")


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
