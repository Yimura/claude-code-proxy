import json
import sqlite3
import pytest
from claude_code_proxy.providers.codex.auth import CodexAuth

NOW = 1_700_000_000


def write_json(path, expires, access="access", refresh="refresh"):
    path.write_text(json.dumps({"openai": {"access": access, "refresh": refresh, "expires": expires, "accountId": "account"}}))


def write_database(path, expires):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE credential (id INTEGER, integration_id TEXT, active INTEGER, value TEXT, time_updated INTEGER)")
        connection.execute("INSERT INTO credential VALUES (1, 'openai', 1, ?, 0)", (json.dumps({"access": "db-access", "refresh": "db-refresh", "expires": expires, "metadata": {"accountID": "db-account"}}),))


class Response:
    status_code = 200
    text = "ok"
    def json(self): return {"access_token": "new-access", "expires_in": 3600, "refresh_token": "new-refresh"}


class Client:
    def __init__(self): self.calls = []
    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return Response()


def test_database_credential_takes_precedence(tmp_path):
    write_json(tmp_path / "auth.json", (NOW + 3600) * 1000)
    write_database(tmp_path / "opencode.db", (NOW + 3600) * 1000)
    assert CodexAuth(tmp_path, clock=lambda: NOW).get_auth() == ("db-access", "db-account")


def test_json_fallback_and_cache_avoid_second_read(tmp_path):
    write_json(tmp_path / "auth.json", (NOW + 3600) * 1000)
    auth = CodexAuth(tmp_path, clock=lambda: NOW)
    assert auth.get_auth() == ("access", "account")
    (tmp_path / "auth.json").unlink()
    assert auth.get_auth() == ("access", "account")


def test_missing_credentials_names_login_command(tmp_path):
    with pytest.raises(RuntimeError, match="opencode auth login"):
        CodexAuth(tmp_path, clock=lambda: NOW).get_auth()


def test_expiring_json_token_refreshes_and_persists_rotation(tmp_path):
    path = tmp_path / "auth.json"
    write_json(path, NOW * 1000)
    client = Client()
    assert CodexAuth(tmp_path, client, lambda: NOW).get_auth() == ("new-access", "account")
    stored = json.loads(path.read_text())["openai"]
    assert stored["access"] == "new-access"
    assert stored["refresh"] == "new-refresh"
    assert "refresh" not in str(client.calls[0][0])


def test_expired_token_without_refresh_requires_login(tmp_path):
    write_json(tmp_path / "auth.json", NOW * 1000, refresh="")
    with pytest.raises(RuntimeError, match="no refresh token"):
        CodexAuth(tmp_path, clock=lambda: NOW).get_auth()
