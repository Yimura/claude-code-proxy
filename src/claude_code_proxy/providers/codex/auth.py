"""Codex OAuth credential discovery, refresh, persistence, and caching."""

import asyncio
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any

import httpx

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_URL = "https://auth.openai.com/oauth/token"
FRESHNESS_WINDOW_MS = 60_000


class CodexAuth:
    def __init__(self, data_dir: Path, client=httpx, clock=time.time) -> None:
        self._data_dir = data_dir
        self._client = client
        self._clock = clock
        self._cache: dict[str, Any] = {"access": None, "expires": 0, "account_id": None}

    async def initialize(self) -> None:
        await self._reload()

    async def get_auth(self) -> tuple[str, str]:
        cached = self._cached_auth()
        if cached is not None:
            return cached
        return await self._reload()

    def _cached_auth(self) -> tuple[str, str] | None:
        now = int(self._clock() * 1000)
        if self._cache["access"] and self._cache["expires"] > now + FRESHNESS_WINDOW_MS:
            return self._cache["access"], self._cache["account_id"]
        return None

    async def _reload(self) -> tuple[str, str]:
        credential = await asyncio.to_thread(self._load_credential, None)
        self._cache = {
            "access": credential["access"],
            "expires": credential["expires"],
            "account_id": credential["account_id"],
        }
        return credential["access"], credential["account_id"]

    def _load_credential(self, rejected_access: str | None) -> dict[str, Any]:
        credential = self._read_credential()
        self._validate_credential(credential)
        now = int(self._clock() * 1000)
        if (
            rejected_access is not None
            and credential["access"] == rejected_access
        ) or credential["expires"] <= now + FRESHNESS_WINDOW_MS:
            credential = self._refresh(credential)
            self._validate_credential(credential)
        return credential

    def _validate_credential(self, credential: dict[str, Any]) -> None:
        access = credential.get("access")
        account_id = credential.get("account_id")
        expires = credential.get("expires")
        if (
            not isinstance(access, str)
            or not access
            or not isinstance(account_id, str)
            or not account_id
            or not isinstance(expires, (int, float))
            or isinstance(expires, bool)
            or not math.isfinite(expires)
            or expires <= 0
        ):
            raise RuntimeError(
                "Invalid opencode credential. Run 'opencode auth login'."
            )

    def _read_credential(self) -> dict[str, Any]:
        credential = self._read_database_credential()
        if credential is not None:
            return credential
        credential = self._read_json_credential()
        if credential is not None:
            return credential
        raise RuntimeError(f"No opencode auth found in {self._data_dir}. Run 'opencode auth login'.")

    def _read_database_credential(self):
        path = self._data_dir / "opencode.db"
        if not path.exists():
            return None
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT value FROM credential WHERE integration_id = 'openai' AND active = 1 LIMIT 1").fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        metadata = value.get("metadata") or {}
        return {"access": value.get("access", ""), "refresh": value.get("refresh", ""), "expires": value.get("expires", 0), "account_id": metadata.get("accountID", ""), "source": "db"}

    def _read_json_credential(self):
        path = self._data_dir / "auth.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8")).get("openai", {})
        return {"access": value.get("access", ""), "refresh": value.get("refresh", ""), "expires": value.get("expires", 0), "account_id": value.get("accountId", ""), "source": "json"}

    def _refresh(self, credential):
        if not credential["refresh"]:
            raise RuntimeError("Codex token expired, no refresh token. Run 'opencode auth login'.")
        response = self._client.post(CODEX_AUTH_URL, data={"grant_type": "refresh_token", "refresh_token": credential["refresh"], "client_id": CODEX_CLIENT_ID}, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code != 200:
            raise RuntimeError(f"Codex token refresh failed: {response.status_code}")
        body = response.json()
        credential = dict(credential)
        credential["access"] = body["access_token"]
        credential["expires"] = int(self._clock() * 1000) + body.get("expires_in", 864000) * 1000
        self._validate_credential(credential)
        self._save_refreshed(credential, body.get("refresh_token"))
        if body.get("refresh_token"):
            credential["refresh"] = body["refresh_token"]
        return credential

    def _save_refreshed(self, credential, new_refresh):
        if credential["source"] == "db":
            self._save_database_credential(credential, new_refresh)
            return
        path = self._data_dir / "auth.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["openai"]["access"] = credential["access"]
        data["openai"]["expires"] = credential["expires"]
        if new_refresh:
            data["openai"]["refresh"] = new_refresh
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_database_credential(self, credential, new_refresh):
        path = self._data_dir / "opencode.db"
        with sqlite3.connect(path) as connection:
            row = connection.execute("SELECT id, value FROM credential WHERE integration_id = 'openai' AND active = 1 LIMIT 1").fetchone()
            if row is None:
                return
            value = json.loads(row[1])
            value["access"] = credential["access"]
            value["expires"] = credential["expires"]
            if new_refresh:
                value["refresh"] = new_refresh
            connection.execute("UPDATE credential SET value = ?, time_updated = ? WHERE id = ?", (json.dumps(value), int(self._clock() * 1000), row[0]))
