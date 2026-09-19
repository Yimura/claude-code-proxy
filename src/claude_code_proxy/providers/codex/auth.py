"""Codex OAuth credential discovery, refresh, persistence, and caching."""

import asyncio
import base64
import binascii
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Literal

import httpx

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_URL = "https://auth.openai.com/oauth/token"
FRESHNESS_WINDOW_MS = 60_000
_PROFILE_CLAIM = "https://api.openai.com/profile"
_MAX_DISPLAY_EMAIL_LENGTH = 320
_SOURCE_LABELS = {"db": "opencode.db", "json": "auth.json"}


@dataclass(frozen=True)
class CodexAccountIdentity:
    """Redacted account metadata approved for startup reporting."""

    account_id: str
    masked_email: str | None
    source: Literal["opencode.db", "auth.json"]


class CodexAuth:
    def __init__(self, data_dir: Path, client=httpx, clock=time.time) -> None:
        self._data_dir = data_dir
        self._client = client
        self._clock = clock
        self._cache: dict[str, Any] = {
            "access": None,
            "expires": 0,
            "account_id": None,
            "source": None,
        }
        self._operation_lock = asyncio.Lock()
        self._inflight: asyncio.Task[tuple[str, str]] | None = None
        self._inflight_rejected: str | None = None
        self._recoveries: dict[str, asyncio.Task[tuple[str, str]]] = {}
        self._rejected_accesses: set[str] = set()
        self._recovery_active = asyncio.Event()

    async def initialize(self) -> CodexAccountIdentity:
        await self._resolve()
        source = _SOURCE_LABELS[self._cache["source"]]
        return CodexAccountIdentity(
            account_id=self._cache["account_id"],
            masked_email=_masked_email(self._cache["access"]),
            source=source,
        )

    async def get_auth(self) -> tuple[str, str]:
        if self._recovery_active.is_set():
            recovery = await self._recovery_for_cached_token()
            if recovery is not None:
                return await asyncio.shield(recovery)
        cached = self._cached_auth()
        if cached is not None:
            return cached
        return await self._resolve()

    async def recover_rejected(self, rejected_access: str) -> tuple[str, str]:
        recovery = await self._recovery_task(rejected_access)
        if isinstance(recovery, tuple):
            return recovery
        return await asyncio.shield(recovery)

    def _cached_auth(self) -> tuple[str, str] | None:
        now = int(self._clock() * 1000)
        if self._cache["access"] and self._cache["expires"] > now + FRESHNESS_WINDOW_MS:
            return self._cache["access"], self._cache["account_id"]
        return None

    async def _resolve(self, rejected_access: str | None = None) -> tuple[str, str]:
        while True:
            task, task_rejected = await self._credential_task(rejected_access)
            result = await asyncio.shield(task)
            if (
                rejected_access is None
                or task_rejected == rejected_access
                or result[0] != rejected_access
            ):
                return result

    async def _recovery_for_cached_token(
        self,
    ) -> asyncio.Task[tuple[str, str]] | None:
        async with self._operation_lock:
            access = self._cache["access"]
            if not isinstance(access, str) or not access:
                return None
            self._discard_obsolete_rejections(access)
            recovery = self._recoveries.get(access)
            if recovery is None and access in self._rejected_accesses:
                recovery = self._create_recovery(access)
            self._update_recovery_gate()
            return recovery

    async def _recovery_task(
        self, rejected_access: str
    ) -> asyncio.Task[tuple[str, str]] | tuple[str, str]:
        async with self._operation_lock:
            cached = self._cached_auth()
            if cached is not None and cached[0] != rejected_access:
                cached_access = cached[0]
                recovery = self._recoveries.get(cached_access)
                if recovery is None and cached_access in self._rejected_accesses:
                    recovery = self._create_recovery(cached_access)
                if recovery is not None:
                    self._update_recovery_gate()
                    return recovery
                self._discard_obsolete_rejections(cached_access)
                self._update_recovery_gate()
                return cached
            self._rejected_accesses.add(rejected_access)
            recovery = self._recoveries.get(rejected_access)
            if recovery is None:
                recovery = self._create_recovery(rejected_access)
            self._update_recovery_gate()
            return recovery

    def _create_recovery(
        self, rejected_access: str
    ) -> asyncio.Task[tuple[str, str]]:
        recovery = asyncio.create_task(self._run_recovery(rejected_access))
        self._recoveries[rejected_access] = recovery
        recovery.add_done_callback(self._consume_task_exception)
        return recovery

    def _discard_obsolete_rejections(self, access: str) -> None:
        self._rejected_accesses.intersection_update({access})

    def _update_recovery_gate(self) -> None:
        if self._recoveries or self._rejected_accesses:
            self._recovery_active.set()
        else:
            self._recovery_active.clear()

    async def _run_recovery(self, rejected_access: str) -> tuple[str, str]:
        current = asyncio.current_task()
        try:
            return await self._resolve(rejected_access)
        finally:
            async with self._operation_lock:
                if self._recoveries.get(rejected_access) is current:
                    del self._recoveries[rejected_access]
                    self._update_recovery_gate()

    async def _credential_task(
        self, rejected_access: str | None
    ) -> tuple[asyncio.Task[tuple[str, str]], str | None]:
        async with self._operation_lock:
            if self._inflight is None:
                self._inflight = asyncio.create_task(
                    self._run_credential_operation(rejected_access)
                )
                self._inflight_rejected = rejected_access
                self._inflight.add_done_callback(self._consume_task_exception)
            return self._inflight, self._inflight_rejected

    async def _run_credential_operation(
        self, rejected_access: str | None
    ) -> tuple[str, str]:
        current = asyncio.current_task()
        try:
            credential = await asyncio.to_thread(
                self._load_credential, rejected_access
            )
            result = credential["access"], credential["account_id"]
            async with self._operation_lock:
                if self._inflight is current:
                    cached = self._cached_auth()
                    if (
                        rejected_access is not None
                        and cached is not None
                        and cached[0] != rejected_access
                    ):
                        result = cached
                    else:
                        self._cache = {
                            "access": credential["access"],
                            "expires": credential["expires"],
                            "account_id": credential["account_id"],
                            "source": credential.get("source"),
                        }
                    if rejected_access is not None:
                        self._rejected_accesses.discard(rejected_access)
                    self._discard_obsolete_rejections(result[0])
                    self._update_recovery_gate()
                    self._inflight = None
                    self._inflight_rejected = None
            return result
        except BaseException:
            async with self._operation_lock:
                if self._inflight is current:
                    self._inflight = None
                    self._inflight_rejected = None
            raise

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[tuple[str, str]]) -> None:
        if not task.cancelled():
            task.exception()

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


def _masked_email(access_token: str) -> str | None:
    claims = _jwt_claims(access_token)
    profile = claims.get(_PROFILE_CLAIM)
    if not isinstance(profile, dict) or profile.get("email_verified") is not True:
        return None
    email = profile.get("email")
    if (
        not isinstance(email, str)
        or len(email) > _MAX_DISPLAY_EMAIL_LENGTH
        or email.count("@") != 1
    ):
        return None
    local, domain = email.split("@")
    if not local or not domain or email != email.strip():
        return None
    return f"{local[0]}***@{domain}"


def _jwt_claims(access_token: str) -> dict[str, Any]:
    parts = access_token.split(".")
    if len(parts) != 3 or not parts[1]:
        return {}
    payload = parts[1].encode("ascii")
    payload += b"=" * (-len(payload) % 4)
    try:
        decoded = base64.b64decode(payload, altchars=b"-_", validate=True)
        claims = json.loads(decoded)
    except (UnicodeError, ValueError, binascii.Error):
        return {}
    return claims if isinstance(claims, dict) else {}
