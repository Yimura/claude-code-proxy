from datetime import UTC, datetime

from httpx import ASGITransport, AsyncClient
import pytest

from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.observability import SessionRegistry


async def get(app, path: str):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://control"
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_disabled_control_app_advertises_only_inventory() -> None:
    sessions = SessionRegistry(10, performance_enabled=False)
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=123,
    )

    health = await get(app, "/v1/health")

    assert health.status_code == 200
    assert health.json()["capabilities"] == ["sessions", "agents"]
    assert (await get(app, "/v1/performance")).status_code == 404
    assert (await get(app, "/v1/performance/events")).status_code == 404


@pytest.mark.asyncio
async def test_enabled_control_app_advertises_and_serves_performance() -> None:
    sessions = SessionRegistry(10, performance_enabled=True)
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=123,
    )

    health = await get(app, "/v1/health")
    snapshot = await get(app, "/v1/performance")

    assert health.json()["capabilities"] == [
        "sessions",
        "agents",
        "performance",
        "performance_events",
    ]
    assert snapshot.status_code == 200
