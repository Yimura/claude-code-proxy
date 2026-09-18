import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import signal
import socket
from types import SimpleNamespace

from fastapi import FastAPI
import pytest
import uvicorn

import claude_code_proxy.server as server_module
from claude_code_proxy.observability import SessionRegistry


class FakeLease:
    def __init__(self, close_error=None):
        self.socket = object()
        self.close_calls = 0
        self.close_error = close_error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeServer:
    def __init__(
        self,
        name,
        *,
        start_error=None,
        completion_error=None,
        stubborn=False,
    ):
        self.name = name
        self.started = False
        self.should_exit = False
        self.force_exit = False
        self.start_error = start_error
        self.completion_error = completion_error
        self.stubborn = stubborn
        self.allow_start = asyncio.Event()
        self.allow_start.set()
        self.entered = asyncio.Event()
        self.ready = asyncio.Event()
        self.complete = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cancelled = False
        self.socket_calls = []
        self.timeline = []

    async def serve(self, sockets=None):
        self.timeline.append(f"{self.name}:serve")
        self.socket_calls.append(sockets)
        self.entered.set()
        try:
            await self.allow_start.wait()
            if self.start_error is not None:
                raise self.start_error
            self.started = True
            self.timeline.append(f"{self.name}:started")
            self.ready.set()
            if self.stubborn:
                await asyncio.Event().wait()
            while not self.complete.is_set() and not self.should_exit:
                await asyncio.sleep(0)
            if self.complete.is_set() and self.completion_error is not None:
                raise self.completion_error
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.stopped.set()


class GraceGateServer(FakeServer):
    def __init__(self, name):
        super().__init__(name)
        self.allow_shutdown = asyncio.Event()
        self.shutdown_entered = asyncio.Event()

    async def serve(self, sockets=None):
        self.socket_calls.append(sockets)
        self.entered.set()
        try:
            self.started = True
            self.ready.set()
            while not self.should_exit:
                await asyncio.sleep(0)
            self.shutdown_entered.set()
            await self.allow_shutdown.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.stopped.set()


class StopOnStartServer(FakeServer):
    def __init__(self, name, stop):
        super().__init__(name)
        self.stop = stop

    async def serve(self, sockets=None):
        self.socket_calls.append(sockets)
        self.entered.set()
        try:
            self.started = True
            self.stop.set()
            self.ready.set()
            while not self.should_exit:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.stopped.set()


class QueueFactory:
    def __init__(self, *servers):
        self.servers = iter(servers)
        self.configs = []

    def __call__(self, config):
        self.configs.append(config)
        return next(self.servers)


def runtime():
    return SimpleNamespace(
        settings=SimpleNamespace(
            proxy_host="198.51.100.7",
            proxy_port=9876,
            openai_transport="litellm",
        ),
        service=object(),
        codex_auth=object(),
        sessions=SessionRegistry(7),
        started_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )


@pytest.fixture
def fake_apps(monkeypatch):
    public_app = object()
    control_app = object()
    observed = {}

    def create_public(received_runtime):
        observed["runtime"] = received_runtime
        return public_app

    def create_control(sessions, *, started_at):
        observed["sessions"] = sessions
        observed["started_at"] = started_at
        return control_app

    monkeypatch.setattr(server_module, "create_app", create_public)
    monkeypatch.setattr(server_module, "create_control_app", create_control)
    return public_app, control_app, observed


async def test_control_starts_before_public_and_gates_public_start(fake_apps):
    control = FakeServer("control")
    public = FakeServer("public")
    control.allow_start.clear()
    lease = FakeLease()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(), lease, server_factory=QueueFactory(control, public)
        )
    )

    await control.entered.wait()
    await asyncio.sleep(0)
    assert not public.entered.is_set()

    control.allow_start.set()
    await public.entered.wait()
    assert control.started

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_control_started_and_done_same_turn_prevents_public_start(fake_apps):
    control = FakeServer("control")
    control.complete.set()
    public = FakeServer("public")
    lease = FakeLease()
    ready = asyncio.Event()

    with pytest.raises(RuntimeError, match="control.*startup"):
        await server_module.serve_proxy(
            runtime(),
            lease,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )

    assert control.started
    assert not public.entered.is_set()
    assert not ready.is_set()
    assert lease.close_calls == 1


async def test_control_started_and_stop_same_turn_prevents_public_start(fake_apps):
    stop = asyncio.Event()
    control = StopOnStartServer("control", stop)
    public = FakeServer("public")
    lease = FakeLease()
    ready = asyncio.Event()

    await server_module.serve_proxy(
        runtime(),
        lease,
        stop_event=stop,
        server_factory=QueueFactory(control, public),
        _ready_event=ready,
    )

    assert control.started
    assert not public.entered.is_set()
    assert not ready.is_set()
    assert lease.close_calls == 1


async def test_ready_hook_is_set_only_after_both_servers_start(fake_apps):
    control = FakeServer("control")
    public = FakeServer("public")
    control.allow_start.clear()
    public.allow_start.clear()
    ready = asyncio.Event()
    stop = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )

    await control.entered.wait()
    assert not ready.is_set()
    control.allow_start.set()
    await public.entered.wait()
    assert not ready.is_set()
    public.allow_start.set()
    await ready.wait()
    assert control.started and public.started

    stop.set()
    await task


async def test_external_stop_gracefully_stops_both_and_closes_lease_once(fake_apps):
    control = FakeServer("control")
    public = FakeServer("public")
    lease = FakeLease()
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            lease,
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()

    stop.set()
    await task

    assert control.should_exit and public.should_exit
    assert control.stopped.is_set() and public.stopped.is_set()
    assert lease.close_calls == 1


@pytest.mark.parametrize(
    ("uvicorn_timeout", "hard_timeout", "message"),
    [
        (-0.1, 1.0, "Uvicorn graceful shutdown timeout"),
        (1.0, 1.0, "Coordinator hard shutdown timeout"),
        (2.0, 1.0, "Coordinator hard shutdown timeout"),
    ],
)
async def test_shutdown_deadlines_are_validated_and_lease_is_closed(
    fake_apps,
    uvicorn_timeout,
    hard_timeout,
    message,
):
    lease = FakeLease()
    control = FakeServer("control")
    public = FakeServer("public")

    with pytest.raises(ValueError, match=message):
        await server_module.serve_proxy(
            runtime(),
            lease,
            server_factory=QueueFactory(control, public),
            _uvicorn_grace_timeout=uvicorn_timeout,
            _hard_timeout=hard_timeout,
        )

    assert not control.entered.is_set()
    assert not public.entered.is_set()
    assert lease.close_calls == 1


async def test_app_constructors_receive_exact_shared_runtime_values(fake_apps):
    _, _, observed = fake_apps
    shared_runtime = runtime()
    control = FakeServer("control")
    public = FakeServer("public")
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            shared_runtime,
            FakeLease(),
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()

    assert observed["runtime"] is shared_runtime
    assert observed["sessions"] is shared_runtime.sessions
    assert observed["started_at"] is shared_runtime.started_at

    stop.set()
    await task


async def test_control_startup_exception_prevents_public_start_and_preserves_cause(
    fake_apps,
):
    cause = ValueError("control boom")
    control = FakeServer("control", start_error=cause)
    public = FakeServer("public")
    lease = FakeLease()

    with pytest.raises(RuntimeError, match="control.*startup") as raised:
        await server_module.serve_proxy(
            runtime(), lease, server_factory=QueueFactory(control, public)
        )

    assert raised.value.__cause__ is cause
    assert not public.entered.is_set()
    assert lease.close_calls == 1


async def test_system_exit_during_control_startup_is_reported_as_runtime_failure(
    fake_apps,
):
    control = FakeServer("control", start_error=SystemExit(1))
    public = FakeServer("public")
    lease = FakeLease()

    with pytest.raises(RuntimeError, match="control.*startup") as raised:
        await server_module.serve_proxy(
            runtime(), lease, server_factory=QueueFactory(control, public)
        )

    assert isinstance(raised.value.__cause__, SystemExit)
    assert lease.close_calls == 1


async def test_public_startup_exception_stops_control_and_preserves_cause(fake_apps):
    cause = ValueError("public boom")
    control = FakeServer("control")
    public = FakeServer("public", start_error=cause)
    lease = FakeLease()

    with pytest.raises(RuntimeError, match="public.*startup") as raised:
        await server_module.serve_proxy(
            runtime(), lease, server_factory=QueueFactory(control, public)
        )

    assert raised.value.__cause__ is cause
    assert control.should_exit
    assert control.stopped.is_set()
    assert lease.close_calls == 1


@pytest.mark.parametrize("finished_name", ["control", "public"])
async def test_server_completion_after_readiness_stops_peer_and_raises(
    fake_apps, finished_name
):
    control = FakeServer("control")
    public = FakeServer("public")
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()

    finished = control if finished_name == "control" else public
    peer = public if finished is control else control
    finished.complete.set()

    with pytest.raises(RuntimeError, match=f"{finished_name}.*unexpectedly"):
        await task
    assert peer.should_exit
    assert peer.stopped.is_set()


async def test_external_cancellation_stops_servers_closes_lease_and_propagates(
    fake_apps,
):
    control = FakeServer("control")
    public = FakeServer("public")
    lease = FakeLease()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            lease,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert control.should_exit and public.should_exit
    assert control.stopped.is_set() and public.stopped.is_set()
    assert lease.close_calls == 1


async def test_repeated_cancellation_cannot_interrupt_cleanup(fake_apps):
    baseline = set(asyncio.all_tasks())
    control = FakeServer("control")
    public = GraceGateServer("public")
    lease = FakeLease()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            lease,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
            _uvicorn_grace_timeout=0.1,
            _hard_timeout=1,
        )
    )
    await ready.wait()

    task.cancel("original cancellation")
    await public.shutdown_entered.wait()
    task.cancel("repeated cancellation")
    for _ in range(3):
        await asyncio.sleep(0)
    public.allow_shutdown.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value.args == ("original cancellation",)
    assert control.stopped.is_set() and public.stopped.is_set()
    assert not public.force_exit
    assert lease.close_calls == 1
    await asyncio.sleep(0)
    assert set(asyncio.all_tasks()) == baseline
    assert not {
        "control-server",
        "public-server",
        "server-cleanup",
        "shutdown-force-waiter",
    } & {pending.get_name() for pending in asyncio.all_tasks()}


async def test_graceful_shutdown_waits_for_all_servers_within_grace():
    control = FakeServer("control")
    public = GraceGateServer("public")
    control_task = asyncio.create_task(control.serve())
    public_task = asyncio.create_task(public.serve())
    await control.ready.wait()
    await public.ready.wait()

    cleanup = asyncio.create_task(
        server_module._stop_server_tasks(
            [control, public],
            [control_task, public_task],
            asyncio.Event(),
            1,
        )
    )
    await control.stopped.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not cleanup.done()
    assert not public.force_exit
    assert not public.cancelled

    public.allow_shutdown.set()
    assert await cleanup == []


async def test_stubborn_servers_are_forced_and_cancelled_without_sleep(fake_apps):
    control = FakeServer("control", stubborn=True)
    public = FakeServer("public", stubborn=True)
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
            _uvicorn_grace_timeout=0,
            _hard_timeout=0.001,
        )
    )
    await ready.wait()

    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert control.force_exit and public.force_exit
    assert control.cancelled and public.cancelled


async def test_force_cancels_and_awaits_fake_owned_child_tasks(fake_apps):
    baseline = set(asyncio.all_tasks())
    child_cancelled = [asyncio.Event(), asyncio.Event()]

    async def owned_child(cancelled):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    control = FakeServer("control", stubborn=True)
    public = FakeServer("public", stubborn=True)
    child_tasks = [
        asyncio.create_task(owned_child(child_cancelled[0])),
        asyncio.create_task(owned_child(child_cancelled[1])),
    ]
    for fake, child in zip((control, public), child_tasks, strict=True):
        owned = {child}
        child.add_done_callback(owned.discard)
        fake.server_state = SimpleNamespace(tasks=owned)

    force = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
            _force_event=force,
            _uvicorn_grace_timeout=0.1,
            _hard_timeout=0.5,
        )
    )
    await ready.wait()

    try:
        force.set()
        await asyncio.wait_for(task, timeout=1)

        assert all(cancelled.is_set() for cancelled in child_cancelled)
        assert all(child.done() for child in child_tasks)
        assert not control.server_state.tasks
        assert not public.server_state.tasks
    finally:
        for child in child_tasks:
            if not child.done():
                child.cancel()
        await asyncio.gather(*child_tasks, return_exceptions=True)

    await asyncio.sleep(0)
    assert set(asyncio.all_tasks()) == baseline


async def test_graceful_owned_task_resisting_cancellation_hits_hard_deadline(
    fake_apps,
):
    cancel_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_child():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancel_seen.set()

    control = FakeServer("control")
    public = FakeServer("public")
    child = asyncio.create_task(resistant_child())
    owned = {child}
    child.add_done_callback(owned.discard)
    control.server_state = SimpleNamespace(tasks=owned)
    lease = FakeLease()
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            lease,
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
            _uvicorn_grace_timeout=0,
            _hard_timeout=0.001,
        )
    )
    await ready.wait()
    child.cancel()
    await cancel_seen.wait()

    try:
        stop.set()
        with pytest.raises(RuntimeError, match="cleanup failed") as raised:
            await asyncio.wait_for(task, timeout=1)

        assert "resisted cancellation" in str(raised.value.__cause__)
        assert control.force_exit and public.force_exit
        assert lease.close_calls == 1
        assert not child.done()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, child, return_exceptions=True)


async def test_force_reports_unsupported_owned_task_shape(fake_apps):
    control = FakeServer("control", stubborn=True)
    public = FakeServer("public", stubborn=True)
    control.server_state = SimpleNamespace(tasks={object()})
    force = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
            _force_event=force,
            _uvicorn_grace_timeout=0.1,
            _hard_timeout=0.5,
        )
    )
    await ready.wait()

    force.set()
    with pytest.raises(RuntimeError, match="cleanup failed") as raised:
        await task

    assert "non-Task" in str(raised.value.__cause__)
    assert control.cancelled and public.cancelled


async def test_simultaneous_server_exceptions_are_both_preserved(fake_apps):
    control_error = ValueError("control failed")
    public_error = LookupError("public failed")
    control = FakeServer("control", completion_error=control_error)
    public = FakeServer("public", completion_error=public_error)
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()

    control.complete.set()
    public.complete.set()
    with pytest.raises(RuntimeError) as raised:
        await task

    cause = raised.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert cause.exceptions == (control_error, public_error)


async def test_only_control_receives_prebound_socket(fake_apps):
    control = FakeServer("control")
    public = FakeServer("public")
    lease = FakeLease()
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            lease,
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()
    stop.set()
    await task

    assert control.socket_calls == [[lease.socket]]
    assert public.socket_calls == [None]


async def test_apps_share_runtime_data_configs_are_explicit_and_routes_are_separate():
    shared_runtime = runtime()
    control = FakeServer("control")
    public = FakeServer("public")
    factory = QueueFactory(control, public)
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            shared_runtime,
            FakeLease(),
            stop_event=stop,
            server_factory=factory,
            _ready_event=ready,
        )
    )
    await ready.wait()

    control_config, public_config = factory.configs
    public_app = public_config.app
    control_app = control_config.app
    public_paths = {
        route.path for route in public_app.routes if hasattr(route, "path")
    }
    control_paths = {
        route.path for route in control_app.routes if hasattr(route, "path")
    }

    assert public_app.state.runtime is shared_runtime
    health_route = next(
        route for route in control_app.routes if route.path == "/v1/health"
    )
    control_context = health_route.endpoint.__closure__[0].cell_contents
    assert control_context.sessions is shared_runtime.sessions
    assert control_context.started_at == shared_runtime.started_at
    assert "/v1/health" not in public_paths
    assert "/v1/sessions" not in public_paths
    assert {"/v1/health", "/v1/sessions"} <= control_paths

    assert public_config.host == shared_runtime.settings.proxy_host
    assert public_config.port == shared_runtime.settings.proxy_port
    for config in factory.configs:
        assert config.lifespan == "on"
        assert config.access_log is False
        assert config.log_config is None
        assert (
            config.timeout_graceful_shutdown
            == server_module.UVICORN_GRACEFUL_TIMEOUT_SECONDS
        )
    assert (
        server_module.UVICORN_GRACEFUL_TIMEOUT_SECONDS
        < server_module.COORDINATOR_HARD_TIMEOUT_SECONDS
    )

    stop.set()
    await task


async def test_real_uvicorn_drains_then_cancels_request_and_runs_lifespan(
    monkeypatch,
):
    baseline = set(asyncio.all_tasks())
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    request_cleanup_gate = asyncio.Event()
    lifespan_shutdown = asyncio.Event()
    request_tasks = []

    @asynccontextmanager
    async def control_lifespan(_application):
        yield
        lifespan_shutdown.set()

    control_app = FastAPI(lifespan=control_lifespan)

    @control_app.get("/block")
    async def block_request():
        request_tasks.append(asyncio.current_task())
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            await request_cleanup_gate.wait()
            raise

    public_app = FastAPI()
    monkeypatch.setattr(server_module, "create_app", lambda runtime: public_app)
    monkeypatch.setattr(
        server_module,
        "create_control_app",
        lambda sessions, *, started_at: control_app,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    address = listener.getsockname()
    lease = FakeLease()
    lease.socket = listener
    shared_runtime = runtime()
    shared_runtime.settings.proxy_host = "127.0.0.1"
    shared_runtime.settings.proxy_port = 0
    servers = []

    def factory(config):
        managed = server_module.ManagedServer(config)
        servers.append(managed)
        return managed

    stop = asyncio.Event()
    ready = asyncio.Event()
    proxy_task = asyncio.create_task(
        server_module.serve_proxy(
            shared_runtime,
            lease,
            stop_event=stop,
            server_factory=factory,
            _ready_event=ready,
            _uvicorn_grace_timeout=0.01,
            _hard_timeout=0.5,
        )
    )
    ready_waiter = asyncio.create_task(ready.wait())
    writer = None
    client_task = None
    try:
        done, _ = await asyncio.wait_for(
            asyncio.wait(
                {proxy_task, ready_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            ),
            timeout=1,
        )
        if proxy_task in done:
            await proxy_task
        await ready_waiter

        reader, writer = await asyncio.open_connection(*address)
        writer.write(
            b"GET /block HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Connection: close\r\n\r\n"
        )
        await writer.drain()
        client_task = asyncio.create_task(reader.read())
        await asyncio.wait_for(request_started.wait(), timeout=1)
        serve_tasks = {
            task
            for task in asyncio.all_tasks()
            if task.get_name() in {"control-server", "public-server"}
        }
        assert len(serve_tasks) == 2

        stop.set()
        await asyncio.wait_for(request_cancelled.wait(), timeout=1)
        await asyncio.wait_for(
            asyncio.gather(*serve_tasks, return_exceptions=True),
            timeout=1,
        )
        for _ in range(100):
            await asyncio.sleep(0)
        assert not proxy_task.done()
        assert lease.close_calls == 0
        request_cleanup_gate.set()
        await asyncio.wait_for(proxy_task, timeout=2)
        await asyncio.wait_for(client_task, timeout=1)
    finally:
        stop.set()
        request_cleanup_gate.set()
        ready_waiter.cancel()
        await asyncio.gather(ready_waiter, return_exceptions=True)
        if not proxy_task.done():
            proxy_task.cancel()
        await asyncio.gather(proxy_task, return_exceptions=True)
        if client_task is not None and not client_task.done():
            client_task.cancel()
            await asyncio.gather(client_task, return_exceptions=True)
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        listener.close()

    assert request_cancelled.is_set()
    assert lifespan_shutdown.is_set()
    assert request_tasks and all(task.done() for task in request_tasks)
    assert proxy_task.done()
    assert all(not managed.server_state.tasks for managed in servers)
    assert lease.close_calls == 1
    await asyncio.sleep(0)
    assert set(asyncio.all_tasks()) == baseline


async def test_real_uvicorn_force_cancels_requests_and_lifespans(monkeypatch):
    baseline = set(asyncio.all_tasks())
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    lifespan_cancelled = [asyncio.Event(), asyncio.Event()]
    graceful_shutdown = [asyncio.Event(), asyncio.Event()]
    request_tasks = []

    def application(index):
        @asynccontextmanager
        async def lifespan(_application):
            try:
                yield
            except asyncio.CancelledError:
                lifespan_cancelled[index].set()
                raise
            else:
                graceful_shutdown[index].set()

        return FastAPI(lifespan=lifespan)

    control_app = application(0)
    public_app = application(1)

    @control_app.get("/block")
    async def block_request():
        request_tasks.append(asyncio.current_task())
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    monkeypatch.setattr(server_module, "create_app", lambda runtime: public_app)
    monkeypatch.setattr(
        server_module,
        "create_control_app",
        lambda sessions, *, started_at: control_app,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    address = listener.getsockname()
    lease = FakeLease()
    lease.socket = listener
    shared_runtime = runtime()
    shared_runtime.settings.proxy_host = "127.0.0.1"
    shared_runtime.settings.proxy_port = 0
    servers = []

    def factory(config):
        managed = server_module.ManagedServer(config)
        servers.append(managed)
        return managed

    def current_owned_tasks():
        owned = set()
        for managed in servers:
            owned.update(managed.server_state.tasks)
            lifespan = getattr(managed, "lifespan", None)
            for candidate in asyncio.all_tasks():
                coroutine = candidate.get_coro()
                frame = getattr(coroutine, "cr_frame", None)
                if frame is not None and frame.f_locals.get("self") is lifespan:
                    owned.add(candidate)
        return owned

    stop = asyncio.Event()
    force = asyncio.Event()
    ready = asyncio.Event()
    proxy_task = asyncio.create_task(
        server_module.serve_proxy(
            shared_runtime,
            lease,
            stop_event=stop,
            server_factory=factory,
            _ready_event=ready,
            _force_event=force,
            _uvicorn_grace_timeout=0.1,
            _hard_timeout=0.5,
        )
    )
    writer = None
    client_task = None
    owned_before = set()
    serve_tasks = set()
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
        reader, writer = await asyncio.open_connection(*address)
        writer.write(
            b"GET /block HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Connection: close\r\n\r\n"
        )
        await writer.drain()
        client_task = asyncio.create_task(reader.read())
        await asyncio.wait_for(request_started.wait(), timeout=1)
        owned_before = current_owned_tasks()
        serve_tasks = {
            task
            for task in asyncio.all_tasks()
            if task.get_name() in {"control-server", "public-server"}
        }

        stop.set()
        force.set()
        await asyncio.wait_for(proxy_task, timeout=1)

        assert request_cancelled.is_set()
        assert all(marker.is_set() for marker in lifespan_cancelled)
        assert not any(marker.is_set() for marker in graceful_shutdown)
        assert request_tasks and all(task.done() for task in request_tasks)
        assert owned_before and all(task.done() for task in owned_before)
        assert serve_tasks and all(task.done() for task in serve_tasks)
        assert all(not managed.server_state.tasks for managed in servers)
        assert lease.close_calls == 1
        await asyncio.wait_for(client_task, timeout=1)
    finally:
        stop.set()
        force.set()
        if not proxy_task.done():
            proxy_task.cancel()
        await asyncio.gather(proxy_task, return_exceptions=True)
        lingering = current_owned_tasks() | owned_before | serve_tasks
        current = asyncio.current_task()
        for pending in lingering:
            if pending is not current and not pending.done():
                pending.cancel()
        await asyncio.gather(*lingering, return_exceptions=True)
        for managed in servers:
            for uvicorn_listener in getattr(managed, "servers", ()):
                uvicorn_listener.close()
                await uvicorn_listener.wait_closed()
        if client_task is not None and not client_task.done():
            client_task.cancel()
            await asyncio.gather(client_task, return_exceptions=True)
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        listener.close()

    await asyncio.sleep(0)
    assert set(asyncio.all_tasks()) == baseline


@pytest.mark.parametrize("_iteration", range(3))
@pytest.mark.parametrize("force_during_public_startup", [False, True])
async def test_force_during_uvicorn_startup_settles_created_lifespan_task(
    monkeypatch,
    _iteration,
    force_during_public_startup,
):
    baseline = set(asyncio.all_tasks())
    force = asyncio.Event()
    if not force_during_public_startup:
        force.set()

    class ForceOnServeManagedServer(server_module.ManagedServer):
        async def serve(self, sockets=None):
            asyncio.get_running_loop().call_soon(force.set)
            await super().serve(sockets=sockets)

    control_app = FastAPI()
    public_app = FastAPI()
    monkeypatch.setattr(server_module, "create_app", lambda runtime: public_app)
    monkeypatch.setattr(
        server_module,
        "create_control_app",
        lambda sessions, *, started_at: control_app,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    lease = FakeLease()
    lease.socket = listener
    shared_runtime = runtime()
    shared_runtime.settings.proxy_host = "127.0.0.1"
    shared_runtime.settings.proxy_port = 0
    servers = []

    def factory(config):
        server_type = server_module.ManagedServer
        if force_during_public_startup and servers:
            server_type = ForceOnServeManagedServer
        managed = server_type(config)
        servers.append(managed)
        return managed

    task = asyncio.create_task(
        server_module.serve_proxy(
            shared_runtime,
            lease,
            server_factory=factory,
            _force_event=force,
            _uvicorn_grace_timeout=0.1,
            _hard_timeout=0.5,
        )
    )
    try:
        await asyncio.wait_for(task, timeout=1)
    finally:
        force.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        lingering = {
            candidate
            for candidate in asyncio.all_tasks()
            if candidate is not asyncio.current_task()
            and (
                candidate.get_name() in {"control-server", "public-server"}
                or any(
                    getattr(candidate.get_coro(), "cr_frame", None) is not None
                    and candidate.get_coro().cr_frame.f_locals.get("self")
                    is getattr(managed, "lifespan", None)
                    for managed in servers
                )
            )
        }
        for pending in lingering:
            pending.cancel()
        await asyncio.gather(*lingering, return_exceptions=True)
        for managed in servers:
            for uvicorn_listener in getattr(managed, "servers", ()):
                uvicorn_listener.close()
                await uvicorn_listener.wait_closed()
        listener.close()

    assert lease.close_calls == 1
    assert all(not managed.server_state.tasks for managed in servers)
    assert not {
        "control-server",
        "public-server",
        "server-cleanup",
    } & {pending.get_name() for pending in asyncio.all_tasks()}
    await asyncio.sleep(0)
    assert set(asyncio.all_tasks()) == baseline


def test_managed_server_uses_inert_current_signal_capture_seam(monkeypatch):
    application = lambda scope, receive, send: None
    managed = server_module.ManagedServer(uvicorn.Config(application))

    def unexpected_signal_install(*args):
        pytest.fail("ManagedServer installed a signal handler")

    monkeypatch.setattr(signal, "signal", unexpected_signal_install)
    with managed.capture_signals():
        pass


async def test_second_central_signal_requests_force_and_restores_exact_handlers(
    monkeypatch,
):
    previous = {signal.SIGINT: object(), signal.SIGTERM: object()}
    current = previous.copy()

    def install(sig, handler):
        replaced = current[sig]
        current[sig] = handler
        return replaced

    monkeypatch.setattr(server_module.signal, "signal", install)

    async def capture_events(received_runtime, lease, **kwargs):
        stop = kwargs["stop_event"]
        force = kwargs["_force_event"]
        assert not stop.is_set() and not force.is_set()
        current[signal.SIGTERM](signal.SIGTERM, None)
        current[signal.SIGINT](signal.SIGINT, None)
        assert not stop.is_set() and not force.is_set()
        await asyncio.sleep(0)
        assert stop.is_set() and force.is_set()

    monkeypatch.setattr(server_module, "serve_proxy", capture_events)

    await server_module._run_proxy(runtime(), FakeLease())

    assert current == previous


async def test_signal_handlers_restore_identity_after_failure():
    def previous_int(sig, frame):
        return None

    def previous_term(sig, frame):
        return None

    original_int = signal.signal(signal.SIGINT, previous_int)
    original_term = signal.signal(signal.SIGTERM, previous_term)
    try:
        with pytest.raises(RuntimeError, match="serve failed"):
            with server_module._central_signal_handlers(
                asyncio.Event(),
                asyncio.Event(),
            ):
                raise RuntimeError("serve failed")

        assert signal.getsignal(signal.SIGINT) is previous_int
        assert signal.getsignal(signal.SIGTERM) is previous_term
    finally:
        signal.signal(signal.SIGINT, original_int)
        signal.signal(signal.SIGTERM, original_term)


async def test_central_handlers_preserve_existing_loop_signal_registration():
    loop = asyncio.get_running_loop()
    callback_ran = asyncio.Event()
    original = signal.getsignal(signal.SIGTERM)
    loop.add_signal_handler(signal.SIGTERM, callback_ran.set)
    prior_os_handler = signal.getsignal(signal.SIGTERM)
    registered = loop._signal_handlers[signal.SIGTERM]
    stop = asyncio.Event()
    force = asyncio.Event()

    try:
        with server_module._central_signal_handlers(stop, force):
            installed = signal.getsignal(signal.SIGTERM)
            installed(signal.SIGTERM, None)
            await asyncio.sleep(0)
            assert stop.is_set() and not force.is_set()

        assert signal.getsignal(signal.SIGTERM) is prior_os_handler
        assert loop._signal_handlers[signal.SIGTERM] is registered
        registered._run()
        assert callback_ran.is_set()
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, original)


def test_run_proxy_acquires_before_starting_event_loop(monkeypatch, tmp_path):
    lease = FakeLease()
    order = []

    def acquire(path):
        order.append(("acquire", path))
        return lease

    async def run(received_runtime, received_lease):
        order.append(("run", received_runtime, received_lease))

    monkeypatch.setattr(server_module.SocketLease, "acquire", acquire)
    monkeypatch.setattr(server_module, "_run_proxy", run)
    configured_runtime = runtime()
    socket_path = tmp_path / "control.sock"

    server_module.run_proxy(configured_runtime, socket_path)

    assert order == [
        ("acquire", socket_path),
        ("run", configured_runtime, lease),
    ]


def test_run_proxy_acquisition_failure_does_not_start_event_loop(monkeypatch):
    def fail_acquire(path):
        raise OSError("cannot acquire")

    monkeypatch.setattr(server_module.SocketLease, "acquire", fail_acquire)
    monkeypatch.setattr(
        server_module.asyncio,
        "run",
        lambda coroutine: pytest.fail("event loop started after acquire failure"),
    )

    with pytest.raises(OSError, match="cannot acquire"):
        server_module.run_proxy(runtime(), Path("/unused/control.sock"))


def test_run_proxy_treats_keyboard_interrupt_as_clean_exit_and_closes_lease(
    monkeypatch,
):
    lease = FakeLease()
    monkeypatch.setattr(server_module.SocketLease, "acquire", lambda path: lease)

    def interrupt(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module.asyncio, "run", interrupt)

    assert server_module.run_proxy(runtime(), Path("/unused/control.sock")) is None
    assert lease.close_calls == 1


def test_run_proxy_closes_lease_when_signal_handler_setup_fails(monkeypatch):
    lease = FakeLease()

    class HandlerSetupFailure:
        def __enter__(self):
            raise RuntimeError("cannot install signal handlers")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(server_module.SocketLease, "acquire", lambda path: lease)
    monkeypatch.setattr(
        server_module,
        "_central_signal_handlers",
        lambda stop, force: HandlerSetupFailure(),
    )

    with pytest.raises(RuntimeError, match="cannot install signal handlers"):
        server_module.run_proxy(runtime(), Path("/unused/control.sock"))

    assert lease.close_calls == 1


async def test_serve_proxy_leaves_no_tasks_behind(fake_apps):
    baseline = set(asyncio.all_tasks())
    control = FakeServer("control")
    public = FakeServer("public")
    stop = asyncio.Event()
    ready = asyncio.Event()
    task = asyncio.create_task(
        server_module.serve_proxy(
            runtime(),
            FakeLease(),
            stop_event=stop,
            server_factory=QueueFactory(control, public),
            _ready_event=ready,
        )
    )
    await ready.wait()
    stop.set()
    await task
    await asyncio.sleep(0)

    assert set(asyncio.all_tasks()) == baseline
