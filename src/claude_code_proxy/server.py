"""Coordinated lifecycle for the public and local control servers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import signal
from pathlib import Path
from typing import Protocol

import uvicorn

from .app import create_app
from .control.app import create_control_app
from .control.socket import SocketLease
from .runtime import RuntimeServices

UVICORN_GRACEFUL_TIMEOUT_SECONDS = 10.0
COORDINATOR_HARD_TIMEOUT_SECONDS = 12.0
_HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class _Server(Protocol):
    """Lifecycle surface used by the coordinator."""

    started: bool
    should_exit: bool
    force_exit: bool

    async def serve(self, sockets=None) -> None: ...


ServerFactory = Callable[[uvicorn.Config], _Server]


class _ServerTaskFailure(Exception):
    """Keep task failures within Exception while retaining their original cause."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


@dataclass
class _LifecycleEvidence:
    wrapper: asyncio.Task[None] | None
    discovered_lifespan_tasks: set[asyncio.Task] = field(default_factory=set)
    disposed_lifespan_tasks: set[asyncio.Task] = field(default_factory=set)
    wrapper_settled: bool = False


class ManagedServer(uvicorn.Server):
    """Uvicorn server whose signals are managed by the process coordinator."""

    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        """Leave process signal handlers untouched."""
        yield


async def serve_proxy(
    runtime: RuntimeServices,
    lease: SocketLease,
    *,
    stop_event: asyncio.Event | None = None,
    server_factory: ServerFactory = ManagedServer,
    _ready_event: asyncio.Event | None = None,
    _force_event: asyncio.Event | None = None,
    _uvicorn_grace_timeout: float = UVICORN_GRACEFUL_TIMEOUT_SECONDS,
    _hard_timeout: float = COORDINATOR_HARD_TIMEOUT_SECONDS,
) -> None:
    """Serve the control endpoint first, then the public proxy endpoint."""
    timeout_error: ValueError | None = None
    if not math.isfinite(_uvicorn_grace_timeout) or _uvicorn_grace_timeout < 0:
        timeout_error = ValueError(
            "Uvicorn graceful shutdown timeout must be finite and non-negative"
        )
    elif not math.isfinite(_hard_timeout) or _hard_timeout <= _uvicorn_grace_timeout:
        timeout_error = ValueError(
            "Coordinator hard shutdown timeout must be finite and greater than "
            "the Uvicorn graceful shutdown timeout"
        )
    if timeout_error is not None:
        lease_error = _close_lease(lease)
        if lease_error is not None:
            timeout_error.add_note(f"lease cleanup failure: {lease_error!r}")
        raise timeout_error

    stop = stop_event or asyncio.Event()
    force = _force_event or asyncio.Event()
    servers: list[_Server] = []
    tasks: list[asyncio.Task[None]] = []
    failure_message: str | None = None
    direct_errors: list[BaseException] = []
    cancellation: asyncio.CancelledError | None = None

    try:
        control = _create_control_server(
            runtime,
            server_factory,
            _uvicorn_grace_timeout,
        )
        servers.append(control)
        control_task = asyncio.create_task(
            _serve_server(control, sockets=[lease.socket]),
            name="control-server",
        )
        tasks.append(control_task)

        control_state = await _wait_for_start(control, control_task, stop, force)
        if control_state == "exited" or control_task.done():
            failure_message = "control server failed during startup"
        elif control_state == "started" and not (stop.is_set() or force.is_set()):
            public = _create_public_server(
                runtime,
                server_factory,
                _uvicorn_grace_timeout,
            )
            servers.append(public)
            if control_task.done():
                failure_message = "control server failed during startup"
            elif not (stop.is_set() or force.is_set()):
                public_task = asyncio.create_task(
                    _serve_server(public),
                    name="public-server",
                )
                tasks.append(public_task)
                public_state = await _wait_for_start(
                    public,
                    public_task,
                    stop,
                    force,
                )
                if public_state == "exited" or public_task.done():
                    failure_message = "public server failed during startup"
                elif control_task.done():
                    failure_message = "control server failed during startup"
                elif public_state == "started" and not (
                    stop.is_set() or force.is_set()
                ):
                    if _ready_event is not None:
                        _ready_event.set()
                    failure_message = await _wait_for_stop_or_completion(
                        control_task,
                        public_task,
                        stop,
                        force,
                    )
    except asyncio.CancelledError as error:
        cancellation = error
    except BaseException as error:
        failure_message = failure_message or "server coordination failed"
        direct_errors.append(error)
    finally:
        cleanup_cancellations: list[asyncio.CancelledError] = []
        try:
            cleanup_errors, cleanup_cancellations = await _finish_server_cleanup(
                servers,
                tasks,
                force,
                _hard_timeout,
            )
        finally:
            lease_error = _close_lease(lease)

    task_errors = _task_errors(tasks)
    errors = _unique_errors([*direct_errors, *task_errors, *cleanup_errors])
    if lease_error is not None:
        errors.append(lease_error)
    if cancellation is None and cleanup_cancellations:
        if failure_message is None and not direct_errors and not task_errors:
            cancellation = cleanup_cancellations[0]
        else:
            errors = _unique_errors([*errors, *cleanup_cancellations])

    if cancellation is not None:
        _annotate_cleanup_failures(cancellation, errors)
        raise cancellation
    if failure_message is not None:
        _raise_server_failure(failure_message, errors)
    if task_errors:
        _raise_server_failure("server failed during shutdown", errors)
    if cleanup_errors:
        _raise_server_failure("server cleanup failed", errors)
    if lease_error is not None:
        raise RuntimeError("failed to close control socket lease") from lease_error


def _create_control_server(
    runtime: RuntimeServices,
    server_factory: ServerFactory,
    graceful_timeout: float,
) -> _Server:
    application = create_control_app(
        runtime.sessions,
        started_at=runtime.started_at,
    )
    config = _server_config(application, graceful_timeout)
    return server_factory(config)


def _create_public_server(
    runtime: RuntimeServices,
    server_factory: ServerFactory,
    graceful_timeout: float,
) -> _Server:
    application = create_app(runtime)
    config = _server_config(
        application,
        graceful_timeout,
        host=runtime.settings.proxy_host,
        port=runtime.settings.proxy_port,
    )
    return server_factory(config)


def _server_config(
    application,
    graceful_timeout: float,
    **bind,
) -> uvicorn.Config:
    return uvicorn.Config(
        application,
        lifespan="on",
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=graceful_timeout,
        **bind,
    )


async def _serve_server(_server: _Server, sockets=None) -> None:
    try:
        await _server.serve(sockets=sockets)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise
    except BaseException as error:
        raise _ServerTaskFailure(error) from error


async def _wait_for_start(
    server: _Server,
    task: asyncio.Task[None],
    stop: asyncio.Event,
    force: asyncio.Event,
) -> str:
    while True:
        if task.done():
            return "exited"
        if stop.is_set() or force.is_set():
            return "stopped"
        if server.started:
            return "started"
        await asyncio.sleep(0)


async def _wait_for_stop_or_completion(
    control_task: asyncio.Task[None],
    public_task: asyncio.Task[None],
    stop: asyncio.Event,
    force: asyncio.Event,
) -> str | None:
    stop_waiter = asyncio.create_task(stop.wait(), name="proxy-stop-waiter")
    force_waiter = asyncio.create_task(force.wait(), name="proxy-force-waiter")
    try:
        done, _ = await asyncio.wait(
            {control_task, public_task, stop_waiter, force_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if control_task in done:
            return "control server exited unexpectedly"
        if public_task in done:
            return "public server exited unexpectedly"
        return None
    finally:
        await _cancel_waiters(stop_waiter, force_waiter)


async def _finish_server_cleanup(
    servers: list[_Server],
    tasks: list[asyncio.Task[None]],
    force: asyncio.Event,
    grace_timeout: float,
) -> tuple[list[BaseException], list[asyncio.CancelledError]]:
    async def collect_errors() -> list[BaseException]:
        try:
            return await _stop_server_tasks(
                servers,
                tasks,
                force,
                grace_timeout,
            )
        except BaseException as error:
            return [error]

    cleanup_task = asyncio.create_task(
        collect_errors(),
        name="server-cleanup",
    )
    cancellations: list[asyncio.CancelledError] = []
    while True:
        try:
            errors = await asyncio.shield(cleanup_task)
            return errors, cancellations
        except asyncio.CancelledError as error:
            cancellations.append(error)
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()


def _remaining_time(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


def _uvicorn_owned_tasks(
    server: _Server,
    evidence: _LifecycleEvidence | None = None,
    *,
    defer_missing_disposal: bool = False,
) -> set[asyncio.Task]:
    """Snapshot request and lifespan tasks using Uvicorn 0.53's task shape."""
    state = getattr(server, "server_state", None)
    if state is None:
        return set()
    raw_tasks = getattr(state, "tasks", None)
    if raw_tasks is None:
        raise RuntimeError("Unsupported Uvicorn server_state task API")

    owned = set(raw_tasks)
    if not all(isinstance(task, asyncio.Task) for task in owned):
        raise RuntimeError("Uvicorn server_state.tasks contains a non-Task value")
    if not isinstance(server, ManagedServer):
        return _without_current_task(owned)

    lifespan = getattr(server, "lifespan", None)
    if lifespan is None:
        if server.started:
            raise RuntimeError("Started Uvicorn server has no lifespan instance")
        return _without_current_task(owned)

    lifespan_tasks = _tasks_owned_by_coroutine_instance(lifespan)
    if evidence is not None:
        evidence.discovered_lifespan_tasks.update(lifespan_tasks)
    shutdown_event = getattr(lifespan, "shutdown_event", None)
    if not lifespan_tasks and (
        shutdown_event is None or not shutdown_event.is_set()
    ):
        deliberately_disposed = (
            evidence is not None
            and bool(evidence.disposed_lifespan_tasks)
            and all(task.done() for task in evidence.disposed_lifespan_tasks)
        )
        if not deliberately_disposed:
            raise RuntimeError("Unsupported Uvicorn lifespan task API")
        if not evidence.wrapper_settled and not defer_missing_disposal:
            raise RuntimeError("Uvicorn lifespan disappeared before wrapper settlement")
    return _without_current_task(owned | lifespan_tasks)


def _tasks_owned_by_coroutine_instance(owner: object) -> set[asyncio.Task]:
    matched: set[asyncio.Task] = set()
    for task in asyncio.all_tasks():
        coroutine = task.get_coro()
        frame = getattr(coroutine, "cr_frame", None)
        if frame is not None and frame.f_locals.get("self") is owner:
            matched.add(task)
    return matched


def _without_current_task(tasks: set[asyncio.Task]) -> set[asyncio.Task]:
    current = asyncio.current_task()
    return {task for task in tasks if task is not current}


async def _force_stop_owned_work(
    servers: list[_Server],
    wrapper_tasks: list[asyncio.Task[None]],
    deadline: float,
    lifecycle_evidence: dict[int, _LifecycleEvidence],
) -> list[BaseException]:
    """Force closure, cancelling lifespan tasks instead of running shutdown hooks."""
    errors = await _close_uvicorn_listeners(servers, deadline)
    child_errors, _ = await _settle_server_owned_tasks(
        servers,
        deadline,
        lifecycle_evidence,
        cancel_first=True,
        defer_missing_disposal=True,
    )
    errors.extend(child_errors)
    wrapper_errors, _ = await _settle_tasks(
        set(wrapper_tasks),
        deadline,
        cancel_first=True,
        label="server wrapper tasks",
    )
    errors.extend(wrapper_errors)
    for evidence in lifecycle_evidence.values():
        if evidence.wrapper is not None and evidence.wrapper.done():
            evidence.wrapper_settled = True
    final_child_errors, _ = await _settle_server_owned_tasks(
        servers,
        deadline,
        lifecycle_evidence,
        cancel_first=True,
    )
    for error in final_child_errors:
        if not any(
            type(existing) is type(error) and str(existing) == str(error)
            for existing in errors
        ):
            errors.append(error)
    return _unique_errors(errors)


async def _settle_graceful_owned_work(
    servers: list[_Server],
    deadline: float,
    lifecycle_evidence: dict[int, _LifecycleEvidence],
) -> list[BaseException]:
    wrapper_errors: list[BaseException] = []
    for evidence in lifecycle_evidence.values():
        if evidence.wrapper is not None and evidence.wrapper.done():
            wrapper_errors.extend(_completed_task_errors({evidence.wrapper}))
            evidence.wrapper_settled = True
    errors, pending = await _settle_server_owned_tasks(
        servers,
        deadline,
        lifecycle_evidence,
        cancel_first=False,
    )
    if not pending:
        return _unique_errors([*wrapper_errors, *errors])

    for server in servers:
        server.force_exit = True
    forced_errors, _ = await _settle_server_owned_tasks(
        servers,
        deadline,
        lifecycle_evidence,
        cancel_first=True,
    )
    return _unique_errors([*wrapper_errors, *errors, *forced_errors])


async def _close_uvicorn_listeners(
    servers: list[_Server],
    deadline: float,
) -> list[BaseException]:
    shutdown_tasks: set[asyncio.Task] = set()
    errors: list[BaseException] = []
    for index, server in enumerate(servers):
        if not isinstance(server, ManagedServer) or not server.started:
            continue
        shutdown = getattr(server, "shutdown", None)
        listeners = getattr(server, "servers", None)
        if not callable(shutdown) or listeners is None:
            errors.append(RuntimeError("Unsupported Uvicorn shutdown API"))
            continue
        shutdown_tasks.add(asyncio.create_task(
            shutdown(),
            name=f"uvicorn-force-shutdown-{index}",
        ))

    shutdown_errors, pending = await _settle_tasks(
        shutdown_tasks,
        deadline,
        cancel_first=False,
        label="Uvicorn shutdown tasks",
    )
    errors.extend(shutdown_errors)
    if pending:
        forced_errors, _ = await _settle_tasks(
            pending,
            deadline,
            cancel_first=True,
            label="Uvicorn shutdown tasks",
        )
        errors.extend(forced_errors)
    return errors


async def _settle_server_owned_tasks(
    servers: list[_Server],
    deadline: float,
    lifecycle_evidence: dict[int, _LifecycleEvidence],
    *,
    cancel_first: bool,
    defer_missing_disposal: bool = False,
) -> tuple[list[BaseException], set[asyncio.Task]]:
    errors: list[BaseException] = []
    for _ in range(3):
        owned: set[asyncio.Task] = set()
        for server in servers:
            try:
                owned.update(_uvicorn_owned_tasks(
                    server,
                    lifecycle_evidence.get(id(server)),
                    defer_missing_disposal=defer_missing_disposal,
                ))
            except BaseException as error:
                errors.append(error)
        live = {task for task in owned if not task.done()}
        errors.extend(_completed_task_errors(owned - live))
        if not live:
            await asyncio.sleep(0)
            return errors, set()
        disposal_attempts: dict[int, set[asyncio.Task]] = {}
        if cancel_first:
            disposal_attempts = {
                key: evidence.discovered_lifespan_tasks & live
                for key, evidence in lifecycle_evidence.items()
            }
        coordinator_cancelled: set[asyncio.Task] = set()
        task_errors, pending = await _settle_tasks(
            live,
            deadline,
            cancel_first=cancel_first,
            label="Uvicorn-owned tasks",
            cancelled_tasks=coordinator_cancelled,
        )
        errors.extend(task_errors)
        settled = live - pending
        for key, attempted in disposal_attempts.items():
            lifecycle_evidence[key].disposed_lifespan_tasks.update(
                attempted & settled & coordinator_cancelled
            )
        if pending:
            return errors, pending
        await asyncio.sleep(0)

    errors.append(RuntimeError("Uvicorn-owned tasks kept appearing during cleanup"))
    return errors, set()


async def _settle_tasks(
    tasks: set[asyncio.Task],
    deadline: float,
    *,
    cancel_first: bool,
    label: str,
    cancelled_tasks: set[asyncio.Task] | None = None,
) -> tuple[list[BaseException], set[asyncio.Task]]:
    current = asyncio.current_task()
    tracked = {task for task in tasks if task is not current}
    if cancel_first:
        for task in tracked:
            if not task.done() and task.cancel():
                if cancelled_tasks is not None:
                    cancelled_tasks.add(task)

    pending = {task for task in tracked if not task.done()}
    if pending and _remaining_time(deadline) > 0:
        _, pending = await asyncio.wait(
            pending,
            timeout=_remaining_time(deadline),
        )
    elif pending:
        await asyncio.sleep(0)
        pending = {task for task in pending if not task.done()}

    if pending and cancel_first:
        for task in pending:
            if task.cancel() and cancelled_tasks is not None:
                cancelled_tasks.add(task)
            task.add_done_callback(_consume_late_task_exception)
        await asyncio.sleep(0)
        pending = {task for task in pending if not task.done()}

    errors = _completed_task_errors(tracked - pending)
    if pending and cancel_first:
        errors.append(RuntimeError(
            f"{len(pending)} {label} resisted cancellation before hard deadline"
        ))
    return errors, pending


def _completed_task_errors(tasks: set[asyncio.Task]) -> list[BaseException]:
    errors: list[BaseException] = []
    for task in tasks:
        if not task.done() or task.cancelled():
            continue
        error = task.exception()
        if isinstance(error, _ServerTaskFailure):
            errors.append(error.cause)
        elif error is not None:
            errors.append(error)
    return errors


def _consume_late_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


async def _stop_server_tasks(
    servers: list[_Server],
    tasks: list[asyncio.Task[None]],
    force: asyncio.Event,
    grace_timeout: float,
) -> list[BaseException]:
    deadline = asyncio.get_running_loop().time() + max(0.0, grace_timeout)
    lifecycle_evidence = {
        id(server): _LifecycleEvidence(
            tasks[index] if index < len(tasks) else None
        )
        for index, server in enumerate(servers)
    }
    for server in servers:
        server.should_exit = True

    pending = [task for task in tasks if not task.done()]
    if pending and not force.is_set():
        all_done = asyncio.gather(*pending, return_exceptions=True)
        force_waiter = asyncio.create_task(
            force.wait(),
            name="shutdown-force-waiter",
        )
        try:
            await asyncio.wait(
                {all_done, force_waiter},
                timeout=_remaining_time(deadline),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            await _cancel_waiters(force_waiter)

    remaining = [task for task in tasks if not task.done()]
    errors: list[BaseException] = []
    if force.is_set() or remaining:
        for server in servers:
            server.force_exit = True
        errors.extend(await _force_stop_owned_work(
            servers,
            tasks,
            deadline,
            lifecycle_evidence,
        ))
    else:
        errors.extend(await _settle_graceful_owned_work(
            servers,
            deadline,
            lifecycle_evidence,
        ))

    return _unique_errors([*errors, *_task_errors(tasks)])


async def _cancel_waiters(*tasks: asyncio.Task[object]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _task_errors(tasks: list[asyncio.Task[None]]) -> list[BaseException]:
    errors: list[BaseException] = []
    for task in tasks:
        if not task.done() or task.cancelled():
            continue
        error = task.exception()
        if isinstance(error, _ServerTaskFailure):
            errors.append(error.cause)
        elif error is not None:
            errors.append(error)
    return errors


def _close_lease(lease: SocketLease) -> BaseException | None:
    try:
        lease.close()
    except BaseException as error:
        return error
    return None


def _unique_errors(errors: list[BaseException]) -> list[BaseException]:
    unique: list[BaseException] = []
    seen: set[int] = set()
    for error in errors:
        if id(error) in seen:
            continue
        seen.add(id(error))
        unique.append(error)
    return unique


def _raise_server_failure(
    message: str,
    errors: list[BaseException],
) -> None:
    if not errors:
        raise RuntimeError(message)
    if len(errors) == 1:
        raise RuntimeError(message) from errors[0]
    raise RuntimeError(message) from BaseExceptionGroup(
        "multiple server failures",
        errors,
    )


def _annotate_cleanup_failures(
    cancellation: asyncio.CancelledError,
    errors: list[BaseException],
) -> None:
    for error in errors:
        cancellation.add_note(f"cleanup failure: {error!r}")


async def _run_proxy(runtime: RuntimeServices, lease: SocketLease) -> None:
    stop = asyncio.Event()
    force = asyncio.Event()
    with _central_signal_handlers(stop, force):
        await serve_proxy(
            runtime,
            lease,
            stop_event=stop,
            _force_event=force,
        )


@contextmanager
def _central_signal_handlers(
    stop: asyncio.Event,
    force: asyncio.Event,
) -> Generator[None, None, None]:
    loop = asyncio.get_running_loop()
    previous: dict[signal.Signals, object] = {}

    def request_stop() -> None:
        if stop.is_set():
            force.set()
            return
        stop.set()

    def handle_signal(sig, frame) -> None:
        loop.call_soon_threadsafe(request_stop)

    try:
        for sig in _HANDLED_SIGNALS:
            previous[sig] = signal.signal(sig, handle_signal)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_proxy(runtime: RuntimeServices, socket_path: Path) -> None:
    """Acquire the control socket and run both servers until interrupted."""
    lease = SocketLease.acquire(socket_path)
    try:
        try:
            asyncio.run(_run_proxy(runtime, lease))
        except KeyboardInterrupt:
            return
    finally:
        lease.close()
