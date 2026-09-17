import importlib
import sys

import pytest

import claude_code_proxy.app as app_module
import claude_code_proxy.runtime as runtime_module
from claude_code_proxy.config import Settings


def settings(tmp_path, transport="litellm"):
    return Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        openai_base_url=None,
        gemini_api_key=None,
        use_vertex_auth=False,
        vertex_project=None,
        vertex_location=None,
        openai_transport=transport,
        opencode_data_dir=tmp_path,
        model_mapping_path=tmp_path / "missing.json",
    )


class FakeAuth:
    instances = []
    failure = None

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.initialize_calls = 0
        self.instances.append(self)

    async def initialize(self):
        self.initialize_calls += 1
        if self.failure is not None:
            raise self.failure

    async def get_auth(self):
        return "access", "account"

    async def recover_rejected(self, access_token):
        return "access", "account"


@pytest.fixture(autouse=True)
def reset_fake_auth():
    FakeAuth.instances = []
    FakeAuth.failure = None


def test_create_app_stores_runtime_and_installs_shared_request_logging(tmp_path):
    runtime = runtime_module.create_runtime(settings(tmp_path))

    application = app_module.create_app(runtime)

    assert application.state.runtime is runtime
    assert application.user_middleware


async def test_codex_transport_initializes_auth_once_during_startup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime_module, "CodexAuth", FakeAuth)
    runtime = runtime_module.create_runtime(settings(tmp_path, "codex"))
    application = app_module.create_app(runtime)

    async with application.router.lifespan_context(application):
        assert runtime.codex_auth.initialize_calls == 1

    assert runtime.codex_auth.initialize_calls == 1


async def test_litellm_transport_does_not_initialize_codex_auth(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime_module, "CodexAuth", FakeAuth)
    runtime = runtime_module.create_runtime(settings(tmp_path, "litellm"))
    application = app_module.create_app(runtime)

    async with application.router.lifespan_context(application):
        assert runtime.codex_auth.initialize_calls == 0


async def test_codex_auth_failure_aborts_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "CodexAuth", FakeAuth)
    FakeAuth.failure = RuntimeError("credentials unavailable")
    runtime = runtime_module.create_runtime(settings(tmp_path, "codex"))
    application = app_module.create_app(runtime)

    with pytest.raises(RuntimeError, match="credentials unavailable"):
        async with application.router.lifespan_context(application):
            pytest.fail("startup must not enter application lifespan")

    assert runtime.codex_auth.initialize_calls == 1


def test_importing_app_does_not_construct_runtime_or_expose_application(
    monkeypatch,
):
    def unexpected_runtime_construction():
        pytest.fail("importing claude_code_proxy.app constructed a runtime")

    monkeypatch.setattr(
        runtime_module,
        "create_runtime",
        unexpected_runtime_construction,
    )
    package = sys.modules["claude_code_proxy"]
    sys.modules.pop("claude_code_proxy.app")
    try:
        imported = importlib.import_module("claude_code_proxy.app")
    finally:
        sys.modules["claude_code_proxy.app"] = app_module
        package.app = app_module

    assert not hasattr(imported, "app")
