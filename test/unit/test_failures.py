from dataclasses import FrozenInstanceError
import json
import re

import pytest

import claude_code_proxy.failures as failures_module
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)


def test_failure_categories_have_stable_string_values():
    assert {category.name: category.value for category in FailureCategory} == {
        "AUTHENTICATION": "authentication",
        "TRANSPORT": "transport",
        "UPSTREAM_HTTP": "upstream_http",
        "PROVIDER_PROTOCOL": "provider_protocol",
        "TRANSLATION": "translation",
        "INTERNAL": "internal",
    }


def test_failure_stages_have_stable_string_values():
    assert {stage.name: stage.value for stage in FailureStage} == {
        "CREDENTIALS": "credentials",
        "REQUEST": "request",
        "RESPONSE": "response",
        "STREAM": "stream",
        "PROVIDER_TRANSLATION": "provider_translation",
        "CLIENT_TRANSLATION": "client_translation",
        "ROUTE": "route",
    }


def test_failure_diagnostic_defaults_provider_code_to_none_and_is_immutable():
    diagnostic = FailureDiagnostic(
        category=FailureCategory.TRANSPORT,
        stage=FailureStage.REQUEST,
        code="connection_failed",
    )

    assert diagnostic.provider_code is None
    assert diagnostic.exception_type is None
    assert diagnostic.location is None
    with pytest.raises(FrozenInstanceError):
        diagnostic.code = "changed"


def test_failure_diagnostic_uses_slots_for_fixed_schema():
    diagnostic = FailureDiagnostic(
        category=FailureCategory.INTERNAL,
        stage=FailureStage.ROUTE,
        code="unexpected",
    )

    assert not hasattr(diagnostic, "__dict__")


def test_safe_exception_location_prefers_innermost_application_frame():
    namespace = {
        "__name__": "claude_code_proxy.synthetic",
        "json": json,
    }
    exec(
        "def parse_invalid_json():\n"
        "    local_secret = 'LOCAL_SECRET_MUST_NOT_LEAK'\n"
        "    return json.loads('not-json')\n",
        namespace,
    )

    try:
        namespace["parse_invalid_json"]()
    except json.JSONDecodeError as error:
        location = failures_module.safe_exception_location(error)

    assert re.fullmatch(
        r"claude_code_proxy\.synthetic:parse_invalid_json:\d+", location
    )
    assert "json.decoder" not in location
    assert "LOCAL_SECRET_MUST_NOT_LEAK" not in location
    assert "/" not in location


def test_safe_exception_location_without_application_frame_is_stable():
    try:
        json.loads("not-json")
    except json.JSONDecodeError as error:
        location = failures_module.safe_exception_location(error)

    assert location == "unknown:unknown:0"


def test_unexpected_failure_diagnostic_preserves_requested_shape_and_evidence():
    namespace = {"__name__": "claude_code_proxy.synthetic"}
    exec(
        "def fail_request():\n"
        "    secret = 'SECRET_MUST_NOT_LEAK'\n"
        "    raise RuntimeError('unsafe message')\n",
        namespace,
    )

    try:
        namespace["fail_request"]()
    except RuntimeError as error:
        diagnostic = failures_module.unexpected_failure_diagnostic(
            error,
            stage=FailureStage.PROVIDER_TRANSLATION,
            code="response_translation_failed",
            category=FailureCategory.TRANSLATION,
        )

    assert diagnostic.category == FailureCategory.TRANSLATION
    assert diagnostic.stage == FailureStage.PROVIDER_TRANSLATION
    assert diagnostic.code == "response_translation_failed"
    assert diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.synthetic:fail_request:\d+",
        diagnostic.location,
    )
    assert "unsafe message" not in repr(diagnostic)
    assert "SECRET_MUST_NOT_LEAK" not in repr(diagnostic)


def test_retryable_status_treats_missing_status_as_retryable():
    assert failures_module.retryable_status(None) is True


def test_safe_exception_location_chooses_innermost_app_across_external_frame():
    inner_namespace = {
        "__name__": "claude_code_proxy.inner",
        "json": json,
    }
    exec(
        "def inner_app():\n"
        "    return json.loads('not-json')\n",
        inner_namespace,
    )
    external_namespace = {
        "__name__": "third_party.wrapper",
        "inner_app": inner_namespace["inner_app"],
    }
    exec(
        "def external_wrapper():\n"
        "    return inner_app()\n",
        external_namespace,
    )
    outer_namespace = {
        "__name__": "claude_code_proxy.outer",
        "external_wrapper": external_namespace["external_wrapper"],
    }
    exec(
        "def outer_app():\n"
        "    return external_wrapper()\n",
        outer_namespace,
    )

    try:
        outer_namespace["outer_app"]()
    except json.JSONDecodeError as error:
        location = failures_module.safe_exception_location(error)

    assert re.fullmatch(
        r"claude_code_proxy\.inner:inner_app:\d+", location
    )
