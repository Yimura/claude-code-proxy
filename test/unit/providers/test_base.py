import pytest

from claude_code_proxy.providers.base import ProviderError, stream_error_from_exception


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (402, "billing_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (409, "conflict_error"),
        (413, "request_too_large"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (503, "api_error"),
        (504, "timeout_error"),
        (529, "overloaded_error"),
    ],
)
def test_stream_error_maps_provider_status_to_anthropic_type(
    status_code, error_type
):
    error = ProviderError("provider detail", provider="fake", status_code=status_code)

    stream_error = stream_error_from_exception(
        error, provider="fake", expose_message=True
    )

    assert stream_error.error_type == error_type
    assert stream_error.message == "provider detail"
