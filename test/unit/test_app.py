from claude_code_proxy.app import create_app
from claude_code_proxy.config import Settings


def test_create_app_installs_shared_request_logging(tmp_path):
    settings = Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        openai_base_url=None,
        gemini_api_key=None,
        use_vertex_auth=False,
        vertex_project=None,
        vertex_location=None,
        openai_transport="litellm",
        opencode_data_dir=tmp_path,
        model_mapping_path=tmp_path / "missing.json",
    )
    application = create_app(settings)
    assert application.user_middleware
