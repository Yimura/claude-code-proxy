from pathlib import Path

ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"


def test_readme_documents_complete_performance_operator_contract() -> None:
    text = README.read_text(encoding="utf-8")

    required_snippets = (
        "`ps` is always available",
        "uv run claude-code-proxy proxy --performance collector",
        "uv run claude-code-proxy proxy --performance logging",
        "uv run claude-code-proxy perf --watch --format json",
        "default Compose service overrides that complete `CMD` with collector mode",
        'command: ["claude-code-proxy", "proxy", "--performance", "collector"]',
        'command: ["claude-code-proxy", "proxy", "--performance", "logging"]',
        "replaces the image's complete `CMD`",
        "latest 20 finalized requests",
        "4,096-event journal",
        "64-event subscriber queue",
        "does not reconnect",
        "first non-empty client-visible text, reasoning, or tool event",
        "prompts, messages, system instructions, tool names, tool descriptions",
        "encrypted reasoning",
        "raw client session, agent, or parent-agent IDs",
        "ASVS 5.0.0 V13.2.2",
        "V16.2.5",
        "V16.4.1",
        "Session Management Cheat Sheet",
        "No environment variable enables performance telemetry",
    )
    for snippet in required_snippets:
        assert snippet in text
