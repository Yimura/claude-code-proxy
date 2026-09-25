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
        "uv run claude-code-proxy tui",
        "docker compose exec proxy claude-code-proxy tui",
        "interactive TTY input and output",
        "↑/↓ or `j`/`k` select",
        "case-insensitive substring search",
        "Repeated filters for the same field are OR alternatives; different fields are ANDed",
        "`id` is a case-insensitive prefix match; all other public fields are case-insensitive exact matches",
        "baseline, session ID, state/phase, recency, model, elapsed, TTFT, input tokens, output tokens, cache ratio, or tool calls",
        "Unavailable sort values always remain last",
        "120 columns or wider",
        "90–119 columns",
        "narrower than 90 columns",
        "shorter than 22 rows",
        "Unavailable and not-applicable values render as `—`",
        "Partial aggregates render with `+?`",
        "0.5, 1, 2, and 4 seconds",
        "retains the last snapshot as stale",
        "exits nonzero after retry exhaustion",
        "masked while entered, sent only to the local control endpoint",
        "TUI state is ephemeral",
        "durable console logs remain the operational record",
        "provider request or response payloads",
    )
    for snippet in required_snippets:
        assert snippet in text


def test_readme_documents_ps_watch_output_contract() -> None:
    text = README.read_text(encoding="utf-8")

    required_snippets = (
        "uv run claude-code-proxy ps --watch",
        "uv run claude-code-proxy ps --watch --format json",
        "same normal `ps` table immediately",
        "refreshes it in place once per second",
        "requires TTY standard output",
        "only rows owned by the previous frame",
        "does not clear or take over the whole terminal",
        "works through pipes and non-TTY output",
        "JSON Lines",
        "one compact JSON array per line",
        "complete session snapshot with the same fields and ordering as one-shot JSON",
        "`perf --watch` is an append-only performance event stream, not repeated session snapshots",
    )
    for snippet in required_snippets:
        assert snippet in text
