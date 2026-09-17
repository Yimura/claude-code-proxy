# Anthropic API Proxy for Gemini, OpenAI, and Anthropic Models

Use Anthropic-compatible clients such as Claude Code with Gemini, OpenAI, Codex subscription, or direct Anthropic backends. The proxy translates requests through a provider-neutral service layer and LiteLLM or Codex adapters.

![Anthropic API Proxy](docs/assets/pic.png)

## Quick Start

### Prerequisites

Choose prerequisites for targets in your `model_mapping.json`:

- Python 3.14.x and [uv](https://docs.astral.sh/uv/) for source setup.
- Docker with Docker Compose for container setup.
- One or more provider credentials:
  - OpenAI API key for `openai/...` targets using LiteLLM.
  - Google AI Studio API key or Vertex AI Application Default Credentials for `gemini/...` targets.
  - Anthropic API key for `anthropic/...` targets.
  - OpenCode OAuth credentials for `openai/...` targets using Codex transport.

### Run from source

```bash
git clone https://github.com/Yimura/claude-code-proxy.git
cd claude-code-proxy
cp .env.example .env
uv sync --locked --dev
uv run uvicorn claude_code_proxy.app:app --app-dir src --host 0.0.0.0 --port 8082
```

Edit `.env` before starting. Configure only credentials required by targets in `model_mapping.json`.

### Run with Docker Compose

```bash
git clone https://github.com/Yimura/claude-code-proxy.git
cd claude-code-proxy
cp .env.example .env
docker compose up --build -d
```

The default Compose configuration publishes the proxy only on `127.0.0.1:8082`. Do not bind it to an external interface without adding authentication and network access controls; the proxy does not authenticate inbound requests.

Compose supplies repository `model_mapping.json` as a read-only application config at `/claude-code-proxy/model_mapping.json`. Edit the local file, then recreate the service to apply changes:

```bash
docker compose up --build -d --force-recreate
```

### Connect Claude Code

Point your existing [Claude Code installation](https://code.claude.com/docs/en/setup) at the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Environment Variables

Model selection belongs in `model_mapping.json`. Environment variables configure credentials, provider authentication, transport, and file locations.

### Variable reference

| Variable | Purpose | Required when | Default |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Authenticates direct Anthropic requests | A mapping resolves to `anthropic/...` | Unset |
| `OPENAI_API_KEY` | Authenticates OpenAI or compatible API requests through LiteLLM | A mapping resolves to `openai/...` and `OPENAI_TRANSPORT=litellm` | Unset |
| `OPENAI_BASE_URL` | Overrides OpenAI-compatible API endpoint | Using a compatible endpoint instead of OpenAI | LiteLLM/OpenAI default |
| `GEMINI_API_KEY` | Authenticates Gemini through Google AI Studio | A mapping resolves to `gemini/...` and Vertex auth is disabled | Unset |
| `USE_VERTEX_AUTH` | Uses Google Application Default Credentials for Gemini | Gemini should use Vertex AI instead of an API key | `false` |
| `VERTEX_PROJECT` | Selects Google Cloud project | `USE_VERTEX_AUTH=true` | `unset` |
| `VERTEX_LOCATION` | Selects Vertex AI region | `USE_VERTEX_AUTH=true` | `unset` |
| `OPENAI_TRANSPORT` | Routes `openai/...` targets through `litellm` or `codex` | Optional | `litellm` |
| `OPENCODE_DATA_DIR` | Locates OpenCode OAuth credentials used by Codex | `OPENAI_TRANSPORT=codex` | `~/.local/share/opencode` |
| `MODEL_MAPPING_PATH` | Selects model mapping JSON file | Mapping lives outside working directory | `model_mapping.json` |

### Conflicting choices

| Choose one | Do not combine with | Why |
|---|---|---|
| `GEMINI_API_KEY` authentication | `USE_VERTEX_AUTH=true` | Vertex uses Application Default Credentials instead of Google AI Studio key authentication. |
| `OPENAI_TRANSPORT=litellm` | `OPENAI_TRANSPORT=codex` | One transport handles all resolved `openai/...` targets for a process. |
| Exact `model` selector | `tier` selector in same mapping | Each mapping entry must select exactly one target source. |

### Settings that belong together

| Configuration | Variables/files | Requirement |
|---|---|---|
| Vertex AI | `USE_VERTEX_AUTH=true`, `VERTEX_PROJECT`, `VERTEX_LOCATION` | Set all three; authenticate with Google Application Default Credentials. |
| Codex subscription | `OPENAI_TRANSPORT=codex`, `OPENCODE_DATA_DIR` | Directory must contain usable OpenCode OAuth credentials. |
| Custom OpenAI-compatible endpoint | `OPENAI_TRANSPORT=litellm`, `OPENAI_BASE_URL`, usually `OPENAI_API_KEY` | Mapping targets must use `openai/...`. |
| Tier mapping | `model_mapping.json` `tiers` and mapping `tier` | Every referenced tier must exist in top-level `tiers`. |

## Model Mapping

`model_mapping.json` is the source of truth for upstream model definitions, context capabilities, tier selection, Claude-name matching, and default reasoning effort:

```json
{
  "models": {
    "terra": {
      "target": "openai/gpt-5.6-terra",
      "context_window": 1000000
    },
    "sol": {
      "target": "openai/gpt-5.6-sol",
      "context_window": 1000000
    }
  },
  "tiers": {
    "small": "terra",
    "big": "sol"
  },
  "mappings": {
    "haiku": {"tier": "small", "effort": "medium"},
    "sonnet": {"tier": "big", "effort": "medium"},
    "opus": {"tier": "big", "effort": "high"},
    "fable": {"model": "sol", "effort": "xhigh"}
  }
}
```

Every model definition requires:

- `target`: exact upstream provider model.
- `context_window`: positive token count, or `null` when capability is unknown.

Tier values reference model-definition names. Each mapping selects exactly one model definition:

- `"tier": "big"` resolves through `tiers.big`.
- `"model": "sol"` bypasses tiers and selects `models.sol` directly.

Raw provider targets are no longer valid in `tiers` or mapping `model` selectors. Move each target into `models`, then reference its definition name. Invalid and unknown references stop startup with a path-specific error.

Model mapping maintains three separate identities:

1. `original_model` preserves the exact inbound value for conservative prompt matching and logging.
2. `model` names the resolved upstream target used for provider routing, execution, and upstream system identity.
3. `response_model` advertises the client-facing Claude Code capability identity.

For an explicitly mapped model with a known context window, response identity is canonicalized from the inbound model. A context window of at least 1,000,000 tokens adds one terminal `[1m]` suffix; a smaller known window removes it. `context_window: null` advertises no capability and preserves the previous upstream response identity. Direct and unmapped requests preserve existing behavior.

Use explicit prefixes in model-definition targets to select providers:

- `openai/...` uses `OPENAI_TRANSPORT` (`litellm` by default, or `codex`).
- `gemini/...` uses LiteLLM with Google AI Studio or Vertex AI authentication.
- `anthropic/...` uses LiteLLM with Anthropic authentication.
- Unprefixed targets default to `openai/...`; explicit prefixes are recommended.

When Claude Code includes recognized model-identity metadata, mapped requests keep Claude Code identified as the coding-agent CLI harness while naming the exact resolved upstream target as the model generating the response. OpenAI and Gemini targets explicitly state that they are not Anthropic Claude models. Response identity does not change this upstream prompt identity. Unmapped requests, unrecognized metadata, and unrelated system instructions remain unchanged.

For direct Anthropic mappings, define and select Anthropic targets explicitly:

```json
{
  "models": {
    "haiku": {
      "target": "anthropic/claude-haiku-4-5-20251001",
      "context_window": 200000
    },
    "sonnet": {
      "target": "anthropic/claude-sonnet-5",
      "context_window": 1000000
    },
    "opus": {
      "target": "anthropic/claude-opus-5",
      "context_window": 1000000
    }
  },
  "tiers": {},
  "mappings": {
    "haiku": {"model": "haiku"},
    "sonnet": {"model": "sonnet"},
    "opus": {"model": "opus"}
  }
}
```

Valid mapping efforts are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. Request controls override mapping defaults in this order:

1. `output_config.effort`
2. Enabled, adaptive, or disabled `thinking`
3. Mapping `effort`
4. Provider default

Incoming effort `max` maps to provider effort `high` for broad compatibility. Enabled or adaptive thinking without a configured effort uses `medium`. Disabled thinking omits upstream reasoning configuration.

If the configured mapping file does not exist, the proxy warns and uses built-in defaults. An existing but invalid file stops startup with a path-specific validation error.

## Codex Agent orchestration

For `OPENAI_TRANSPORT=codex`, the Codex provider reinforces Claude Code's Agent lifecycle in the upstream system and tool descriptions:

- Agent completion is push-based and arrives through one automatic parent notification.
- Parents continue independent work after dispatch instead of polling.
- `TaskOutput` is not used for Agent tasks.
- `TaskOutput` remains available for non-Agent background tasks that require explicit retrieval and have no automatic completion notification.

This is provider-local guidance, not task-ID filtering or runtime interception. Tool names, input schemas, and capabilities remain unchanged, and LiteLLM, Gemini, and direct Anthropic requests are unaffected.

A billable opt-in evaluation exercises the behavior against a running proxy and real Codex model. It is skipped during normal test runs:

```bash
RUN_CODEX_AGENT_EVAL=1 \
ANTHROPIC_BASE_URL=http://127.0.0.1:8082 \
uv run pytest -q test/integration/test_codex_agent_polling.py
```

The evaluation simulates a running Agent, requires the parent to continue independent work, fails on `TaskOutput` polling, then delivers one synthetic completion notification. Because model behavior is nondeterministic, keep this evaluation outside required CI.

## How It Works

1. Receive an Anthropic-compatible request.
2. Resolve its model and reasoning policy through `model_mapping.json`.
3. Select LiteLLM or Codex transport.
4. Translate and send the upstream request.
5. Normalize streaming or non-streaming output into Anthropic-compatible responses.

## Development

Application code lives in `src/claude_code_proxy/`; unit tests live in `test/unit/`.

```bash
uv sync --locked --dev
uv run pytest
uv run uvicorn claude_code_proxy.app:app --app-dir src --reload --port 8082
```

Before refreshing dependencies, review and update direct constraints in `pyproject.toml`. Then regenerate and verify the lock for the supported Python series:

```bash
uv lock --python 3.14 --upgrade
uv sync --locked --dev
uv run pytest
```

Validate Compose wiring without requiring a populated `.env`:

```bash
docker compose config --no-env-resolution --quiet
```

## Contributing

Contributions are welcome. Open an issue or pull request with tests for behavior changes.
