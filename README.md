# Anthropic API Proxy for Gemini, OpenAI, and Anthropic Models

Use Anthropic-compatible clients such as Claude Code with Gemini, OpenAI, Codex subscription, or direct Anthropic backends. The proxy translates requests through a provider-neutral service layer and LiteLLM or Codex adapters.

![Anthropic API Proxy](docs/assets/pic.png)

## Quick Start

### Prerequisites

Choose prerequisites for targets in your `model_mapping.json`:

- Python 3.10+ and [uv](https://docs.astral.sh/uv/) for source setup.
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
uv sync --dev
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

Compose supplies repository `model_mapping.json` as a read-only application config at `/claude-code-proxy/model_mapping.json`. Edit the local file, then recreate the service to apply changes:

```bash
docker compose up --build -d --force-recreate
```

### Connect Claude Code

Install Claude Code, then point it at the proxy:

```bash
npm install -g @anthropic-ai/claude-code
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

`model_mapping.json` is the source of truth for Claude-name matching, upstream model selection, and default reasoning effort:

```json
{
  "tiers": {
    "small": "openai/gpt-5.6-terra",
    "big": "openai/gpt-5.6-sol"
  },
  "mappings": {
    "haiku": {"tier": "small", "effort": "medium"},
    "sonnet": {"tier": "big", "effort": "medium"},
    "opus": {"tier": "big", "effort": "high"},
    "fable": {"model": "openai/gpt-daybreak-blue-latest", "effort": "high"}
  }
}
```

Mapping patterns are matched case-insensitively against incoming model names. Each mapping selects exactly one target:

- `"tier": "small"` resolves through top-level `tiers.small`.
- `"tier": "big"` resolves through top-level `tiers.big`.
- `"model": "..."` selects an exact target.

Use explicit prefixes to select providers:

- `openai/...` uses `OPENAI_TRANSPORT` (`litellm` by default, or `codex`).
- `gemini/...` uses LiteLLM with Google AI Studio or Vertex AI authentication.
- `anthropic/...` uses LiteLLM with Anthropic authentication.
- Unprefixed mapping targets default to `openai/...`; explicit prefixes are recommended.

For direct Anthropic mappings, select Anthropic targets explicitly:

```json
{
  "tiers": {},
  "mappings": {
    "haiku": {"model": "anthropic/claude-haiku-4-5-20251001"},
    "sonnet": {"model": "anthropic/claude-sonnet-5"},
    "opus": {"model": "anthropic/claude-opus-5"}
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

## How It Works

1. Receive an Anthropic-compatible request.
2. Resolve its model and reasoning policy through `model_mapping.json`.
3. Select LiteLLM or Codex transport.
4. Translate and send the upstream request.
5. Normalize streaming or non-streaming output into Anthropic-compatible responses.

## Development

Application code lives in `src/claude_code_proxy/`; unit tests live in `test/unit/`.

```bash
uv sync --dev
uv run pytest
uv run uvicorn claude_code_proxy.app:app --app-dir src --reload --port 8082
```

Validate Compose wiring without requiring a populated `.env`:

```bash
docker compose config --no-env-resolution --quiet
```

## Contributing

Contributions are welcome. Open an issue or pull request with tests for behavior changes.
