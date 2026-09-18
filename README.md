# Anthropic API Proxy for Gemini, OpenAI, and Anthropic Models

Use Anthropic-compatible clients such as Claude Code with Gemini, OpenAI, Codex subscription, or direct Anthropic backends. The proxy translates requests through a provider-neutral service layer and LiteLLM or Codex adapters.

![Anthropic API Proxy](docs/assets/pic.png)

## Quick Start

### Prerequisites

Choose prerequisites for targets in your `model_mapping.json`:

- Linux, Python 3.14.x, and [uv](https://docs.astral.sh/uv/) for source setup. Source `proxy` and host-side `ps` use a Linux-only secure Unix-socket backend.
- Docker with Docker Compose for container setup. Linux containers remain supported on macOS and Windows hosts.
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
uv run claude-code-proxy proxy
```

Source `proxy` and host-side `ps` are supported on Linux only. On macOS or Windows, use the Docker setup below and run `ps` inside the Linux container.

Edit `.env` before starting. Configure only credentials required by targets in `model_mapping.json`. The `proxy` and `ps` commands read `.env` from the current working directory; already-exported environment variables win, and command options override both. The example binds source deployments to `127.0.0.1` because the public API does not authenticate inbound requests. The proxy command runs in the foreground.

### Run with Docker Compose

```bash
git clone https://github.com/Yimura/claude-code-proxy.git
cd claude-code-proxy
cp .env.example .env
docker compose up --build -d
```

The default Compose configuration makes the application listen on `0.0.0.0:8082` inside the container while publishing it only on `127.0.0.1:8082` on the host. `PROXY_PORT` changes the application listen port and both sides of the loopback-only publication together. Do not bind it to an external interface without adding authentication and network access controls; the proxy does not authenticate inbound requests.

Compose supplies repository `model_mapping.json` as a read-only application config at `/claude-code-proxy/model_mapping.json`. Edit the local file, then recreate the service to apply changes:

```bash
docker compose up --build -d --force-recreate
```

The image starts through the installed `claude-code-proxy` executable; `uv run` is not required inside the container. Query the running container with:

```bash
docker compose exec proxy claude-code-proxy ps
```

The default control socket is container-local at `/run/claude-code-proxy/control.sock`. It is not published or mounted, so a CLI running on the host cannot query the container unless you deliberately change the deployment. On macOS and Windows hosts, run `ps` through `docker compose exec` as shown above; the source launcher and host-side control client are supported on Linux only.

### Connect Claude Code

Point your existing [Claude Code installation](https://code.claude.com/docs/en/setup) at the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## CLI and Session Inspection

Run the CLI without a subcommand to show help:

```bash
uv run claude-code-proxy
```

`proxy` is foreground-only; it does not fork, write a PID file, or manage a background daemon. Docker Compose's `restart: unless-stopped` policy supplies the container lifecycle. For source deployments, use your process manager when a daemon-like lifecycle is required. Direct Uvicorn application-target startup is retired, not deprecated or retained as a compatible startup path, because only the CLI coordinates both servers.

From the same source environment, inspect the running process over its local Unix socket:

```bash
uv run claude-code-proxy ps
uv run claude-code-proxy ps --filter state=active --filter provider=openai
uv run claude-code-proxy ps --format json
uv run claude-code-proxy ps --no-trunc
```

Filters use `key=value` and may be repeated. Repeated values for one key are alternatives, while different keys are combined. Supported keys are `id`, `state`, `provider`, `transport`, `model`, and `effort`; ID values may be unambiguous prefixes. Output defaults to a table, `--format json` returns the complete structured rows, and `--no-trunc` preserves full session and model values in table output. Source commands automatically select a socket under `XDG_RUNTIME_DIR` or another private runtime fallback. Use `--socket PATH` to override the socket for one command, or set `CONTROL_SOCKET_PATH` to an absolute socket path whose private parent directory already exists.

Each row has a safe, opaque, process-local hashed session ID. States are `active` while one or more requests are running, `idle` after the latest request completes, and `failed` after the latest request fails. Rows and their latest metadata are retained in memory only until the proxy process restarts; they are not prompt history. By default the registry retains the 1,000 most recently inactive logical rows, while active rows are never evicted.

The session registry excludes prompts and messages, system instructions, tool definitions, tool inputs and results, credentials, encrypted reasoning, and raw client session IDs. It records only the latest operational metadata needed for inspection, such as resolved model, provider, transport, effort, request counts, state, and timestamps.

The control app listens only on a local Unix socket and its routes are not added to the public TCP API. In Docker, that socket remains inside the container by default; use `docker compose exec proxy claude-code-proxy ps` to query it. A host-side CLI cannot access it unless you deliberately change the deployment.

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
| `PROXY_HOST` | Selects the public proxy bind host | Optional; keep source deployments on loopback unless external access is secured | Built-in: `0.0.0.0`; `.env.example`: `127.0.0.1`; Compose container: `0.0.0.0` |
| `PROXY_PORT` | Selects the public proxy TCP port; Compose applies it to the listener and loopback publication | Optional | `8082` |
| `CONTROL_SOCKET_PATH` | Overrides the private local control Unix socket path | Optional for source; must be absolute with an existing private parent | Automatic XDG/private fallback for source; image uses `/run/claude-code-proxy/control.sock` |
| `SESSION_RETENTION_LIMIT` | Sets the maximum inactive logical session rows retained in memory | Optional; must be an integer from `0` through `9223372036854775807` | `1000` |

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
- `context_window`: positive token count up to `9223372036854775807`, or `null` when capability is unknown.

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

1. The CLI starts the public TCP proxy and a separate private local control app on a Unix socket. Control routes are never registered on the public TCP listener.
2. The public app receives an Anthropic-compatible request.
3. The proxy resolves its model and reasoning policy through `model_mapping.json`.
4. The proxy selects LiteLLM or Codex transport.
5. The proxy translates and sends the upstream request.
6. The proxy normalizes streaming or non-streaming output into Anthropic-compatible responses and updates in-memory session metadata.

## Development

Application code lives in `src/claude_code_proxy/`; CLI coordination is in `cli.py`, the local control app and socket support are under `control/`, session metadata is in `observability.py`, and unit tests live in `test/unit/`.

```bash
uv sync --locked --dev
uv run pytest
uv run claude-code-proxy proxy
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
