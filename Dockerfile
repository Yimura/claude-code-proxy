FROM python:3.14-slim

WORKDIR /claude-code-proxy

# Install dependencies separately so application-only changes reuse this layer.
COPY pyproject.toml uv.lock ./
RUN pip install --upgrade uv && uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY README.md model_mapping.json ./
RUN uv sync --locked --no-dev

EXPOSE 8082
CMD ["uv", "run", "--no-dev", "--no-sync", "uvicorn", "claude_code_proxy.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8082"]
