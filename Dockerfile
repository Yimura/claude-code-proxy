FROM python:3.14-slim

WORKDIR /claude-code-proxy

ENV PATH=/claude-code-proxy/.venv/bin:$PATH
ENV CONTROL_SOCKET_PATH=/run/claude-code-proxy/control.sock

RUN install -d -m 0700 /run/claude-code-proxy

# Install dependencies separately so application-only changes reuse this layer.
COPY pyproject.toml uv.lock ./
RUN pip install --upgrade uv && uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY README.md model_mapping.json ./
RUN uv sync --locked --no-dev

EXPOSE 8082
CMD ["claude-code-proxy", "proxy"]
