# Stage 1 image: a place to run one-off commands (ingestion, the integration
# tests, the evaluation) with the same Python and libraries everywhere.
# Stage 4 turns this into a finished service image.
FROM python:3.13-slim

# uv installs the locked dependencies. Use the same uv version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /bin/uv

WORKDIR /app

# HF_HOME is where the embedding model is cached; compose.yaml mounts a volume
# there so a new container does not download it again.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, so this slow layer is reused until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Then the project itself.
COPY . .
RUN uv sync --locked

CMD ["python", "-m", "guide_search.ingest"]
