# Two images from one file.
#
#   runtime   the code, its libraries, the embedding model and the reranker.
#             No tests.
#             Runs the HTTP API by default, as an ordinary user.
#   tools     runtime plus the tests and their libraries, for the integration
#             tests and the evaluation.
#
# compose.yaml builds both: "api" from runtime, "tools" from tools.

FROM python:3.13-slim AS runtime

# uv installs the locked dependencies. Use the same uv version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /bin/uv

WORKDIR /app

# HF_HOME is where the two models are kept inside the image.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/opt/models \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, so this slow layer is reused until pyproject.toml or
# uv.lock change. --no-dev leaves out the test libraries.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Then the project itself: the code, the settings and the guides.
COPY README.md LICENSE settings.toml agent.toml ./
COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

# An ordinary user to run as. It owns the model folder and one folder it may
# write to (the agent's token records); the program files stay read-only to it.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /opt/models /app/var \
    && chown app /opt/models /app/var
USER app

# Download the two models into the image, at the revisions pinned in e5.py
# and cross_encoder.py. A container then starts without downloading anything.
# The reranker is the large one: about 2.3 GB.
RUN python -c "from guide_search.e5 import E5Embedder; E5Embedder()"
RUN python -c "from guide_search.cross_encoder import CrossEncoderReranker; CrossEncoderReranker()"

# From here on the models' host is never contacted.
ENV HF_HUB_OFFLINE=1 \
    USAGE_LOG=/app/var/usage.jsonl

EXPOSE 8000
CMD ["uvicorn", "guide_search.service:build", "--factory", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime AS tools

# The tools image runs as root on purpose: the evaluation writes its report
# into a folder shared with the host, which the ordinary user could not do.
USER root
COPY tests ./tests
COPY scripts ./scripts
COPY evaluation ./evaluation
RUN uv sync --locked

CMD ["pytest", "-q"]
