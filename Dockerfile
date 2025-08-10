ARG BASE_IMAGE=python:3.12.2-slim
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git

RUN pip install --no-cache uv

# Set up project directory first
WORKDIR /app

# Copy dependency files for installation (better caching)
COPY pyproject.toml uv.lock* ./

# Let uv create and manage its own .venv in the project directory
RUN uv lock && uv sync

# Update PATH and VIRTUAL_ENV to use uv's managed environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

RUN git clone https://github.com/santoshgdev/fitnessllm-shared.git /tmp/fitnessllm-shared \
 && uv pip install -e /tmp/fitnessllm-shared

# Copy application code (separate layer for better caching)
COPY notebooks ./notebooks
COPY tooling ./tooling

RUN cd /tmp/fitnessllm-shared \
 && echo "$(git rev-parse --short=5 HEAD)" > /app/commit_hash.txt

RUN printf '#!/usr/bin/env bash\n\
    export FITNESSLLM_SHARED_COMMIT_HASH=$(cat /app/commit_hash.txt)\n\
    exec uv run \"${@}\"' > /app/entrypoint.sh \
     && chmod +x /app/entrypoint.sh
