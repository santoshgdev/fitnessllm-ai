ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git

RUN pip install --no-cache uv

RUN uv venv /opt/venv 
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

COPY pyproject.toml ./pyproject.toml

# Use --python to explicitly target the existing venv
RUN uv lock && \
    uv sync

RUN git clone https://github.com/santoshgdev/fitnessllm-shared.git /tmp/fitnessllm-shared \
 && uv pip install -e /tmp/fitnessllm-shared

WORKDIR /app

COPY notebooks ./notebooks
COPY tooling ./tooling

RUN cd /tmp/fitnessllm-shared \
 && echo "$(git rev-parse --short=5 HEAD)" > /app/commit_hash.txt

RUN printf '#!/usr/bin/env bash\n\
    export FITNESSLLM_SHARED_COMMIT_HASH=$(cat /app/commit_hash.txt)\n\
    exec uv run \"${@}\"' > /app/entrypoint.sh \
     && chmod +x /app/entrypoint.sh
