ARG BASE_IMAGE=python:3.12.2-slim
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git

RUN pip install --no-cache uv

RUN uv venv /opt/venv 
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./pyproject.toml

# Now VIRTUAL_ENV is not set, so --python flag works correctly
RUN uv lock && \
    uv sync --python /opt/venv/bin/python

# Set VIRTUAL_ENV after sync so it doesn't interfere
ENV VIRTUAL_ENV="/opt/venv"

RUN git clone https://github.com/santoshgdev/fitnessllm-shared.git /tmp/fitnessllm-shared \
 && uv pip install --python /opt/venv/bin/python -e /tmp/fitnessllm-shared

WORKDIR /app

COPY notebooks ./notebooks
COPY tooling ./tooling

RUN cd /tmp/fitnessllm-shared \
 && echo "$(git rev-parse --short=5 HEAD)" > /app/commit_hash.txt

RUN printf '#!/usr/bin/env bash\n\
    export FITNESSLLM_SHARED_COMMIT_HASH=$(cat /app/commit_hash.txt)\n\
    exec uv run \"${@}\"' > /app/entrypoint.sh \
     && chmod +x /app/entrypoint.sh
