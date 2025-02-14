FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_HOME="/var/poetry" \
    #POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y \
    cuda-minimal-build-12-2 \
    wget \
    curl \
    gcc \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh
VOLUME /root/.ollama
EXPOSE 11434

RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz \
    && tar xzf Python-3.12.2.tgz \
    && cd Python-3.12.2 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.12.2.tgz Python-3.12.2

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python3 \
    && ln -s /usr/local/bin/python3.12 /usr/local/bin/python

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN mkdir /venv
WORKDIR /venv
COPY pyproject.toml .
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

RUN mkdir /app
WORKDIR /app

RUN /usr/local/bin/pip3.12 install notebook pysqlite2

CMD ["ollama", "serve"]