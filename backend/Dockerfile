FROM python:3.13-slim

# Accept build arguments from docker-compose
ARG BACKEND_INTERNAL_PORT=

WORKDIR /app

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=2.2.1 \
    BACKEND_INTERNAL_PORT=${BACKEND_INTERNAL_PORT}

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./

# Copy application code
COPY app ./app

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Clean up build dependencies to reduce image size (keep curl for health checks)
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

CMD uvicorn main:app --host 0.0.0.0 --port ${BACKEND_INTERNAL_PORT}
