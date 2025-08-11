FROM python:3.12-slim as builder

WORKDIR /app
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

FROM python:3.12-slim

RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser/app

COPY --from=builder /app/.venv ./.venv
COPY agent.py .

ENV PATH="/home/appuser/app/.venv/bin:$PATH"