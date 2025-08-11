# --- Stage 1: Builder ---
# Use a specific version for reproducibility.
FROM python:3.12-slim as builder

# Set up the environment.
WORKDIR /app
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install Poetry.
RUN pip install poetry

# Copy only the dependency definition files first to leverage caching.
COPY pyproject.toml poetry.lock ./
# Install dependencies.
RUN poetry install --no-root

# --- Stage 2: Final Image ---
# Use the same base image for a smaller final footprint.
FROM python:3.12-slim

# Create a non-root user for security.
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser/app

# Copy the virtual environment with dependencies from the builder stage.
COPY --from=builder /app/.venv ./.venv
# Copy the application source code.
COPY agent.py .

# Add the venv to the PATH so 'python' commands work directly.
ENV PATH="/home/appuser/app/.venv/bin:$PATH"

# Command to run will be specified in docker-compose.yml