# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

RUN id -u proxy >/dev/null 2>&1 || useradd --create-home --uid 10001 proxy \
    && mkdir -p /var/lib/reclaimllm-enterprise-proxy \
    && chown proxy /var/lib/reclaimllm-enterprise-proxy

COPY app ./app
USER proxy

EXPOSE 8779

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8779"]
