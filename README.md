# RCLM Enterprise Proxy

Part of [ReclaimLLM](https://reclaimllm.com) — the capture, search, and governance layer for AI coding assistants and LLM traffic.

This service is RCLM's **Enterprise LLM Gateway**: an OpenAI-compatible proxy that organizations point their apps and tools at instead of calling providers directly. It resolves a gateway key to a team/provider configuration via [ReclaimLLM-Backend](https://reclaimllm.com), forwards the request to the underlying provider through [LiteLLM](https://github.com/BerriAI/litellm), and logs the interaction back to the backend for cost attribution, audit, and observability.

It contains no business logic of its own — key resolution, storage, and policy all live in ReclaimLLM-Backend. This proxy only routes and forwards.

## How it works

```
client → POST /{enterprise_slug}/v1/chat/completions
       → proxy resolves gateway key + provider config via ReclaimLLM-Backend
       → proxy calls the provider through LiteLLM
       → proxy sends the request/response record back to ReclaimLLM-Backend for ingestion
       → response streamed/returned to client
```

Supported endpoints (mirroring the OpenAI API shape):

- `POST /{enterprise_slug}/v1/chat/completions` (streaming and non-streaming)
- `POST /{enterprise_slug}/v1/responses`
- `POST /{enterprise_slug}/v1/embeddings`
- `GET /health`

Requests are authenticated with a gateway key (`Authorization: Bearer <key>`) scoped to an organization by `{enterprise_slug}`. If a backend ingest call fails, the record is appended to a local JSONL queue (`GATEWAY_LOG_QUEUE_PATH`) instead of being dropped.

## Run locally

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # fill in the values below
uv run uvicorn app.main:app --reload --port 8779
```

The proxy needs a running ReclaimLLM-Backend instance to resolve gateway keys and ingest records against — see [reclaimllm.com](https://reclaimllm.com) for backend setup.

### Configuration

Set these as environment variables or in a `.env` file in this directory (also loaded from `../local.env` if present):

| Variable | Required | Description |
|---|---|---|
| `BACKEND_SERVER` | yes | Base URL of ReclaimLLM-Backend |
| `ENTERPRISE_PROXY_SHARED_SECRET` | yes | Shared secret sent as `X-Proxy-Secret` to authenticate this proxy to the backend |
| `SUPABASE_PROJECT_REF` | no | Supabase project ref, if the deployment resolves org data via Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | no | Supabase service role key |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | no | AWS credentials, if S3 access is needed at this layer |
| `AWS_REGION` | no | Defaults to `us-east-1` |
| `AWS_ENDPOINT_URL` | no | Override for S3-compatible endpoints (e.g. MinIO in dev) |
| `S3_BUCKET_US` / `S3_BUCKET_EU` | no | Region-scoped buckets for data residency |
| `GATEWAY_LOG_QUEUE_PATH` | no | Path for the failed-ingest fallback queue. Defaults to `/var/lib/reclaimllm-enterprise-proxy/failed_logs.jsonl` |
| `BACKEND_TIMEOUT_SECONDS` | no | HTTP timeout for backend calls. Defaults to `15` |
| `LOG_FLUSH_INTERVAL_SECONDS` | no | Defaults to `15` |

### Test a request

```bash
curl -X POST http://localhost:8779/acme/v1/chat/completions \
  -H "Authorization: Bearer <gateway-key>" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}'
```

## Run with Docker

```bash
docker build -t rclm-enterprise-proxy .
docker run --rm -p 8779:8779 --env-file .env rclm-enterprise-proxy
```

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## Learn more

Full product documentation, the enterprise gateway model, and account setup live at [reclaimllm.com](https://reclaimllm.com).
