# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.0.2] — 2026-10-06

### Added
- Support for locally-hosted and self-hosted model providers:
  - Configurable routing for `local`, `litellm_proxy`, and custom local provider slugs (`ollama`, `runpod`, etc.) with automatic `/v1` endpoint normalization and prefix stripping (`app/main.py`).
  - Added test coverage in `tests/test_main.py` for model name resolution and kwargs generation across custom endpoints.
- Model discovery endpoints:
  - Added `GET /{enterprise_slug}/v1/models` and `GET /{enterprise_slug}/models` to list authorized models for a gateway key (`app/main.py`).
  - Added `GET /{enterprise_slug}/v1/models/{model_id:path}` and `GET /{enterprise_slug}/models/{model_id:path}` to retrieve specific model details (`app/main.py`).
  - Added `list_gateway_models()` in `BackendClient` calling backend `/api/enterprise/gateway/models` (`app/backend.py`).
- Distinct quota exhaustion and credit balance error handling:
  - Added `_is_no_credits_error()` to detect `credit_balance_exhausted` and `insufficient_quota` across completions, streaming, responses, embeddings, and model tests (`app/main.py`).
  - Emits focused `logger.error` logs without stack traces to avoid log noise when provider credits are exhausted.
  - Returns structured HTTP 429 response with `code: "credit_balance_exhausted"` and `type: "insufficient_quota"`.
- Upstream provider error sanitization:
  - Intercepts and parses Cloudflare Error 524 timeouts into structured HTTP 504 `upstream_timeout` responses (`app/main.py`).
  - Intercepts Cloudflare Turnstile/challenge pages (HTTP 403) and sanitizes into structured `upstream_cloudflare_challenge` responses (`app/main.py`).
  - Parses HTML error pages into concise summaries without dumping raw HTML into logs or client responses (`app/main.py`).
- Explicit route logging in provider errors:
  - Added `route` parameter to `_handle_gateway_provider_error` and `_stream_chat_completion` to include the HTTP method and request path (e.g. `route=POST /{org}/v1/responses`) in failure logs (`app/main.py`).
- Silent healthcheck logging:
  - Added `EndpointFilter` to `uvicorn.access` logger to suppress continuous `/health` access log noise (`app/main.py`).
- PostHog error analytics integration:
  - Added `app/analytics.py` and `track_litellm_error()` to track gateway exceptions and credit exhaustion events with org, team, and model metadata.
  - Added `POSTHOG_KEY` and `POSTHOG_HOST` configuration settings in `app/config.py`.
- Gateway utility endpoints:
  - Added `GET /` and `HEAD /` service health check endpoints (`app/main.py`).
  - Added `GET /robots.txt` (`app/main.py`).
  - Added `GET /{enterprise_slug}/v1` and `GET /{enterprise_slug}` organization ping routes (`app/main.py`).
  - Added non-v1 path aliases for chat completions, responses, and embeddings (`app/main.py`).
- Token usage extraction and normalization:
  - Added `_normalise_usage_dict` to unify Chat Completions and Responses API usage metrics (`app/main.py`).
  - Added `_usage_from_chunks` to extract stream usage from final chunks (`app/main.py`).

### Changed
- Forward client `service_tier` through `_build_litellm_kwargs` for OpenAI and Azure OpenAI (`app/main.py`).
- Model-test operation dispatching recognizes explicit `mode` and embedding naming patterns (`app/main.py`).
- Azure OpenAI handling in `/responses` drops `base_url` and `api_version` for LiteLLM compatibility (`app/main.py`).
- Cleaned up exception log formatting across provider request failures, streaming errors, model-analysis requests, and gateway key resolution in `app/main.py`.

### Security
- Sanitized upstream provider HTML responses and Cloudflare challenge blocks to prevent leaking raw upstream HTML, Ray IDs, or server internals to clients.
- Enforced blocked payload parameters so clients cannot redirect credentials or alter backend-resolved provider settings.

### Performance
- Suppressed LiteLLM debug info, verbosity, and unnecessary parameter serialization at startup (`app/main.py`).
- Suppressed Uvicorn `/health` access log I/O overhead via `EndpointFilter`.

### Deps
- Added `posthog>=3.0,<4` for gateway error analytics (`pyproject.toml`).
- Relaxed `litellm` dependency specifier to `>=1.84.0,<2.0` (`pyproject.toml`).
- Updated `uv.lock`.

---

## [v0.0.1] — 2026-08-31

### Added
- `app/openapi_specs.py` — bounded OpenAPI loading, GitHub blob→raw normalization, 6-hour operation cache, and model-test operation planning from schema enums
- Proxy-owned OpenAPI spec URLs for OpenAI, Azure OpenAI, Anthropic, and Moonshot/Kimi (`provider_openapi_spec_url`)
- `POST /internal/provider-model-test` — minimal provider credential tests (chat, Responses, embeddings, image, speech, transcription, moderation, text completion, video) without session capture
- `POST /internal/model-analysis/v1/chat/completions` — trusted model-analysis chat completions resolved via backend, without ingest
- `ModelAnalysisContext` and `BackendClient.resolve_model_analysis()` calling `/api/enterprise/gateway/model-analysis/resolve`
- `_provider_model_name()` to strip provider prefixes for LiteLLM while preserving `moonshot/` and `deepseek/` prefixes
- Per-operation response preview helpers and explicit rejection of realtime models (WebSocket-only)
- README section documenting schema-aware model tests and proxy-owned spec sources
- Tests in `tests/test_openapi_specs.py` and expanded coverage in `tests/test_main.py`

### Changed
- `_build_litellm_kwargs` accepts `GatewayContext | ModelAnalysisContext` and passes normalized model names to LiteLLM
- `_normalise_provider_config` drops `openapi_spec_url` from forwarded provider config (proxy owns spec locations)
- Model-test operation selection prefers OpenAPI model enums, then LiteLLM metadata, then name-based heuristics

### Security
- Provider model-test failures return redacted 502 messages without leaking API keys
- OpenAPI fetches capped at 5 MiB, reject credential-bearing URLs, and spec URLs are never forwarded to LiteLLM

### Performance
- Parsed OpenAPI operations cached for six hours (128-entry cap; 5-minute TTL on fetch failures)

### Deps
- Added `pyyaml>=6,<7` for OpenAPI YAML parsing
- Refreshed `uv.lock`

---
