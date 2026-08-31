# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
