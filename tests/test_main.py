import sys
import json
import logging
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend import GatewayContext, ModelAnalysisContext
from app.config import settings
from app.ingest import SupabaseGatewayIngestor
from app.main import _build_litellm_kwargs, _build_record, _extract_usage, _provider_model_name
from app.openapi_specs import ModelTestPlan


@pytest.fixture(autouse=True)
def stub_proxy_owned_openapi_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def plan_model_test(**kwargs) -> ModelTestPlan:
        return ModelTestPlan(
            operation=kwargs["preferred_operation"],
            source=kwargs["fallback_source"],
        )

    monkeypatch.setattr(
        "app.main.openapi_spec_registry.plan_model_test",
        plan_model_test,
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("azure/text-embedding-small", "azure/text-embedding-small"),
        ("azure_openai/gpt-4o", "azure/gpt-4o"),
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("moonshot/kimi-k2.5", "moonshot/kimi-k2.5"),
        ("deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ("  gpt-4o-mini  ", "gpt-4o-mini"),
        ("  azure/gpt-4o-mini  ", "azure/gpt-4o-mini"),
        ("local/llama3.2", "llama3.2"),
        ("litellm_proxy/llama3.2", "llama3.2"),
    ],
)
def test_provider_model_name(model: str, expected: str) -> None:
    assert _provider_model_name(model) == expected


def test_provider_model_name_with_provider() -> None:
    assert _provider_model_name("gpt-4o", provider="azure_openai") == "azure/gpt-4o"
    assert _provider_model_name("azure/gpt-4o", provider="azure_openai") == "azure/gpt-4o"
    assert _provider_model_name("text-embedding-small", provider="azure") == "azure/text-embedding-small"
    assert _provider_model_name("ollama/llama3.2", provider="ollama") == "llama3.2"
    assert (
        _provider_model_name("runpod/meta-llama/Llama-3-70b", provider="runpod")
        == "meta-llama/Llama-3-70b"
    )


def test_build_litellm_kwargs_for_custom_local_provider() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="ollama",
        provider_config={
            "base_url": "http://localhost:11434",
            "provider_type": "local",
        },
    )

    kwargs = _build_litellm_kwargs(
        context,
        {
            "model": "ollama/llama3.2",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert kwargs["model"] == "llama3.2"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_base"] == "http://localhost:11434/v1"


def test_build_litellm_kwargs_maps_azure_base_url_aliases() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="azure_openai",
        provider_config={
            "api_key": "azure-key",
            "base_url": "https://example.openai.azure.com",
            "api_version": "2024-10-21",
        },
    )

    kwargs = _build_litellm_kwargs(
        context,
        {
            "model": "azure/text-embedding-small",
            "input": ["hello"],
        },
    )

    assert kwargs["model"] == "azure/text-embedding-small"
    assert kwargs["api_key"] == "azure-key"
    assert kwargs["base_url"] == "https://example.openai.azure.com"
    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["api_version"] == "2024-10-21"
    assert "azure_api_version" not in kwargs


def test_build_litellm_kwargs_preserves_service_tier() -> None:
    """OpenAI and Azure OpenAI both accept request-level service_tier.

    Azure: auto | default | priority (flex currently falls back; hard-errors after
    2026-09-25 if the deployment lacks Flex). Forward the client value unchanged.
    """
    for provider, model in (
        ("openai", "openai/gpt-4o-mini"),
        ("azure_openai", "azure/gpt-4o-mini"),
    ):
        context = GatewayContext(
            org_id="org",
            org_slug="acme",
            team_id="team",
            user_id="user",
            key_id="key",
            provider=provider,
            provider_config={"api_key": "org-key"},
        )

        kwargs = _build_litellm_kwargs(
            context,
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "service_tier": "priority",
            },
        )

        assert kwargs["service_tier"] == "priority"
        assert kwargs["api_key"] == "org-key"


def test_build_litellm_kwargs_strips_credential_and_routing_params() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="openai",
        provider_config={"api_key": "org-key"},
    )

    kwargs = _build_litellm_kwargs(
        context,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "api_base": "https://attacker.example",
            "base_url": "https://attacker.example",
            "api_key": "attacker-key",
            "extra_headers": {"X-Exfil": "1"},
            "azure_ad_token": "stolen",
            "aws_access_key_id": "AKIA...",
            "vertex_credentials": "{}",
            "custom_llm_provider": "openai",
        },
    )

    assert kwargs["api_key"] == "org-key"
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["api_base"] == "https://api.openai.com/v1"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    for blocked in (
        "extra_headers",
        "azure_ad_token",
        "aws_access_key_id",
        "vertex_credentials",
        "custom_llm_provider",
    ):
        assert blocked not in kwargs


def test_build_litellm_kwargs_defaults_openai_base_url() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="openai",
        provider_config={"api_key": "sk-openai"},
    )
    kwargs = _build_litellm_kwargs(
        context,
        {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["api_base"] == "https://api.openai.com/v1"


def test_build_litellm_kwargs_preserves_custom_openai_base_url() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="openai",
        provider_config={
            "api_key": "sk-openai",
            "base_url": "https://custom-openai-proxy.internal/v1",
        },
    )
    kwargs = _build_litellm_kwargs(
        context,
        {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert kwargs["base_url"] == "https://custom-openai-proxy.internal/v1"
    assert kwargs["api_base"] == "https://custom-openai-proxy.internal/v1"


def test_build_litellm_kwargs_local_provider_uses_openai_custom_provider_and_base_url() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="local",
        provider_config={
            "api_key": "lX9tnk7RtpzTv6drVyv9VdVmZ2Wy-UTB68NQcqNFaVk",
            "base_url": "https://llm.cv-fit.click",
        },
    )
    kwargs = _build_litellm_kwargs(
        context,
        {"model": "local/llama3.2", "messages": [{"role": "user", "content": "Hi!"}]},
    )
    assert kwargs["model"] == "llama3.2"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["base_url"] == "https://llm.cv-fit.click/v1"
    assert kwargs["api_base"] == "https://llm.cv-fit.click/v1"
    assert kwargs["api_key"] == "lX9tnk7RtpzTv6drVyv9VdVmZ2Wy-UTB68NQcqNFaVk"


def test_build_litellm_kwargs_provider_config_wins_over_payload() -> None:
    context = GatewayContext(
        org_id="org",
        org_slug="acme",
        team_id="team",
        user_id="user",
        key_id="key",
        provider="openai",
        provider_config={"api_key": "org-key", "api_base": "https://org.example"},
    )

    kwargs = _build_litellm_kwargs(context, {"model": "gpt-4o-mini", "messages": []})

    assert kwargs["api_key"] == "org-key"
    assert kwargs["api_base"] == "https://org.example"
    assert kwargs["base_url"] == "https://org.example"


@pytest.mark.asyncio
async def test_model_analysis_endpoint_resolves_org_and_does_not_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from httpx import ASGITransport, AsyncClient

    from app.config import settings
    from app.main import app

    class Backend:
        async def resolve_model_analysis(
            self,
            *,
            org_id: str,
            run_id: str,
            result_id: str,
            purpose: str,
            model: str,
        ):
            assert org_id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
            assert run_id == "11111111-1111-1111-1111-111111111111"
            assert result_id == "22222222-2222-2222-2222-222222222222"
            assert purpose == "candidate"
            assert model == "openai/gpt-4o-mini"
            return ModelAnalysisContext(
                org_id=org_id,
                org_slug="acme",
                provider="openai",
                provider_config={"api_key": "secret"},
            )

    async def completion(**kwargs):
        assert kwargs["api_key"] == "secret"
        assert kwargs["stream"] is False
        return {
            "id": "chatcmpl-analysis",
            "choices": [{"message": {"role": "assistant", "content": "billing"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

    def unexpected_cost_lookup(**kwargs):
        raise AssertionError("model-analysis cost lookup must remain disabled")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.acompletion", completion)
    monkeypatch.setattr("app.main.litellm.completion_cost", unexpected_cost_lookup)
    app.state.backend = Backend()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/model-analysis/v1/chat/completions",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "org_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "run_id": "11111111-1111-1111-1111-111111111111",
                "result_id": "22222222-2222-2222-2222-222222222222",
                "purpose": "candidate",
                "request": {
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "classify"}],
                    "stream": False,
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["response"]["choices"][0]["message"]["content"] == "billing"
    assert response.json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}
    assert response.json()["cost_usd"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_preview"),
    [
        ("chat_completion", "OK"),
        ("embedding", "Embedding returned 3 dimensions"),
    ],
)
async def test_provider_model_test_uses_minimal_non_ingested_request(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    expected_preview: str,
) -> None:
    from app.main import app

    calls: list[dict] = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "OK"}}]}

    async def embedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.acompletion", completion)
    monkeypatch.setattr("app.main.litellm.aembedding", embedding)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/provider-model-test",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "provider": "openai",
                "model": "openai/text-embedding-3-small"
                if operation == "embedding"
                else "openai/gpt-4o-mini",
                "operation": operation,
                "provider_config": {"api_key": "secret"},
            },
        )

    assert response.status_code == 200
    assert response.json()["operation"] == operation
    assert response.json()["response_preview"] == expected_preview
    assert calls[0]["api_key"] == "secret"
    if operation == "embedding":
        assert calls[0]["input"] == ["test"]
    else:
        assert calls[0]["messages"] == [{"role": "user", "content": "Reply 1."}]
        assert calls[0]["max_tokens"] == 100
        assert calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_provider_model_test_redacts_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app

    class ProviderFailure(Exception):
        status_code = 401

    async def completion(**_kwargs):
        raise ProviderFailure("request failed with api_key=secret")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.acompletion", completion)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/provider-model-test",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "provider": "openai",
                "model": "openai/gpt-4o-mini",
                "operation": "chat_completion",
                "provider_config": {"api_key": "secret"},
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Provider rejected the chat completion test (HTTP 401)"
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_provider_model_test_dispatches_image_model_to_image_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app

    calls: list[dict] = []

    async def image_generation(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": "image-bytes"}]}

    async def unexpected_completion(**_kwargs):
        raise AssertionError("image model must not use chat completion")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.aimage_generation", image_generation)
    monkeypatch.setattr("app.main.litellm.acompletion", unexpected_completion)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/provider-model-test",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "provider": "openai",
                "model": "openai/chatgpt-image-latest",
                "operation": "chat_completion",
                "provider_config": {"api_key": "secret"},
            },
        )

    assert response.status_code == 200
    assert response.json()["operation"] == "image_generation"
    assert response.json()["spec_source"] == "litellm"
    assert response.json()["response_preview"] == "Image generation returned 1 image"
    assert calls == [
        {
            "model": "chatgpt-image-latest",
            "prompt": "A solid blue circle on a white background.",
            "n": 1,
            "api_key": "secret",
            "base_url": "https://api.openai.com/v1",
            "api_base": "https://api.openai.com/v1",
        }
    ]


@pytest.mark.asyncio
async def test_provider_model_test_rejects_realtime_instead_of_using_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app

    async def unexpected_completion(**_kwargs):
        raise AssertionError("realtime model must not use chat completion")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.acompletion", unexpected_completion)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/provider-model-test",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "provider": "openai",
                "model": "openai/gpt-realtime",
                "operation": "chat_completion",
                "provider_config": {"api_key": "secret"},
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Model mode 'realtime' does not support a bounded credential test"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["ollama/embeddinggemma", "ollama/embed-v-4-0", "local/bge-embed-large"],
)
async def test_provider_model_test_dispatches_embed_name_to_embedding(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    from app.main import app

    calls: list[dict] = []

    async def embedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    async def unexpected_completion(**_kwargs):
        raise AssertionError("embed model must not use chat completion")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.aembedding", embedding)
    monkeypatch.setattr("app.main.litellm.acompletion", unexpected_completion)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/provider-model-test",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "provider": "ollama",
                "model": model_name,
                "operation": "chat_completion",
                "provider_config": {"api_key": "secret", "base_url": "http://localhost:11434/v1"},
            },
        )

    assert response.status_code == 200
    assert response.json()["operation"] == "embedding"
    assert response.json()["spec_source"] == "heuristic"
    assert calls[0]["input"] == ["test"]


@pytest.mark.asyncio
async def test_provider_model_test_dispatches_explicit_mode_to_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app

    calls: list[dict] = []

    async def embedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    async def unexpected_completion(**_kwargs):
        raise AssertionError("embedding mode must not use chat completion")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.aembedding", embedding)
    monkeypatch.setattr("app.main.litellm.acompletion", unexpected_completion)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/internal/provider-model-test",
            headers={"X-Proxy-Secret": "proxy-secret"},
            json={
                "provider": "ollama",
                "model": "ollama/custom-model-without-embed-in-name",
                "operation": "chat_completion",
                "mode": "embedding",
                "provider_config": {"api_key": "secret", "base_url": "http://localhost:11434/v1"},
            },
        )

    assert response.status_code == 200
    assert response.json()["operation"] == "embedding"
    assert calls[0]["input"] == ["test"]


@pytest.mark.asyncio
async def test_gateway_ingest_calls_backend_owned_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "proxy_shared_secret", "test-proxy-secret")
    ingest_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith(
            "/api/enterprise/gateway/ingest"
        ):
            ingest_requests.append(json.loads(request.content))
            assert request.headers["X-Proxy-Secret"] == "test-proxy-secret"
            return httpx.Response(
                201,
                json={
                    "session_id": "11111111-1111-1111-1111-111111111111",
                    "user_id": "22222222-2222-2222-2222-222222222222",
                    "blob_key": "sessions/11111111-1111-1111-1111-111111111111.json",
                },
            )
        return httpx.Response(404, json={"message": "unexpected request"})

    ingestor = SupabaseGatewayIngestor()
    await ingestor._backend_client.aclose()
    ingestor._backend_client = httpx.AsyncClient(
        base_url="http://backend.test",
        transport=httpx.MockTransport(handler),
    )
    context = GatewayContext(
        org_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        org_slug="acme",
        team_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
        user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        key_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        provider="openai",
        provider_config={},
    )

    result = await ingestor.ingest_gateway_record(
        context=context,
        record={
            "session_id": "11111111-1111-1111-1111-111111111111",
            "method": "POST",
            "url": "https://proxy.reclaimllm.com/acme/v1/chat/completions",
            "response_status": 200,
            "is_streaming": False,
            "duration_ms": 42,
            "model": "openai/gpt-4o-mini",
            "total_input_tokens": 5,
            "total_output_tokens": 7,
            "request": {"model": "openai/gpt-4o-mini", "messages": []},
            "response": {"id": "chatcmpl-test", "usage": {}},
        },
    )

    await ingestor.close()

    assert result.user_id == "22222222-2222-2222-2222-222222222222"
    assert result.session_id == "11111111-1111-1111-1111-111111111111"
    assert len(ingest_requests) == 1
    body = ingest_requests[0]
    assert body["org_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert body["team_id"] == "dddddddd-dddd-dddd-dddd-dddddddddddd"
    assert body["user_id"] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert body["key_id"] == "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    assert body["provider"] == "openai"
    assert body["record"]["session_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["record"]["request"] == {"model": "openai/gpt-4o-mini", "messages": []}
    assert body["record"]["response"] == {"id": "chatcmpl-test", "usage": {}}


@pytest.mark.asyncio
async def test_responses_endpoint_azure_openai_removes_base_url_and_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app

    class Backend:
        async def resolve_gateway_key(
            self,
            *,
            org_slug: str,
            bearer_token: str,
            model: str,
        ):
            return GatewayContext(
                org_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                org_slug="acme",
                team_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
                user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                key_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                provider="azure_openai",
                provider_config={
                    "api_key": "azure-secret-key",
                    "base_url": "https://example.openai.azure.com",
                    "api_version": "2024-10-21",
                },
            )

    calls: list[dict] = []

    async def fake_aresponses(**kwargs):
        calls.append(kwargs)
        return {"id": "resp-1", "output": "Hello world"}

    class MockIngestor:
        async def ingest_gateway_record(self, *args, **kwargs):
            pass

    monkeypatch.setattr("app.main.litellm.aresponses", fake_aresponses)
    app.state.backend = Backend()
    app.state.ingestor = MockIngestor()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/acme/v1/responses",
            headers={"Authorization": "Bearer rclm-gw-test"},
            json={
                "model": "azure/gpt-4o",
                "input": "Hello",
                "service_tier": "priority",
                "max_output_tokens": 100,
            },
        )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0]["api_key"] == "azure-secret-key"
    assert calls[0]["api_base"] == "https://example.openai.azure.com"
    assert calls[0]["model"] == "azure/gpt-4o"
    assert calls[0]["input"] == "Hello"
    assert calls[0]["service_tier"] == "priority"
    assert "base_url" not in calls[0]
    assert "max_output_tokens" not in calls[0]
    assert "api_version" not in calls[0]


@pytest.mark.asyncio
async def test_responses_endpoint_openai_removes_base_url_and_max_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app

    class Backend:
        async def resolve_gateway_key(
            self,
            *,
            org_slug: str,
            bearer_token: str,
            model: str,
        ):
            return GatewayContext(
                org_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                org_slug="acme",
                team_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
                user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                key_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                provider="openai",
                provider_config={
                    "api_key": "openai-secret-key",
                },
            )

    calls: list[dict] = []

    async def fake_aresponses(**kwargs):
        calls.append(kwargs)
        return {"id": "resp-1", "output": "Hello world"}

    class MockIngestor:
        async def ingest_gateway_record(self, *args, **kwargs):
            pass

    monkeypatch.setattr("app.main.litellm.aresponses", fake_aresponses)
    app.state.backend = Backend()
    app.state.ingestor = MockIngestor()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/acme/v1/responses",
            headers={"Authorization": "Bearer rclm-gw-test"},
            json={
                "model": "openai/gpt-4o",
                "input": "Hello",
                "max_output_tokens": 50,
            },
        )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0]["api_key"] == "openai-secret-key"
    assert calls[0]["api_base"] == "https://api.openai.com/v1"
    assert calls[0]["model"] == "gpt-4o"
    assert calls[0]["input"] == "Hello"
    assert "base_url" not in calls[0]
    assert "max_output_tokens" not in calls[0]


@pytest.mark.asyncio
async def test_chat_completions_litellm_error_tracks_posthog_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.main import app

    class Backend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug="acme",
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    async def faulty_acompletion(**kwargs):
        class ProviderError(Exception):
            status_code = 429
        raise ProviderError("Rate limit exceeded")

    tracked_errors: list[dict] = []

    def mock_track_litellm_error(exc, **kwargs):
        tracked_errors.append({"exc": exc, **kwargs})

    monkeypatch.setattr("app.main.litellm.acompletion", faulty_acompletion)
    monkeypatch.setattr("app.main.track_litellm_error", mock_track_litellm_error)

    app.state.backend = Backend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/acme/v1/chat/completions",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert response.status_code == 502
    assert len(tracked_errors) == 1
    err = tracked_errors[0]
    assert err["operation"] == "chat_completions"
    assert err["model"] == "openai/gpt-4o"
    assert err["provider"] == "openai"
    assert err["status_code"] == 429
    assert err["org_id"] == "org-test-id"
    assert err["org_slug"] == "acme"
    assert err["user_id"] == "user-test-id"
    assert err["team_id"] == "team-test-id"
    assert err["tag"] == "litellm_exception"
    assert any(
        "Provider request failed provider=openai model=openai/gpt-4o operation=chat_completions org=acme"
        in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_chat_completions_no_credits_remaining_error_handled_and_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import litellm
    from app.main import app

    class Backend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug="acme",
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    no_credits_msg = (
        "RateLimitError: OpenAIException - Error code: 429 - "
        "{'error': {'message': 'You have no credits remaining. Add credits to continue using the API at "
        "https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', "
        "'param': None, 'code': 'credit_balance_exhausted'}}"
    )

    async def faulty_acompletion(**kwargs):
        raise litellm.RateLimitError(
            message=no_credits_msg,
            llm_provider="openai",
            model="openai/gpt-4o",
        )

    tracked_errors: list[dict] = []

    def mock_track_litellm_error(exc, **kwargs):
        tracked_errors.append({"exc": exc, **kwargs})

    monkeypatch.setattr("app.main.litellm.acompletion", faulty_acompletion)
    monkeypatch.setattr("app.main.track_litellm_error", mock_track_litellm_error)

    app.state.backend = Backend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/acme/v1/chat/completions",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert response.status_code == 429
    data = response.json()
    assert data["error"]["code"] == "credit_balance_exhausted"
    assert data["error"]["type"] == "insufficient_quota"
    assert "no credits remaining" in data["error"]["message"].lower()

    # Verify PostHog tracking
    assert len(tracked_errors) == 1
    err = tracked_errors[0]
    assert err["operation"] == "chat_completions"
    assert err["tag"] == "credit_balance_exhausted"
    assert err["status_code"] == 429

    # Verify logging difference:
    assert any("Provider credit balance exhausted (no credits remaining)" in rec.message for rec in caplog.records)
    assert not any("Provider request failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_completions_no_credits_remaining(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import litellm
    from app.main import app

    class Backend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug="acme",
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    class Ingestor:
        async def ingest_gateway_record(self, **kwargs):
            pass

    no_credits_msg = (
        "RateLimitError: OpenAIException - Error code: 429 - "
        "{'error': {'message': 'You have no credits remaining. Add credits to continue using the API at "
        "https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', "
        "'param': None, 'code': 'credit_balance_exhausted'}}"
    )

    async def faulty_stream(**kwargs):
        raise litellm.RateLimitError(
            message=no_credits_msg,
            llm_provider="openai",
            model="openai/gpt-4o",
        )

    tracked_errors: list[dict] = []

    def mock_track_litellm_error(exc, **kwargs):
        tracked_errors.append({"exc": exc, **kwargs})

    monkeypatch.setattr("app.main.litellm.acompletion", faulty_stream)
    monkeypatch.setattr("app.main.track_litellm_error", mock_track_litellm_error)

    app.state.backend = Backend()
    app.state.ingestor = Ingestor()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/acme/v1/chat/completions",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert "credit_balance_exhausted" in response.text
    assert "insufficient_quota" in response.text
    assert len(tracked_errors) == 1
    assert tracked_errors[0]["tag"] == "credit_balance_exhausted"
    assert any("Provider credit balance exhausted (no credits remaining) during stream" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_fastapi_rate_limit_exception_handler_registered():
    import litellm
    from app.main import app

    assert litellm.RateLimitError in app.exception_handlers


@pytest.mark.asyncio
async def test_embeddings_no_credits_remaining(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import logging
    import litellm
    from app.main import app

    class Backend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug="acme",
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    no_credits_msg = (
        "RateLimitError: OpenAIException - Error code: 429 - "
        "{'error': {'message': 'You have no credits remaining. Add credits to continue using the API at "
        "https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', "
        "'param': None, 'code': 'credit_balance_exhausted'}}"
    )

    async def faulty_embedding(**kwargs):
        raise litellm.RateLimitError(
            message=no_credits_msg,
            llm_provider="openai",
            model="text-embedding-3-small",
        )

    tracked_errors: list[dict] = []
    monkeypatch.setattr("app.main.litellm.aembedding", faulty_embedding)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: tracked_errors.append({"exc": exc, **kwargs}))

    app.state.backend = Backend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/acme/v1/embeddings",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={"model": "text-embedding-3-small", "input": ["hello"]},
            )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "credit_balance_exhausted"
    assert tracked_errors[0]["tag"] == "credit_balance_exhausted"
    assert any("Provider credit balance exhausted (no credits remaining)" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_provider_model_test_no_credits_remaining(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import litellm
    from app.main import app

    no_credits_msg = (
        "RateLimitError: OpenAIException - Error code: 429 - "
        "{'error': {'message': 'You have no credits remaining. Add credits to continue using the API at "
        "https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', "
        "'param': None, 'code': 'credit_balance_exhausted'}}"
    )

    async def completion(**_kwargs):
        raise litellm.RateLimitError(
            message=no_credits_msg,
            llm_provider="openai",
            model="openai/gpt-4o-mini",
        )

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.acompletion", completion)

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/internal/provider-model-test",
                headers={"X-Proxy-Secret": "proxy-secret"},
                json={
                    "provider": "openai",
                    "model": "openai/gpt-4o-mini",
                    "operation": "chat_completion",
                    "provider_config": {"api_key": "secret"},
                },
            )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "credit_balance_exhausted"
    assert any("Provider model test failed: no credits remaining" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_responses_no_credits_remaining(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import litellm
    from app.main import app

    class Backend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug="acme",
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    no_credits_msg = (
        "RateLimitError: OpenAIException - Error code: 429 - "
        "{'error': {'message': 'You have no credits remaining. Add credits to continue using the API at "
        "https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', "
        "'param': None, 'code': 'credit_balance_exhausted'}}"
    )

    async def faulty_responses(**kwargs):
        raise litellm.RateLimitError(
            message=no_credits_msg,
            llm_provider="openai",
            model="openai/gpt-4o",
        )

    tracked_errors: list[dict] = []
    monkeypatch.setattr("app.main.litellm.aresponses", faulty_responses, raising=False)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: tracked_errors.append({"exc": exc, **kwargs}))

    app.state.backend = Backend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/acme/v1/responses",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={"model": "openai/gpt-4o", "input": "hi"},
            )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "credit_balance_exhausted"
    assert tracked_errors[0]["tag"] == "credit_balance_exhausted"
    assert any("Provider credit balance exhausted (no credits remaining)" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_completions_error_logs_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.main import app

    class Backend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug="acme",
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    class Ingestor:
        async def ingest_gateway_record(self, **kwargs):
            pass

    async def faulty_stream(**kwargs):
        raise RuntimeError("Stream connection failed")

    monkeypatch.setattr("app.main.litellm.acompletion", faulty_stream)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: None)

    app.state.backend = Backend()
    app.state.ingestor = Ingestor()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/acme/v1/chat/completions",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert any(
        "Provider request failed during stream provider=openai model=gpt-4o org=acme" in rec.message
        or "Provider request failed during stream provider=openai model=openai/gpt-4o org=acme" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_model_analysis_chat_completions_error_logs_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.config import settings
    from app.main import app

    class Backend:
        async def resolve_model_analysis(self, **kwargs):
            return ModelAnalysisContext(
                org_id=kwargs["org_id"],
                org_slug="acme",
                provider="openai",
                provider_config={"api_key": "secret"},
            )

    async def faulty_completion(**kwargs):
        raise RuntimeError("Upstream provider dropped connection")

    monkeypatch.setattr(settings, "proxy_shared_secret", "proxy-secret")
    monkeypatch.setattr("app.main.litellm.acompletion", faulty_completion)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: None)

    app.state.backend = Backend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/internal/model-analysis/v1/chat/completions",
                headers={"X-Proxy-Secret": "proxy-secret"},
                json={
                    "org_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "run_id": "11111111-1111-1111-1111-111111111111",
                    "result_id": "22222222-2222-2222-2222-222222222222",
                    "purpose": "candidate",
                    "request": {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": "classify"}],
                        "stream": False,
                    },
                },
            )

    assert response.status_code == 502
    assert any(
        "Model-analysis provider request failed provider=openai model=openai/gpt-4o-mini org=acme" in rec.message
        for rec in caplog.records
    )


def test_extract_usage_chat_completions_prompt_completion_tokens() -> None:
    usage = _extract_usage(
        {"usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}}
    )
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 4


def test_extract_usage_responses_api_input_output_tokens() -> None:
    usage = _extract_usage(
        {
            "usage": {
                "input_tokens": 27420,
                "output_tokens": 1225,
                "total_tokens": 28645,
                "input_tokens_details": {"cached_tokens": 4352},
            }
        }
    )
    assert usage["prompt_tokens"] == 27420
    assert usage["completion_tokens"] == 1225
    assert usage["input_tokens"] == 27420
    assert usage["output_tokens"] == 1225


def test_extract_usage_from_streaming_chunks() -> None:
    usage = _extract_usage(
        {
            "chunks": [
                {"choices": [{"delta": {"content": "hi"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
                },
            ]
        }
    )
    assert usage["prompt_tokens"] == 9
    assert usage["completion_tokens"] == 2


def test_build_record_maps_responses_api_usage_to_totals() -> None:
    record = _build_record(
        payload={"model": "gpt-5.5", "stream": False},
        response_body={
            "object": "response",
            "usage": {"input_tokens": 5660, "output_tokens": 4025},
        },
        status_code=200,
        duration_ms=100,
        request_url="https://proxy.test/acme/v1/responses",
    )
    assert record["total_input_tokens"] == 5660
    assert record["total_output_tokens"] == 4025


@pytest.mark.asyncio
async def test_root_and_robots_endpoints() -> None:
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        res_get = await client.get("/")
        assert res_get.status_code == 200
        assert res_get.json() == {"status": "ok", "service": "ReclaimLLM Enterprise Proxy"}

        res_head = await client.head("/")
        assert res_head.status_code == 200

        res_robots = await client.get("/robots.txt")
        assert res_robots.status_code == 200
        assert "User-agent: *" in res_robots.text

        res_ping = await client.get("/cv-fit/v1")
        assert res_ping.status_code == 200
        assert res_ping.json() == {"status": "ok", "organization": "cv-fit"}

        res_ping_base = await client.get("/cv-fit")
        assert res_ping_base.status_code == 200
        assert res_ping_base.json() == {"status": "ok", "organization": "cv-fit"}


@pytest.mark.asyncio
async def test_list_and_retrieve_models_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app

    class MockBackend:
        async def list_gateway_models(self, *, org_slug: str, bearer_token: str):
            assert org_slug == "cv-fit"
            assert bearer_token == "rclm-gw-test"
            return [
                {
                    "id": "text-embedding-3-small",
                    "model_name": "text-embedding-3-small",
                    "provider": "openai",
                    "created": 1677610602,
                    "object": "model",
                    "owned_by": "organization",
                },
                {
                    "id": "openai/gpt-4o",
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "created": 1677610602,
                    "object": "model",
                    "owned_by": "organization",
                },
            ]

    app.state.backend = MockBackend()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        # 1. Missing auth
        res_no_auth = await client.get("/cv-fit/v1/models")
        assert res_no_auth.status_code == 401

        # 2. List models via /v1/models
        res_v1 = await client.get(
            "/cv-fit/v1/models",
            headers={"Authorization": "Bearer rclm-gw-test"},
        )
        assert res_v1.status_code == 200
        data_v1 = res_v1.json()
        assert data_v1["object"] == "list"
        assert len(data_v1["data"]) == 2
        assert data_v1["data"][0]["id"] == "text-embedding-3-small"

        # 3. List models via /models alias
        res_alias = await client.get(
            "/cv-fit/models",
            headers={"Authorization": "Bearer rclm-gw-test"},
        )
        assert res_alias.status_code == 200
        assert len(res_alias.json()["data"]) == 2

        # 4. Retrieve single model
        res_model = await client.get(
            "/cv-fit/v1/models/text-embedding-3-small",
            headers={"Authorization": "Bearer rclm-gw-test"},
        )
        assert res_model.status_code == 200
        assert res_model.json()["id"] == "text-embedding-3-small"

        # 5. Retrieve model with slash in id
        res_model_slash = await client.get(
            "/cv-fit/v1/models/openai/gpt-4o",
            headers={"Authorization": "Bearer rclm-gw-test"},
        )
        assert res_model_slash.status_code == 200
        assert res_model_slash.json()["model_name"] == "gpt-4o"

        # 6. Retrieve non-existent model
        res_404 = await client.get(
            "/cv-fit/v1/models/non-existent-model",
            headers={"Authorization": "Bearer rclm-gw-test"},
        )
        assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_non_v1_path_aliases_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app

    class MockBackend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug=org_slug,
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="openai",
                provider_config={"api_key": "fake-key"},
            )

    class MockIngestor:
        async def ingest_gateway_record(self, *args, **kwargs):
            pass

    async def fake_completion(**kwargs):
        return {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    async def fake_embedding(**kwargs):
        return {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

    monkeypatch.setattr("app.main.litellm.acompletion", fake_completion)
    monkeypatch.setattr("app.main.litellm.aembedding", fake_embedding)
    app.state.backend = MockBackend()
    app.state.ingestor = MockIngestor()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        # Non-v1 chat completions
        res_chat = await client.post(
            "/cv-fit/chat/completions",
            headers={"Authorization": "Bearer rclm-gw-test"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert res_chat.status_code == 200
        assert res_chat.json()["choices"][0]["message"]["content"] == "hello"

        # Non-v1 embeddings
        res_embed = await client.post(
            "/cv-fit/embeddings",
            headers={"Authorization": "Bearer rclm-gw-test"},
            json={"model": "text-embedding-3-small", "input": ["hello"]},
        )
        assert res_embed.status_code == 200
        assert len(res_embed.json()["data"]) == 1


@pytest.mark.asyncio
async def test_cloudflare_524_timeout_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.main import app

    class MockBackend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug=org_slug,
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="local",
                provider_config={"base_url": "https://llm.cv-fit.click"},
            )

    cf_524_html = (
        "<!DOCTYPE html><html><head><title>llm.cv-fit.click | 524: A timeout occurred</title></head>"
        "<body><h1>Error 524</h1><p>A timeout occurred</p>" + ("x" * 1000) + "</body></html>"
    )

    class Cloudflare524Error(Exception):
        status_code = 524

    async def faulty_embedding(**kwargs):
        raise Cloudflare524Error(cf_524_html)

    monkeypatch.setattr("app.main.litellm.aembedding", faulty_embedding)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: None)
    app.state.backend = MockBackend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/cv-fit/v1/embeddings",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={"model": "local/text-embedding-3-small", "input": ["test"]},
            )

    assert response.status_code == 504
    data = response.json()
    assert data["error"]["code"] == "upstream_timeout"
    assert data["error"]["type"] == "gateway_timeout"
    assert "Cloudflare Error 524" in data["error"]["message"]
    # Verify raw HTML is NOT dumped in log
    assert not any("<html" in rec.message for rec in caplog.records)
    assert any("Provider request failed provider=local" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_cloudflare_challenge_403_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.main import app

    class MockBackend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug=org_slug,
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="local",
                provider_config={"base_url": "https://llm.cv-fit.click"},
            )

    cf_challenge_html = (
        "<!DOCTYPE html><html><head><title>Just a moment...</title></head>"
        "<body>cf-mitigated: challenge Cloudflare Ray ID: 890123</body></html>"
    )

    class CloudflareChallengeError(Exception):
        status_code = 403

    async def faulty_embedding(**kwargs):
        raise CloudflareChallengeError(cf_challenge_html)

    monkeypatch.setattr("app.main.litellm.aembedding", faulty_embedding)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: None)
    app.state.backend = MockBackend()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/cv-fit/v1/embeddings",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={"model": "local/text-embedding-3-small", "input": ["test"]},
            )

    assert response.status_code == 502
    data = response.json()
    assert data["error"]["code"] == "upstream_cloudflare_challenge"
    assert "Cloudflare challenge" in data["error"]["message"]
    assert not any("<html" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_provider_error_includes_route_in_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.main import app

    class MockBackend:
        async def resolve_gateway_key(self, *, org_slug: str, bearer_token: str, model: str):
            return GatewayContext(
                org_id="org-test-id",
                org_slug=org_slug,
                team_id="team-test-id",
                user_id="user-test-id",
                key_id="key-test-id",
                provider="local",
                provider_config={"base_url": "https://llm.cv-fit.click"},
            )

    async def faulty_responses(**kwargs):
        raise RuntimeError("Local model timeout")

    monkeypatch.setattr("app.main.litellm.aresponses", faulty_responses)
    monkeypatch.setattr("app.main.track_litellm_error", lambda exc, **kwargs: None)
    app.state.backend = MockBackend()

    with caplog.at_level(logging.ERROR):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/cv-fit/v1/responses",
                headers={"Authorization": "Bearer rclm-gw-test"},
                json={"model": "local/qwen2.5:3b", "input": "hello"},
            )

    assert response.status_code == 504
    assert any(
        "Provider request failed provider=local model=local/qwen2.5:3b operation=responses org=cv-fit route=POST /cv-fit/v1/responses"
        in rec.message
        for rec in caplog.records
    )


def test_endpoint_filter_silences_health() -> None:
    from app.main import EndpointFilter

    flt = EndpointFilter()
    health_record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s" %s',
        args=("127.0.0.1:41234", "GET /health HTTP/1.1", 200),
        exc_info=None,
    )
    other_record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s" %s',
        args=("127.0.0.1:41234", "POST /cv-fit/v1/responses HTTP/1.1", 200),
        exc_info=None,
    )

    assert not flt.filter(health_record)
    assert flt.filter(other_record)



