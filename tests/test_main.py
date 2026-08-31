import sys
import json
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend import GatewayContext, ModelAnalysisContext
from app.config import settings
from app.ingest import SupabaseGatewayIngestor
from app.main import _build_litellm_kwargs, _provider_model_name
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
        ("azure/text-embedding-small", "text-embedding-small"),
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("moonshot/kimi-k2.5", "moonshot/kimi-k2.5"),
        ("deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ("  gpt-4o-mini  ", "gpt-4o-mini"),
    ],
)
def test_provider_model_name(model: str, expected: str) -> None:
    assert _provider_model_name(model) == expected


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

    assert kwargs["model"] == "text-embedding-small"
    assert kwargs["api_key"] == "azure-key"
    assert kwargs["base_url"] == "https://example.openai.azure.com"
    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["api_version"] == "2024-10-21"
    assert "azure_api_version" not in kwargs


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
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    for blocked in (
        "api_base",
        "base_url",
        "extra_headers",
        "azure_ad_token",
        "aws_access_key_id",
        "vertex_credentials",
        "custom_llm_provider",
    ):
        assert blocked not in kwargs


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
