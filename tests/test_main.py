import sys
import json
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend import GatewayContext
from app.ingest import SupabaseGatewayIngestor
from app.main import _build_litellm_kwargs


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
