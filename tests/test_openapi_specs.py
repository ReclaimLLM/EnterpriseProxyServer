import json

import httpx
import pytest

from app.openapi_specs import (
    OpenAPISpecRegistry,
    normalize_openapi_url,
    provider_openapi_spec_url,
)


@pytest.mark.parametrize(
    ("provider", "expected_url"),
    [
        (
            "openai",
            "https://github.com/openai/openai-openapi/blob/main/openapi.yaml",
        ),
        (
            "azure_openai",
            "https://github.com/Azure/azure-rest-api-specs/blob/main/specification/"
            "cognitiveservices/data-plane/AzureOpenAI/inference/stable/2024-10-21/inference.json",
        ),
        (
            "anthropic",
            "https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic/"
            "anthropic-893a61e9c1cd6c69a70d1043c626ed02d12d6a492eb6ca6ef7a84c64cfb15393.yml",
        ),
        ("moonshot", "https://platform.kimi.ai/docs/openapi.json"),
        ("gemini", None),
    ],
)
def test_proxy_owns_provider_openapi_sources(
    provider: str, expected_url: str | None
) -> None:
    assert provider_openapi_spec_url(provider) == expected_url


def _spec_document() -> dict:
    return {
        "openapi": "3.0.0",
        "paths": {
            "/chat/completions": {
                "post": {
                    "operationId": "createChatCompletion",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ChatRequest"}
                            }
                        }
                    },
                }
            },
            "/embeddings": {
                "post": {
                    "operationId": "createEmbedding",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EmbeddingRequest"}
                            }
                        }
                    },
                }
            },
            "/images/generations": {
                "post": {
                    "operationId": "createImage",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ImageRequest"}
                            }
                        }
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "ChatRequest": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}, "messages": {"type": "array"}},
                },
                "EmbeddingRequest": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "model": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"enum": ["text-embedding-3-small"]},
                                    ]
                                }
                            },
                        }
                    ]
                },
                "ImageRequest": {
                    "type": "object",
                    "properties": {
                        "model": {"enum": ["gpt-image-1"]},
                        "prompt": {"type": "string"},
                    },
                },
            }
        },
    }


def _anthropic_spec_document() -> dict:
    return {
        "openapi": "3.0.0",
        "paths": {
            "/v1/messages": {
                "post": {
                    "operationId": "messages_post",
                    "summary": "Create a Message",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {}}
                            }
                        }
                    },
                }
            },
            "/v1/complete": {
                "post": {
                    "operationId": "complete_post",
                    "summary": "Create a Text Completion",
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_openapi_model_enum_selects_embedding_and_is_cached() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, content=json.dumps(_spec_document()))

    registry = OpenAPISpecRegistry(
        ttl_seconds=60,
        transport=httpx.MockTransport(handler),
    )
    first = await registry.plan_model_test(
        model="openai/text-embedding-3-small",
        preferred_operation="chat_completion",
        spec_url="https://provider.example/openapi.json",
    )
    second = await registry.plan_model_test(
        model="openai/gpt-4o-mini",
        preferred_operation="chat_completion",
        spec_url="https://provider.example/openapi.json",
    )
    image = await registry.plan_model_test(
        model="openai/chatgpt-image-latest",
        preferred_operation="image_generation",
        fallback_source="litellm",
        spec_url="https://provider.example/openapi.json",
    )

    assert first.operation == "embedding"
    assert first.source == "openapi"
    assert first.operation_id == "createEmbedding"
    assert second.operation == "chat_completion"
    assert second.operation_id == "createChatCompletion"
    assert image.operation == "image_generation"
    assert image.source == "openapi"
    assert image.operation_id == "createImage"
    assert requests == 1


@pytest.mark.asyncio
async def test_unavailable_openapi_spec_falls_back_to_preferred_operation() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    registry = OpenAPISpecRegistry(transport=httpx.MockTransport(handler))
    plan = await registry.plan_model_test(
        model="anthropic/claude-sonnet-4-6",
        preferred_operation="chat_completion",
        spec_url="https://provider.example/openapi.json",
    )

    assert plan.operation == "chat_completion"
    assert plan.source == "heuristic"
    assert plan.operation_id is None


@pytest.mark.asyncio
async def test_anthropic_messages_operation_is_chat_completion() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(_anthropic_spec_document()))

    registry = OpenAPISpecRegistry(transport=httpx.MockTransport(handler))
    plan = await registry.plan_model_test(
        model="anthropic/claude-sonnet-4-6",
        preferred_operation="chat_completion",
        spec_url="https://provider.example/openapi.yml",
    )

    assert plan.operation == "chat_completion"
    assert plan.source == "openapi"
    assert plan.operation_id == "messages_post"


def test_github_blob_url_is_normalized_to_raw_content() -> None:
    assert (
        normalize_openapi_url("https://github.com/openai/openai-openapi/blob/main/openapi.yaml")
        == "https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.yaml"
    )
