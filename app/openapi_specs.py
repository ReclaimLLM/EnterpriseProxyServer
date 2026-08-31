"""Bounded OpenAPI loading and model-test operation planning."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml


TestOperation = Literal[
    "chat_completion",
    "responses",
    "embedding",
    "image_generation",
    "audio_speech",
    "audio_transcription",
    "moderation",
    "text_completion",
    "video_generation",
]
_MAX_SPEC_BYTES = 5 * 1024 * 1024
_DEFAULT_TTL_SECONDS = 6 * 60 * 60
_FAILED_SPEC_TTL_SECONDS = 5 * 60
_MAX_CACHE_ENTRIES = 128
_PROVIDER_OPENAPI_SPEC_URLS = {
    "openai": "https://github.com/openai/openai-openapi/blob/main/openapi.yaml",
    "azure_openai": (
        "https://github.com/Azure/azure-rest-api-specs/blob/main/specification/"
        "cognitiveservices/data-plane/AzureOpenAI/inference/stable/2024-10-21/inference.json"
    ),
    "anthropic": (
        "https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic/"
        "anthropic-893a61e9c1cd6c69a70d1043c626ed02d12d6a492eb6ca6ef7a84c64cfb15393.yml"
    ),
    "moonshot": "https://platform.kimi.ai/docs/openapi.json",
}


class OpenAPISpecError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAPIOperation:
    operation: TestOperation
    operation_id: str | None
    model_ids: frozenset[str]


@dataclass(frozen=True)
class ModelTestPlan:
    operation: TestOperation
    source: Literal["openapi", "litellm", "heuristic"]
    operation_id: str | None = None


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    operations: tuple[OpenAPIOperation, ...]


class OpenAPISpecRegistry:
    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._transport = transport
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def plan_model_test(
        self,
        *,
        model: str,
        preferred_operation: TestOperation,
        spec_url: str | None,
        fallback_source: Literal["litellm", "heuristic"] = "heuristic",
    ) -> ModelTestPlan:
        if not spec_url:
            return ModelTestPlan(operation=preferred_operation, source=fallback_source)
        try:
            operations = await self._operations_for(spec_url)
        except OpenAPISpecError:
            return ModelTestPlan(operation=preferred_operation, source=fallback_source)
        if not operations:
            return ModelTestPlan(operation=preferred_operation, source=fallback_source)

        configured_name = model.split("/", 1)[-1]
        exact = [item for item in operations if configured_name in item.model_ids]
        exact_kinds = {item.operation for item in exact}
        if len(exact_kinds) == 1:
            selected = next(item for item in exact if item.operation in exact_kinds)
            return ModelTestPlan(
                operation=selected.operation,
                source="openapi",
                operation_id=selected.operation_id,
            )

        preferred = next(
            (item for item in operations if item.operation == preferred_operation),
            None,
        )
        if preferred is not None:
            return ModelTestPlan(
                operation=preferred.operation,
                source="openapi",
                operation_id=preferred.operation_id,
            )

        available_kinds = {item.operation for item in operations}
        if len(available_kinds) == 1:
            selected = operations[0]
            return ModelTestPlan(
                operation=selected.operation,
                source="openapi",
                operation_id=selected.operation_id,
            )
        return ModelTestPlan(operation=preferred_operation, source=fallback_source)

    async def _operations_for(self, spec_url: str) -> tuple[OpenAPIOperation, ...]:
        normalized_url = normalize_openapi_url(spec_url)
        now = time.monotonic()
        cached = self._cache.get(normalized_url)
        if cached and cached.expires_at > now:
            return cached.operations
        async with self._lock:
            cached = self._cache.get(normalized_url)
            if cached and cached.expires_at > time.monotonic():
                return cached.operations
            try:
                operations = await self._fetch_operations(normalized_url)
                ttl_seconds = self._ttl_seconds
            except OpenAPISpecError:
                operations = ()
                ttl_seconds = _FAILED_SPEC_TTL_SECONDS
            self._evict_cache_entries()
            self._cache[normalized_url] = _CacheEntry(
                expires_at=time.monotonic() + ttl_seconds,
                operations=operations,
            )
            return operations

    async def _fetch_operations(self, spec_url: str) -> tuple[OpenAPIOperation, ...]:
        try:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    "GET",
                    spec_url,
                    headers={"Accept": "application/json, application/yaml, text/yaml"},
                ) as response:
                    if response.status_code != 200:
                        raise OpenAPISpecError("OpenAPI spec request failed")
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > _MAX_SPEC_BYTES:
                        raise OpenAPISpecError("OpenAPI spec is too large")
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > _MAX_SPEC_BYTES:
                            raise OpenAPISpecError("OpenAPI spec is too large")
        except httpx.HTTPError as exc:
            raise OpenAPISpecError("OpenAPI spec request failed") from exc
        except ValueError as exc:
            raise OpenAPISpecError("OpenAPI spec has invalid content length") from exc
        content = bytes(chunks)
        try:
            if content.lstrip().startswith(b"{"):
                document = json.loads(content)
            else:
                document = yaml.safe_load(content)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise OpenAPISpecError("OpenAPI spec is invalid") from exc
        if not isinstance(document, dict) or not isinstance(document.get("paths"), dict):
            raise OpenAPISpecError("OpenAPI document has no paths")
        return extract_model_test_operations(document)

    def _evict_cache_entries(self) -> None:
        now = time.monotonic()
        self._cache = {
            url: entry for url, entry in self._cache.items() if entry.expires_at > now
        }
        if len(self._cache) >= _MAX_CACHE_ENTRIES:
            oldest_url = min(self._cache, key=lambda url: self._cache[url].expires_at)
            self._cache.pop(oldest_url, None)


def provider_openapi_spec_url(provider: str) -> str | None:
    """Return the proxy-owned OpenAPI source for a supported provider."""
    return _PROVIDER_OPENAPI_SPEC_URLS.get(provider)


def normalize_openapi_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise OpenAPISpecError("OpenAPI spec URL must be HTTP(S)")
    if parsed.username or parsed.password:
        raise OpenAPISpecError("OpenAPI spec URL must not contain credentials")
    if parsed.hostname == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repository, _, branch, *path_parts = parts
            return urlunsplit(
                (
                    "https",
                    "raw.githubusercontent.com",
                    f"/{owner}/{repository}/{branch}/{'/'.join(path_parts)}",
                    "",
                    "",
                )
            )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def extract_model_test_operations(document: dict[str, Any]) -> tuple[OpenAPIOperation, ...]:
    operations: list[OpenAPIOperation] = []
    for path, path_item in document.get("paths", {}).items():
        if not isinstance(path, str) or not isinstance(path_item, dict):
            continue
        operation = path_item.get("post")
        if not isinstance(operation, dict):
            continue
        operation_id = operation.get("operationId")
        marker = " ".join(
            [
                path,
                str(operation_id or ""),
                str(operation.get("summary") or ""),
                " ".join(str(tag) for tag in operation.get("tags") or []),
            ]
        ).casefold()
        kind = _test_operation_kind(path, marker)
        if kind is None:
            continue
        operations.append(
            OpenAPIOperation(
                operation=kind,
                operation_id=str(operation_id) if operation_id else None,
                model_ids=frozenset(_operation_model_ids(document, operation)),
            )
        )
    return tuple(operations)


def _test_operation_kind(path: str, marker: str) -> TestOperation | None:
    normalized_path = path.casefold().rstrip("/")
    if normalized_path.endswith("/images/generations") or "imagegenerations_create" in marker:
        return "image_generation"
    if normalized_path.endswith("/videos") or "createvideo" in marker:
        return "video_generation"
    if normalized_path.endswith("/audio/speech") or "createspeech" in marker:
        return "audio_speech"
    if normalized_path.endswith("/audio/transcriptions") or "transcriptions_create" in marker:
        return "audio_transcription"
    if normalized_path.endswith("/moderations") or "createmoderation" in marker:
        return "moderation"
    if normalized_path.endswith("/embeddings") or "embedding" in marker:
        return "embedding"
    if normalized_path.endswith("/responses") or "createresponse" in marker:
        return "responses"
    if normalized_path.endswith("/messages") or "messages_post" in marker:
        return "chat_completion"
    if normalized_path.endswith("/chat/completions") or "chatcompletion" in marker:
        return "chat_completion"
    if normalized_path.endswith("/complete") or "complete_post" in marker:
        return "text_completion"
    if normalized_path.endswith("/completions") or "completions_create" in marker:
        return "text_completion"
    return None


def _operation_model_ids(document: dict[str, Any], operation: dict[str, Any]) -> set[str]:
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return set()
    content = request_body.get("content")
    if not isinstance(content, dict):
        return set()
    media = content.get("application/json") or content.get("multipart/form-data")
    if not isinstance(media, dict):
        return set()
    schema = _resolve_local_ref(document, media.get("schema"))
    if not isinstance(schema, dict):
        return set()
    return _model_ids_from_schema(document, schema, set())


def _model_ids_from_schema(
    document: dict[str, Any],
    schema: dict[str, Any],
    seen_refs: set[str],
) -> set[str]:
    model_ids: set[str] = set()
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            return model_ids
        resolved = _resolve_local_ref(document, schema)
        if isinstance(resolved, dict):
            return _model_ids_from_schema(document, resolved, seen_refs | {ref})
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(properties.get("model"), dict):
        model_ids.update(_enum_values(document, properties["model"], seen_refs))
    for keyword in ("allOf", "anyOf", "oneOf"):
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    model_ids.update(_model_ids_from_schema(document, child, seen_refs))
    return model_ids


def _enum_values(document: dict[str, Any], schema: dict[str, Any], seen_refs: set[str]) -> set[str]:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            return set()
        resolved = _resolve_local_ref(document, schema)
        if isinstance(resolved, dict):
            return _enum_values(document, resolved, seen_refs | {ref})
    values = schema.get("enum")
    out = {str(value) for value in values} if isinstance(values, list) else set()
    for keyword in ("allOf", "anyOf", "oneOf"):
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    out.update(_enum_values(document, child, seen_refs))
    return out


def _resolve_local_ref(document: dict[str, Any], value: Any) -> Any:
    if not isinstance(value, dict) or not isinstance(value.get("$ref"), str):
        return value
    ref = value["$ref"]
    if not ref.startswith("#/"):
        return value
    current: Any = document
    for part in ref[2:].split("/"):
        if not isinstance(current, dict):
            return value
        current = current.get(part.replace("~1", "/").replace("~0", "~"))
    return current


openapi_spec_registry = OpenAPISpecRegistry()
