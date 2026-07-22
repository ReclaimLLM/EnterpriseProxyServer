from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import litellm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.backend import BackendClient, BackendError, GatewayContext
from app.config import settings
from app.ingest import SupabaseGatewayIngestor

logger = logging.getLogger(__name__)

# Credential/routing kwargs must come only from the backend-resolved provider
# config, never from the client payload — otherwise a gateway-key holder could
# redirect the org's provider credentials to a server they control.
_BLOCKED_PAYLOAD_KEYS = {
    "api_key",
    "api_base",
    "base_url",
    "api_version",
    "organization",
    "azure_ad_token",
    "azure_ad_token_provider",
    "extra_headers",
    "headers",
    "custom_llm_provider",
    "client",
    "mock_response",
}
_BLOCKED_PAYLOAD_PREFIXES = ("aws_", "vertex_", "watsonx_", "azure_")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.backend = BackendClient()
    app.state.ingestor = SupabaseGatewayIngestor()
    try:
        yield
    finally:
        await app.state.backend.close()
        await app.state.ingestor.close()


app = FastAPI(
    title="ReclaimLLM Enterprise Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/{enterprise_slug}/v1/chat/completions", response_model=None)
async def chat_completions(
    enterprise_slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    payload = await _read_json_object(request)
    model = _require_model(payload)
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    context = await _resolve_context(
        backend=backend,
        org_slug=enterprise_slug,
        gateway_key=gateway_key,
        model=model,
    )

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, payload)

    if payload.get("stream") is True:
        return StreamingResponse(
            _stream_chat_completion(
                ingestor=request.app.state.ingestor,
                context=context,
                payload=payload,
                kwargs=kwargs,
                started=started,
                request_url=str(request.url),
            ),
            media_type="text/event-stream",
        )

    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        logger.exception("Provider request failed")
        raise HTTPException(
            status_code=502, detail="Provider request failed"
        ) from exc

    response_body = _to_plain_data(response)
    await _ingest_or_queue(
        request.app.state.ingestor,
        context,
        _build_record(
            payload=payload,
            response_body=response_body,
            status_code=200,
            duration_ms=_duration_ms(started),
            request_url=str(request.url),
        ),
    )
    return JSONResponse(response_body)


@app.post("/{enterprise_slug}/v1/responses", response_model=None)
async def responses(
    enterprise_slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    responses_call = getattr(litellm, "aresponses", None)
    if responses_call is None:
        raise HTTPException(
            status_code=501,
            detail="LiteLLM responses API is not available in this runtime",
        )

    payload = await _read_json_object(request)
    model = _require_model(payload)
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    context = await _resolve_context(
        backend=backend,
        org_slug=enterprise_slug,
        gateway_key=gateway_key,
        model=model,
    )

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, payload)
    try:
        response = await responses_call(**kwargs)
    except Exception as exc:
        logger.exception("Provider request failed")
        raise HTTPException(
            status_code=502, detail="Provider request failed"
        ) from exc

    response_body = _to_plain_data(response)
    await _ingest_or_queue(
        request.app.state.ingestor,
        context,
        _build_record(
            payload=payload,
            response_body=response_body,
            status_code=200,
            duration_ms=_duration_ms(started),
            request_url=str(request.url),
        ),
    )
    return JSONResponse(response_body)


@app.post("/{enterprise_slug}/v1/embeddings", response_model=None)
async def embeddings(
    enterprise_slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    payload = await _read_json_object(request)
    model = _require_model(payload)
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    context = await _resolve_context(
        backend=backend,
        org_slug=enterprise_slug,
        gateway_key=gateway_key,
        model=model,
    )

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, payload)
    try:
        response = await litellm.aembedding(**kwargs)
    except Exception as exc:
        logger.exception("Provider request failed")
        raise HTTPException(
            status_code=502, detail="Provider request failed"
        ) from exc

    response_body = _to_plain_data(response)
    await _ingest_or_queue(
        request.app.state.ingestor,
        context,
        _build_record(
            payload=payload,
            response_body=response_body,
            status_code=200,
            duration_ms=_duration_ms(started),
            request_url=str(request.url),
        ),
    )
    return JSONResponse(response_body)


async def _stream_chat_completion(
    *,
    ingestor: SupabaseGatewayIngestor,
    context: GatewayContext,
    payload: dict[str, Any],
    kwargs: dict[str, Any],
    started: float,
    request_url: str,
) -> AsyncIterator[bytes]:
    chunks: list[dict[str, Any]] = []
    try:
        stream = await litellm.acompletion(**kwargs)
        async for chunk in stream:
            chunk_data = _to_plain_data(chunk)
            chunks.append(chunk_data)
            yield f"data: {json.dumps(chunk_data, separators=(',', ':'))}\n\n".encode()
        yield b"data: [DONE]\n\n"
        await _ingest_or_queue(
            ingestor,
            context,
            _build_record(
                payload=payload,
                response_body={"chunks": chunks},
                status_code=200,
                duration_ms=_duration_ms(started),
                request_url=request_url,
                is_streaming=True,
            ),
        )
    except Exception as exc:
        logger.exception("Provider request failed during stream")
        client_error = {
            "error": {"message": "Provider request failed", "type": "provider_error"}
        }
        yield f"data: {json.dumps(client_error, separators=(',', ':'))}\n\n".encode()
        await _ingest_or_queue(
            ingestor,
            context,
            _build_record(
                payload=payload,
                response_body={
                    "error": {"message": str(exc), "type": "provider_error"}
                },
                status_code=502,
                duration_ms=_duration_ms(started),
                request_url=request_url,
                is_streaming=True,
            ),
        )


async def _resolve_context(
    *,
    backend: BackendClient,
    org_slug: str,
    gateway_key: str,
    model: str,
) -> GatewayContext:
    try:
        return await backend.resolve_gateway_key(
            org_slug=org_slug,
            bearer_token=gateway_key,
            model=model,
        )
    except BackendError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.message
        ) from exc


def _normalise_provider_config(provider_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(provider_config)
    if "api_base" not in config and config.get("base_url"):
        config["api_base"] = config["base_url"]
    return config


def _build_litellm_kwargs(
    context: GatewayContext, payload: dict[str, Any]
) -> dict[str, Any]:
    provider_config = _normalise_provider_config(context.provider_config)
    kwargs = {
        key: value
        for key, value in payload.items()
        if key not in _BLOCKED_PAYLOAD_KEYS
        and not key.startswith(_BLOCKED_PAYLOAD_PREFIXES)
    }
    kwargs.update(provider_config)
    kwargs["model"] = payload["model"]
    if "service_tier" in kwargs:
        del kwargs["service_tier"]
    return kwargs




def _build_record(
    *,
    payload: dict[str, Any],
    response_body: dict[str, Any],
    status_code: int,
    duration_ms: int,
    request_url: str,
    is_streaming: bool | None = None,
) -> dict[str, Any]:
    usage = _extract_usage(response_body)
    return {
        "session_id": str(uuid.uuid4()),
        "method": "POST",
        "url": request_url,
        "response_status": status_code,
        "is_streaming": (
            bool(payload.get("stream"))
            if is_streaming is None
            else is_streaming
        ),
        "duration_ms": duration_ms,
        "model": payload.get("model"),
        "total_input_tokens": usage.get("prompt_tokens"),
        "total_output_tokens": usage.get("completion_tokens"),
        "request": _redact_payload(payload),
        "response": response_body,
    }


async def _ingest_or_queue(
    ingestor: SupabaseGatewayIngestor,
    context: GatewayContext,
    record: dict[str, Any],
) -> None:
    try:
        await ingestor.ingest_gateway_record(context=context, record=record)
    except Exception as exc:
        await _append_failed_log(context=context, record=record, error=str(exc))


async def _append_failed_log(
    *,
    context: GatewayContext,
    record: dict[str, Any],
    error: str,
) -> None:
    path = Path(settings.log_queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    failed_record = {
        "error": error,
        "org_id": context.org_id,
        "team_id": context.team_id,
        "user_id": context.user_id,
        "key_id": context.key_id,
        "record": record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failed_record, separators=(",", ":")) + "\n")


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Authorization bearer key required"
        )
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(
            status_code=401, detail="Authorization bearer key required"
        )
    return token


async def _read_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object"
        )
    return payload


def _require_model(payload: dict[str, Any]) -> str:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=422, detail="model is required")
    return model


def _to_plain_data(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value, default=str))


def _extract_usage(response_body: dict[str, Any]) -> dict[str, Any]:
    usage = response_body.get("usage")
    if isinstance(usage, dict):
        return usage
    return {}


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    for key in ("api_key", "authorization", "headers"):
        redacted.pop(key, None)
    return redacted


def _duration_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
