from __future__ import annotations

import io
import json
import logging
import secrets
import time
import uuid
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import litellm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.backend import BackendClient, BackendError, GatewayContext, ModelAnalysisContext
from app.config import settings
from app.ingest import SupabaseGatewayIngestor
from app.openapi_specs import (
    TestOperation,
    openapi_spec_registry,
    provider_openapi_spec_url,
)

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


@app.post("/internal/provider-model-test", response_model=None)
async def provider_model_test(
    request: Request,
    x_proxy_secret: str | None = Header(default=None, alias="X-Proxy-Secret"),
) -> JSONResponse:
    """Run a minimal provider call without creating a captured session."""
    if not x_proxy_secret or not secrets.compare_digest(
        x_proxy_secret, settings.proxy_shared_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid proxy secret")

    payload = await _read_json_object(request)
    provider = payload.get("provider")
    model = payload.get("model")
    operation = payload.get("operation")
    provider_config = payload.get("provider_config")
    if not isinstance(provider, str) or not provider.strip():
        raise HTTPException(status_code=422, detail="provider is required")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=422, detail="model is required")
    if operation not in {
        "chat_completion",
        "responses",
        "embedding",
        "image_generation",
        "audio_speech",
        "audio_transcription",
        "moderation",
        "text_completion",
        "video_generation",
    }:
        raise HTTPException(status_code=422, detail="Unsupported model test operation")
    if not isinstance(provider_config, dict):
        raise HTTPException(status_code=422, detail="provider_config must be an object")

    context = ModelAnalysisContext(
        org_id="provider-model-test",
        org_slug="provider-model-test",
        provider=provider,
        provider_config=provider_config,
    )
    preferred_operation, fallback_source = _model_test_operation(model, operation)
    plan = await openapi_spec_registry.plan_model_test(
        model=model,
        preferred_operation=preferred_operation,
        spec_url=provider_openapi_spec_url(provider),
        fallback_source=fallback_source,
    )
    operation = plan.operation
    started = time.perf_counter()
    try:
        if operation == "responses":
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "input": "Reply OK.",
                    "max_output_tokens": 1,
                    "stream": False,
                },
            )
            response = await litellm.aresponses(**kwargs)
            response_preview = _responses_test_preview(_to_plain_data(response))
        elif operation == "embedding":
            kwargs = _build_litellm_kwargs(
                context,
                {"model": model, "input": ["test"]},
            )
            response = await litellm.aembedding(**kwargs)
            response_preview = _embedding_test_preview(_to_plain_data(response))
        elif operation == "image_generation":
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "prompt": "A solid blue circle on a white background.",
                    "n": 1,
                },
            )
            response = await litellm.aimage_generation(**kwargs)
            response_preview = _image_test_preview(_to_plain_data(response))
        elif operation == "audio_speech":
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "input": "Test.",
                    "voice": "alloy",
                    "response_format": "mp3",
                },
            )
            response = await litellm.aspeech(**kwargs)
            response_preview = _binary_test_preview(response, "Audio")
        elif operation == "audio_transcription":
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "file": ("test.wav", _silent_wav_bytes(), "audio/wav"),
                },
            )
            response = await litellm.atranscription(**kwargs)
            response_preview = _transcription_test_preview(_to_plain_data(response))
        elif operation == "moderation":
            kwargs = _build_litellm_kwargs(
                context,
                {"model": model, "input": "Hello."},
            )
            response = await litellm.amoderation(**kwargs)
            response_preview = _moderation_test_preview(_to_plain_data(response))
        elif operation == "text_completion":
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "prompt": "Reply OK.",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            response = await litellm.atext_completion(**kwargs)
            response_preview = _text_completion_test_preview(_to_plain_data(response))
        elif operation == "video_generation":
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "prompt": "A solid blue circle on a white background.",
                    "seconds": "4",
                },
            )
            response = await litellm.avideo_generation(**kwargs)
            response_preview = _video_test_preview(_to_plain_data(response))
        else:
            kwargs = _build_litellm_kwargs(
                context,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply 1."}],
                    "max_tokens": 100,
                    "stream": False,
                },
            )
            response = await litellm.acompletion(**kwargs)
            response_preview = _chat_test_preview(_to_plain_data(response))
    except Exception as exc:
        status_code = _provider_exception_status(exc)
        logger.warning(
            "Provider model test failed provider=%s model=%s operation=%s status=%s: %s",
            provider,
            model,
            operation,
            status_code,
            exc
        )
        status_suffix = f" (HTTP {status_code})" if status_code is not None else ""
        raise HTTPException(
            status_code=502,
            detail=f"Provider rejected the {operation.replace('_', ' ')} test{status_suffix}",
        ) from exc

    return JSONResponse(
        {
            "provider": provider,
            "model": model,
            "operation": operation,
            "spec_source": plan.source,
            "openapi_operation_id": plan.operation_id,
            "latency_ms": _duration_ms(started),
            "response_preview": response_preview,
        }
    )


@app.post("/internal/model-analysis/v1/chat/completions", response_model=None)
async def model_analysis_chat_completions(
    request: Request,
    x_proxy_secret: str | None = Header(default=None, alias="X-Proxy-Secret"),
) -> JSONResponse:
    """Execute trusted analysis traffic without creating captured sessions."""
    if not x_proxy_secret or not secrets.compare_digest(
        x_proxy_secret, settings.proxy_shared_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid proxy secret")
    envelope = await _read_json_object(request)
    org_id = envelope.get("org_id")
    run_id = envelope.get("run_id")
    result_id = envelope.get("result_id")
    purpose = envelope.get("purpose")
    payload = envelope.get("request")
    if not isinstance(org_id, str) or not org_id.strip():
        raise HTTPException(status_code=422, detail="org_id is required")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request must be an object")
    if not isinstance(run_id, str) or not run_id.strip():
        raise HTTPException(status_code=422, detail="run_id is required")
    if not isinstance(result_id, str) or not result_id.strip():
        raise HTTPException(status_code=422, detail="result_id is required")
    if purpose not in {"candidate", "judge"}:
        raise HTTPException(status_code=422, detail="purpose must be candidate or judge")
    model = _require_model(payload)
    if payload.get("stream") is True:
        raise HTTPException(status_code=422, detail="model analysis does not support streaming")
    if not isinstance(payload.get("messages"), list) or not payload["messages"]:
        raise HTTPException(status_code=422, detail="messages are required")

    backend: BackendClient = request.app.state.backend
    try:
        context = await backend.resolve_model_analysis(
            org_id=org_id,
            run_id=run_id,
            result_id=result_id,
            purpose=purpose,
            model=model,
        )
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, {**payload, "stream": False})
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        logger.exception("Model-analysis provider request failed")
        raise HTTPException(status_code=502, detail="Provider request failed") from exc
    response_body = _to_plain_data(response)
    return JSONResponse(
        {
            "response": response_body,
            "provider": context.provider,
            "usage": _extract_usage(response_body),
            "latency_ms": _duration_ms(started),
            "provider_request_id": response_body.get("id"),
            # Disabled until ReclaimLLM owns a complete, versioned model-price table.
            "cost_usd": None,
        }
    )


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
    config.pop("openapi_spec_url", None)
    if "api_base" not in config and config.get("base_url"):
        config["api_base"] = config["base_url"]
    return config


def _provider_model_name(model: str) -> str:
    """Return the model name LiteLLM expects for the configured provider."""
    normalized = model.strip()
    if normalized.startswith(("moonshot/", "deepseek/")):
        return normalized
    return normalized.rsplit("/", 1)[-1]


def _build_litellm_kwargs(
    context: GatewayContext | ModelAnalysisContext, payload: dict[str, Any]
) -> dict[str, Any]:
    provider_config = _normalise_provider_config(context.provider_config)
    kwargs = {
        key: value
        for key, value in payload.items()
        if key not in _BLOCKED_PAYLOAD_KEYS
        and not key.startswith(_BLOCKED_PAYLOAD_PREFIXES)
    }
    kwargs.update(provider_config)
    kwargs["model"] = _provider_model_name(str(payload["model"]))
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


def _chat_test_preview(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()[:160]
    return "Response received"


def _embedding_test_preview(response_body: dict[str, Any]) -> str:
    data = response_body.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        embedding = data[0].get("embedding")
        if isinstance(embedding, list):
            return f"Embedding returned {len(embedding)} dimensions"
    return "Embedding response received"


def _provider_exception_status(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _model_test_operation(
    model: str,
    requested: TestOperation,
) -> tuple[TestOperation, Literal["litellm", "heuristic"]]:
    mode_map: dict[str, TestOperation] = {
        "chat": "chat_completion",
        "responses": "responses",
        "embedding": "embedding",
        "image_generation": "image_generation",
        "audio_speech": "audio_speech",
        "audio_transcription": "audio_transcription",
        "moderation": "moderation",
        "completion": "text_completion",
        "video_generation": "video_generation",
    }
    try:
        mode = litellm.get_model_info(model).get("mode")
    except Exception:
        mode = None
    if isinstance(mode, str) and mode in mode_map:
        return mode_map[mode], "litellm"
    if isinstance(mode, str):
        raise HTTPException(
            status_code=422,
            detail=f"Model mode '{mode}' does not support a bounded credential test",
        )

    name = model.casefold().split("/", 1)[-1]
    if "embedding" in name:
        return "embedding", "heuristic"
    if "image" in name or name.startswith("dall-e"):
        return "image_generation", "heuristic"
    if "moderation" in name:
        return "moderation", "heuristic"
    if "transcri" in name or name.startswith("whisper"):
        return "audio_transcription", "heuristic"
    if "tts" in name or "speech" in name:
        return "audio_speech", "heuristic"
    if "instruct" in name or name in {"davinci-002", "babbage-002"}:
        return "text_completion", "heuristic"
    return requested, "heuristic"


def _responses_test_preview(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()[:160]
    return "Response received"


def _image_test_preview(response_body: dict[str, Any]) -> str:
    data = response_body.get("data")
    if isinstance(data, list):
        return f"Image generation returned {len(data)} image{'' if len(data) == 1 else 's'}"
    return "Image response received"


def _binary_test_preview(response: Any, label: str) -> str:
    if isinstance(response, bytes):
        return f"{label} response returned {len(response)} bytes"
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return f"{label} response returned {len(content)} bytes"
    return f"{label} response received"


def _silent_wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 800)
    return output.getvalue()


def _transcription_test_preview(response_body: dict[str, Any]) -> str:
    text = response_body.get("text")
    if isinstance(text, str):
        return text.strip()[:160] or "Transcription response received"
    return "Transcription response received"


def _moderation_test_preview(response_body: dict[str, Any]) -> str:
    results = response_body.get("results")
    if isinstance(results, list):
        return f"Moderation returned {len(results)} result{'' if len(results) == 1 else 's'}"
    return "Moderation response received"


def _text_completion_test_preview(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        text = choices[0].get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()[:160]
    return "Completion response received"


def _video_test_preview(response_body: dict[str, Any]) -> str:
    video_id = response_body.get("id")
    status = response_body.get("status")
    if video_id and status:
        return f"Video job {video_id} is {status}"
    if video_id:
        return f"Video job {video_id} created"
    return "Video generation request accepted"
