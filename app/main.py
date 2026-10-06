from __future__ import annotations

import io
import json
import logging
import re
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
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.analytics import shutdown as analytics_shutdown, track_litellm_error
from app.backend import BackendClient, BackendError, GatewayContext, ModelAnalysisContext
from app.config import settings
from app.ingest import SupabaseGatewayIngestor
from app.openapi_specs import (
    TestOperation,
    openapi_spec_registry,
    provider_openapi_spec_url,
)

logger = logging.getLogger(__name__)
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.drop_params = True


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

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
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, EndpointFilter) for f in access_logger.filters):
        access_logger.addFilter(EndpointFilter())
    for handler in access_logger.handlers:
        if not any(isinstance(f, EndpointFilter) for f in handler.filters):
            handler.addFilter(EndpointFilter())

    app.state.backend = BackendClient()
    app.state.ingestor = SupabaseGatewayIngestor()
    try:
        yield
    finally:
        await app.state.backend.close()
        await app.state.ingestor.close()
        analytics_shutdown()


app = FastAPI(
    title="ReclaimLLM Enterprise Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


def _is_no_credits_error(exc: Exception) -> bool:
    """Check whether an exception represents an exhausted credit/quota balance."""
    code = getattr(exc, "code", None)
    if code == "credit_balance_exhausted":
        return True
    error_type = getattr(exc, "type", None)
    if error_type == "insufficient_quota":
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_dict = body.get("error")
        if isinstance(error_dict, dict):
            if (
                error_dict.get("code") == "credit_balance_exhausted"
                or error_dict.get("type") == "insufficient_quota"
            ):
                return True
            msg = str(error_dict.get("message", "")).lower()
            if "no credits remaining" in msg or "insufficient_quota" in msg:
                return True

    err_str = f"{exc} {getattr(exc, 'message', '')}".lower()
    return (
        "credit_balance_exhausted" in err_str
        or "insufficient_quota" in err_str
        or "no credits remaining" in err_str
    )


def _no_credits_response(exc: Exception | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": (
                    "You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/."
                ),
                "type": "insufficient_quota",
                "param": None,
                "code": "credit_balance_exhausted",
            },
            "detail": "Provider credit balance exhausted: You have no credits remaining.",
        },
    )


@app.exception_handler(litellm.RateLimitError)
async def litellm_rate_limit_error_handler(
    request: Request, exc: litellm.RateLimitError
) -> JSONResponse:
    context = getattr(request.state, "gateway_context", None)
    model = getattr(request.state, "model", None) or getattr(exc, "model", None)
    org_slug = getattr(context, "org_slug", None)
    provider = getattr(context, "provider", None) or getattr(exc, "llm_provider", None)

    if _is_no_credits_error(exc):
        logger.error(
            "Provider credit balance exhausted (no credits remaining) provider=%s model=%s org=%s: %s",
            provider,
            model,
            org_slug,
            exc,
        )
        return _no_credits_response(exc)

    logger.warning(
        "Provider rate limit exceeded provider=%s model=%s org=%s: %s",
        provider,
        model,
        org_slug,
        exc,
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": str(exc),
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            },
            "detail": "Provider rate limit exceeded",
        },
    )


def _provider_exception_status(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _sanitize_provider_error(exc: Exception) -> dict[str, Any]:
    raw_str = f"{exc} {getattr(exc, 'message', '')}"
    raw_lower = raw_str.lower()
    status_code = _provider_exception_status(exc) or 502

    # 1. Cloudflare 524 Timeout
    if "524" in raw_str and any(
        kw in raw_lower for kw in ("timeout", "cloudflare", "a timeout occurred")
    ):
        return {
            "status_code": 504,
            "message": (
                "Upstream provider timed out (Cloudflare Error 524). "
                "The upstream server took longer than 100 seconds to respond."
            ),
            "type": "gateway_timeout",
            "code": "upstream_timeout",
            "is_html": True,
        }

    # 2. Cloudflare Challenge / Bot Fight Mode (403)
    is_cf_challenge = (
        (
            "cf-mitigated" in raw_lower
            or "just a moment..." in raw_lower
            or "cloudflare ray id" in raw_lower
            or "turnstile" in raw_lower
        )
        and (
            "challenge" in raw_lower
            or "403" in raw_str
            or "forbidden" in raw_lower
            or status_code == 403
        )
    )
    if is_cf_challenge:
        return {
            "status_code": 502,
            "message": (
                "Upstream provider blocked the request with a Cloudflare challenge (HTTP 403). "
                "Configure the upstream Cloudflare WAF/Bot settings to allow ReclaimLLM proxy requests."
            ),
            "type": "provider_error",
            "code": "upstream_cloudflare_challenge",
            "is_html": True,
        }

    # 3. HTML Error pages (e.g. nginx 502/504, Cloudflare 5xx)
    if "<!doctype html" in raw_lower or "<html" in raw_lower or "<head>" in raw_lower:
        title_match = re.search(r"<title>(.*?)</title>", raw_str, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = " ".join(title_match.group(1).split())[:120]
            message = f"Upstream provider returned an HTML error page: {title}"
        else:
            message = "Upstream provider returned an HTML error response instead of JSON."
        return {
            "status_code": 502,
            "message": message,
            "type": "provider_error",
            "code": "upstream_html_error",
            "is_html": True,
        }

    # 4. Connection / read timeouts
    if any(kw in raw_lower for kw in ("timeout", "timed out", "readtimedout", "connecttimeout")):
        return {
            "status_code": 504,
            "message": f"Upstream provider request timed out: {str(exc)[:200]}",
            "type": "gateway_timeout",
            "code": "upstream_timeout",
            "is_html": False,
        }

    # 5. Default provider error
    clean_msg = str(exc).strip()
    if len(clean_msg) > 300:
        clean_msg = clean_msg[:297] + "..."
    prefix = "" if clean_msg.startswith("Provider request failed") else "Provider request failed: "
    return {
        "status_code": 502,
        "message": f"{prefix}{clean_msg}",
        "type": "provider_error",
        "code": "provider_error",
        "is_html": False,
    }


def _handle_gateway_provider_error(
    exc: Exception,
    *,
    operation: str,
    model: str,
    context: GatewayContext,
    route: str | None = None,
) -> JSONResponse:
    route_str = f" route={route}" if route else ""
    if _is_no_credits_error(exc):
        track_litellm_error(
            exc,
            operation=operation,
            model=model,
            provider=context.provider,
            status_code=429,
            org_id=context.org_id,
            org_slug=context.org_slug,
            user_id=context.user_id,
            team_id=context.team_id,
            tag="credit_balance_exhausted",
        )
        logger.error(
            "Provider credit balance exhausted (no credits remaining) provider=%s model=%s org=%s%s: %s",
            context.provider,
            model,
            context.org_slug,
            route_str,
            exc,
        )
        return _no_credits_response(exc)

    sanitized = _sanitize_provider_error(exc)
    status_code = _provider_exception_status(exc)
    track_litellm_error(
        exc,
        operation=operation,
        model=model,
        provider=context.provider,
        status_code=status_code,
        org_id=context.org_id,
        org_slug=context.org_slug,
        user_id=context.user_id,
        team_id=context.team_id,
        tag="litellm_exception",
    )
    if sanitized["is_html"]:
        logger.error(
            "Provider request failed provider=%s model=%s operation=%s org=%s%s: %s",
            context.provider,
            model,
            operation,
            context.org_slug,
            route_str,
            sanitized["message"],
        )
    else:
        logger.exception(
            "Provider request failed provider=%s model=%s operation=%s org=%s%s: %s",
            context.provider,
            model,
            operation,
            context.org_slug,
            route_str,
            exc,
        )

    return JSONResponse(
        status_code=sanitized["status_code"],
        content={
            "error": {
                "message": sanitized["message"],
                "type": sanitized["type"],
                "param": None,
                "code": sanitized["code"],
            },
            "detail": sanitized["message"],
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
@app.head("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "ReclaimLLM Enterprise Proxy"}


@app.get("/robots.txt")
async def robots_txt() -> PlainTextResponse:
    return PlainTextResponse("User-agent: *\nDisallow: /\n")


@app.get("/{enterprise_slug}/v1")
@app.head("/{enterprise_slug}/v1")
@app.get("/{enterprise_slug}")
@app.head("/{enterprise_slug}")
async def enterprise_ping(enterprise_slug: str) -> dict[str, str]:
    return {"status": "ok", "organization": enterprise_slug}


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
    request.state.model = model
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
    mode = payload.get("mode") if isinstance(payload.get("mode"), str) else None
    preferred_operation, fallback_source = _model_test_operation(model, operation, mode=mode)
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
                    "stream": False,
                },
            )
            kwargs.pop("base_url", None)
            kwargs.pop("max_output_tokens", None)
            if context.provider == "azure_openai" or str(model).strip().startswith("azure/"):
                kwargs.pop("api_version", None)
            response = await litellm.aresponses(**kwargs)
            response_preview = _responses_test_preview(_to_plain_data(response))
        elif operation == "embedding":
            kwargs = _build_litellm_kwargs(
                context,
                {"model": model, "input": ["test"], "drop_params": True},
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
    except (litellm.RateLimitError, Exception) as exc:
        if _is_no_credits_error(exc):
            track_litellm_error(
                exc,
                operation=operation,
                model=model,
                provider=provider,
                status_code=429,
                tag="credit_balance_exhausted",
            )
            logger.error(
                "Provider model test failed: no credits remaining provider=%s model=%s operation=%s: %s",
                provider,
                model,
                operation,
                exc,
            )
            return _no_credits_response(exc)

        status_code = _provider_exception_status(exc)
        track_litellm_error(
            exc,
            operation=operation,
            model=model,
            provider=provider,
            status_code=status_code,
            tag="litellm_exception",
        )
        logger.warning(
            "Provider model test failed provider=%s model=%s operation=%s status=%s: %s",
            provider,
            model,
            operation,
            status_code,
            exc,
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
    request.state.model = model
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
        logger.error(
            "Model-analysis backend resolution failed model=%s org_id=%s: %s",
            model,
            org_id,
            exc.message,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    request.state.gateway_context = context

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, {**payload, "stream": False})
    try:
        response = await litellm.acompletion(**kwargs)
    except (litellm.RateLimitError, Exception) as exc:
        if _is_no_credits_error(exc):
            track_litellm_error(
                exc,
                operation="model_analysis_chat_completions",
                model=model,
                provider=context.provider,
                status_code=429,
                org_id=context.org_id,
                tag="credit_balance_exhausted",
                extra={"purpose": purpose, "run_id": run_id, "result_id": result_id},
            )
            logger.error(
                "Model-analysis provider credit balance exhausted (no credits remaining) provider=%s model=%s org=%s: %s",
                context.provider,
                model,
                context.org_slug,
                exc,
            )
            return _no_credits_response(exc)

        status_code = _provider_exception_status(exc)
        track_litellm_error(
            exc,
            operation="model_analysis_chat_completions",
            model=model,
            provider=context.provider,
            status_code=status_code,
            org_id=context.org_id,
            tag="litellm_exception",
            extra={"purpose": purpose, "run_id": run_id, "result_id": result_id},
        )
        sanitized = _sanitize_provider_error(exc)
        if sanitized["is_html"]:
            logger.error(
                "Model-analysis provider request failed provider=%s model=%s org=%s: %s",
                context.provider,
                model,
                context.org_slug,
                sanitized["message"],
            )
        else:
            logger.exception(
                "Model-analysis provider request failed provider=%s model=%s org=%s: %s",
                context.provider,
                model,
                context.org_slug,
                exc,
            )
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


@app.get("/{enterprise_slug}/v1/models")
@app.get("/{enterprise_slug}/models")
async def list_models(
    enterprise_slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    try:
        models = await backend.list_gateway_models(
            org_slug=enterprise_slug,
            bearer_token=gateway_key,
        )
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return JSONResponse(
        {
            "object": "list",
            "data": models,
        }
    )


@app.get("/{enterprise_slug}/v1/models/{model_id:path}")
@app.get("/{enterprise_slug}/models/{model_id:path}")
async def retrieve_model(
    enterprise_slug: str,
    model_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    try:
        models = await backend.list_gateway_models(
            org_slug=enterprise_slug,
            bearer_token=gateway_key,
        )
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    for item in models:
        if item.get("id") == model_id or item.get("model_name") == model_id:
            return JSONResponse(item)

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' not found or not authorized for this gateway key",
    )


@app.post("/{enterprise_slug}/v1/chat/completions", response_model=None)
@app.post("/{enterprise_slug}/chat/completions", response_model=None)
async def chat_completions(
    enterprise_slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    payload = await _read_json_object(request)
    model = _require_model(payload)
    request.state.model = model
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    context = await _resolve_context(
        backend=backend,
        org_slug=enterprise_slug,
        gateway_key=gateway_key,
        model=model,
    )
    request.state.gateway_context = context

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
                route=f"{request.method} {request.url.path}",
            ),
            media_type="text/event-stream",
        )

    try:
        response = await litellm.acompletion(**kwargs)
    except litellm.RateLimitError as exc:
        return _handle_gateway_provider_error(
            exc,
            operation="chat_completions",
            model=model,
            context=context,
            route=f"{request.method} {request.url.path}",
        )
    except Exception as exc:
        return _handle_gateway_provider_error(
            exc,
            operation="chat_completions",
            model=model,
            context=context,
            route=f"{request.method} {request.url.path}",
        )

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
@app.post("/{enterprise_slug}/responses", response_model=None)
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
    request.state.model = model
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    context = await _resolve_context(
        backend=backend,
        org_slug=enterprise_slug,
        gateway_key=gateway_key,
        model=model,
    )
    request.state.gateway_context = context

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, payload)
    kwargs.pop("base_url", None)
    kwargs.pop("max_output_tokens", None)
    if context.provider == "azure_openai" or str(model).strip().startswith("azure/"):
        kwargs.pop("api_version", None)
    try:
        response = await responses_call(**kwargs)
    except litellm.RateLimitError as exc:
        return _handle_gateway_provider_error(
            exc,
            operation="responses",
            model=model,
            context=context,
            route=f"{request.method} {request.url.path}",
        )
    except Exception as exc:
        return _handle_gateway_provider_error(
            exc,
            operation="responses",
            model=model,
            context=context,
            route=f"{request.method} {request.url.path}",
        )

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
@app.post("/{enterprise_slug}/embeddings", response_model=None)
async def embeddings(
    enterprise_slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    payload = await _read_json_object(request)
    model = _require_model(payload)
    request.state.model = model
    gateway_key = _extract_bearer_token(authorization)
    backend: BackendClient = request.app.state.backend
    context = await _resolve_context(
        backend=backend,
        org_slug=enterprise_slug,
        gateway_key=gateway_key,
        model=model,
    )
    request.state.gateway_context = context

    started = time.perf_counter()
    kwargs = _build_litellm_kwargs(context, payload)
    try:
        response = await litellm.aembedding(**kwargs)
    except litellm.RateLimitError as exc:
        return _handle_gateway_provider_error(
            exc,
            operation="embeddings",
            model=model,
            context=context,
            route=f"{request.method} {request.url.path}",
        )
    except Exception as exc:
        return _handle_gateway_provider_error(
            exc,
            operation="embeddings",
            model=model,
            context=context,
            route=f"{request.method} {request.url.path}",
        )

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
    route: str | None = None,
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
    except (litellm.RateLimitError, Exception) as exc:
        is_no_credits = _is_no_credits_error(exc)
        status_code = (
            _provider_exception_status(exc)
            or (429 if (is_no_credits or isinstance(exc, litellm.RateLimitError)) else None)
        )
        tag = "credit_balance_exhausted" if is_no_credits else "litellm_exception"
        track_litellm_error(
            exc,
            operation="chat_completions_stream",
            model=kwargs.get("model") or payload.get("model"),
            provider=context.provider,
            status_code=status_code,
            org_id=context.org_id,
            org_slug=context.org_slug,
            user_id=context.user_id,
            team_id=context.team_id,
            tag=tag,
        )
        route_str = f" route={route}" if route else ""
        if is_no_credits:
            logger.error(
                "Provider credit balance exhausted (no credits remaining) during stream provider=%s model=%s org=%s%s: %s",
                context.provider,
                kwargs.get("model") or payload.get("model"),
                context.org_slug,
                route_str,
                exc,
            )
            client_error = {
                "error": {
                    "message": "You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "credit_balance_exhausted",
                }
            }
        else:
            sanitized = _sanitize_provider_error(exc)
            if sanitized["is_html"]:
                logger.error(
                    "Provider request failed during stream provider=%s model=%s org=%s%s: %s",
                    context.provider,
                    kwargs.get("model") or payload.get("model"),
                    context.org_slug,
                    route_str,
                    sanitized["message"],
                )
            else:
                logger.exception(
                    "Provider request failed during stream provider=%s model=%s org=%s%s: %s",
                    context.provider,
                    kwargs.get("model") or payload.get("model"),
                    context.org_slug,
                    route_str,
                    exc,
                )
            client_error = {
                "error": {
                    "message": sanitized["message"],
                    "type": sanitized["type"],
                    "param": None,
                    "code": sanitized["code"],
                }
            }
        yield f"data: {json.dumps(client_error, separators=(',', ':'))}\n\n".encode()
        err_message = str(exc) if is_no_credits else sanitized["message"]
        err_type = "insufficient_quota" if is_no_credits else sanitized["type"]
        err_code = "credit_balance_exhausted" if is_no_credits else sanitized["code"]
        record_status = status_code if is_no_credits else sanitized["status_code"]
        await _ingest_or_queue(
            ingestor,
            context,
            _build_record(
                payload=payload,
                response_body={
                    "error": {
                        "message": err_message,
                        "type": err_type,
                        "code": err_code,
                    }
                },
                status_code=record_status,
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
        logger.error(
            "Gateway key resolution failed model=%s org=%s: %s",
            model,
            org_slug,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code, detail=exc.message
        ) from exc


def _normalise_provider_config(
    provider_config: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    config = dict(provider_config)
    config.pop("openapi_spec_url", None)
    is_openai = provider == "openai" or (
        isinstance(model, str) and model.strip().startswith("openai/")
    )
    if is_openai and not config.get("base_url") and not config.get("api_base"):
        config["base_url"] = "https://api.openai.com/v1"
    if "api_base" not in config and config.get("base_url"):
        config["api_base"] = config["base_url"]
    if "base_url" not in config and config.get("api_base"):
        config["base_url"] = config["api_base"]

    is_local = (
        provider in ("local", "litellm_proxy")
        or config.get("provider_type") == "local"
        or (
            isinstance(model, str)
            and model.strip().startswith(("local/", "litellm_proxy/"))
        )
        or (
            isinstance(provider, str)
            and provider not in ("openai", "azure_openai", "azure", "anthropic", "gemini", "moonshot", "deepseek")
        )
    )
    if is_local:
        config["custom_llm_provider"] = "openai"
        base_url = str(config.get("base_url") or config.get("api_base") or "").strip().rstrip("/")
        if base_url:
            if base_url.endswith("/models"):
                base_url = base_url[:-len("/models")].rstrip("/")
            elif not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            config["base_url"] = base_url
            config["api_base"] = base_url
    return config


def _provider_model_name(model: str, provider: str | None = None) -> str:
    """Return the model name LiteLLM expects for the configured provider."""
    normalized = model.strip()
    is_azure = provider in ("azure", "azure_openai") or normalized.startswith(
        ("azure/", "azure_openai/")
    )
    if is_azure:
        if normalized.startswith("azure/"):
            return normalized
        if normalized.startswith("azure_openai/"):
            return f"azure/{normalized.split('/', 1)[1]}"
        raw_name = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
        return f"azure/{raw_name}"
    if normalized.startswith(("moonshot/", "deepseek/")):
        return normalized
    if normalized.startswith(("local/", "litellm_proxy/")):
        return normalized.split("/", 1)[1]
    if provider and normalized.startswith(f"{provider}/"):
        return normalized[len(provider) + 1 :]
    return normalized.rsplit("/", 1)[-1]


def _build_litellm_kwargs(
    context: GatewayContext | ModelAnalysisContext, payload: dict[str, Any]
) -> dict[str, Any]:
    model_str = str(payload.get("model", ""))
    provider = getattr(context, "provider", None)
    provider_config = _normalise_provider_config(
        context.provider_config,
        provider=provider,
        model=model_str,
    )
    kwargs = {
        key: value
        for key, value in payload.items()
        if key not in _BLOCKED_PAYLOAD_KEYS
        and not key.startswith(_BLOCKED_PAYLOAD_PREFIXES)
    }
    kwargs.update(provider_config)
    kwargs["model"] = _provider_model_name(model_str, provider=provider)
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


def _coerce_token_count(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_usage_dict(usage: dict[str, Any]) -> dict[str, Any]:
    """Map Chat Completions and Responses API usage into a common shape.

    Chat Completions uses ``prompt_tokens`` / ``completion_tokens``.
    Responses API uses ``input_tokens`` / ``output_tokens``.
    """
    normalised = dict(usage)
    prompt = _coerce_token_count(
        _first_present(normalised.get("prompt_tokens"), normalised.get("input_tokens"))
    )
    completion = _coerce_token_count(
        _first_present(
            normalised.get("completion_tokens"), normalised.get("output_tokens")
        )
    )
    if prompt is not None:
        normalised["prompt_tokens"] = prompt
    if completion is not None:
        normalised["completion_tokens"] = completion
    return normalised


def _first_present(*values: object) -> object:
    return next((value for value in values if value is not None), None)


def _usage_from_chunks(chunks: object) -> dict[str, Any] | None:
    if not isinstance(chunks, list):
        return None
    # Prefer the last chunk that carries usage (OpenAI stream usage on final chunk).
    for chunk in reversed(chunks):
        if not isinstance(chunk, dict):
            continue
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            return usage
    return None


async def _ingest_or_queue(
    ingestor: SupabaseGatewayIngestor,
    context: GatewayContext,
    record: dict[str, Any],
) -> None:
    try:
        await ingestor.ingest_gateway_record(context=context, record=record)
    except Exception as exc:
        logger.warning(
            "Failed to ingest gateway record model=%s org=%s: %s",
            record.get("model"),
            context.org_slug,
            exc,
        )
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
    if not isinstance(usage, dict) or not usage:
        usage = _usage_from_chunks(response_body.get("chunks")) or {}
    if isinstance(usage, dict) and usage:
        return _normalise_usage_dict(usage)
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



def _model_test_operation(
    model: str,
    requested: TestOperation,
    mode: str | None = None,
) -> tuple[TestOperation, Literal["litellm", "heuristic"]]:
    mode_map: dict[str, TestOperation] = {
        "chat": "chat_completion",
        "chat_completion": "chat_completion",
        "responses": "responses",
        "embedding": "embedding",
        "embed": "embedding",
        "image_generation": "image_generation",
        "image": "image_generation",
        "audio_speech": "audio_speech",
        "speech": "audio_speech",
        "audio_transcription": "audio_transcription",
        "transcription": "audio_transcription",
        "moderation": "moderation",
        "completion": "text_completion",
        "text_completion": "text_completion",
        "video_generation": "video_generation",
    }
    if isinstance(mode, str) and mode.strip():
        m = mode.strip().lower()
        if m in mode_map:
            return mode_map[m], "heuristic"

    name = model.casefold().split("/", 1)[-1]
    if "embed" in name:
        return "embedding", "heuristic"

    if requested == "embedding":
        return "embedding", "heuristic"

    try:
        info = litellm.get_model_info(model)
        litellm_mode = info.get("mode")
        is_known = (
            model in litellm.model_cost
            or model.split("/", 1)[-1] in litellm.model_cost
            or info.get("key") in litellm.model_cost
        )
    except Exception:
        litellm_mode = None
        is_known = False

    if is_known and isinstance(litellm_mode, str) and litellm_mode in mode_map:
        return mode_map[litellm_mode], "litellm"
    if is_known and isinstance(litellm_mode, str):
        raise HTTPException(
            status_code=422,
            detail=f"Model mode '{litellm_mode}' does not support a bounded credential test",
        )

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


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8779, reload=True)


if __name__ == "__main__":
    run()

