from __future__ import annotations

import logging
import os
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

LITELLM_ERROR_TAG = "litellm_exception"

try:
    import posthog

    _api_key = os.getenv("POSTHOG_KEY") or settings.posthog_key
    _host = os.getenv("POSTHOG_HOST") or settings.posthog_host or "https://us.i.posthog.com"
    if _api_key:
        posthog.project_api_key = _api_key
        posthog.host = _host
        logger.info("PostHog analytics enabled (host=%s)", _host)
    else:
        posthog.disabled = True
except ImportError:
    posthog = None


def capture(distinct_id: str, event: str, properties: dict[str, Any]) -> None:
    api_key = os.getenv("POSTHOG_KEY") or settings.posthog_key
    if not api_key or posthog is None:
        return
    try:
        host = os.getenv("POSTHOG_HOST") or settings.posthog_host or "https://us.i.posthog.com"
        if (
            getattr(posthog, "project_api_key", None) != api_key
            or getattr(posthog, "host", None) != host
        ):
            posthog.project_api_key = api_key
            posthog.host = host
            posthog.disabled = False
        posthog.capture(distinct_id, event, properties)
    except Exception:
        logger.debug("PostHog capture error (swallowed)", exc_info=True)


def track_litellm_error(
    exc: Exception,
    *,
    operation: str,
    model: str | None = None,
    provider: str | None = None,
    status_code: int | None = None,
    org_id: str | None = None,
    org_slug: str | None = None,
    user_id: str | None = None,
    team_id: str | None = None,
    tag: str = LITELLM_ERROR_TAG,
    extra: dict[str, Any] | None = None,
) -> None:
    """Capture a PostHog event for LiteLLM exceptions with filterable tags."""
    distinct_id = user_id or org_id or "enterprise_proxy"
    properties: dict[str, Any] = {
        "tag": tag,
        "error_tag": tag,
        "operation": operation,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:1000],
        "model": model,
        "provider": provider,
        "status_code": status_code,
        "org_id": org_id,
        "org_slug": org_slug,
        "team_id": team_id,
        "user_id": user_id,
        "component": "enterprise_proxy",
    }
    if extra:
        for k, v in extra.items():
            if k not in properties and v is not None:
                properties[k] = v

    capture(distinct_id, "litellm_error", properties)


def shutdown() -> None:
    """Flush queued PostHog events on proxy shutdown."""
    api_key = settings.posthog_key or os.getenv("POSTHOG_KEY")
    if not api_key or posthog is None:
        return
    try:
        posthog.shutdown(timeout=3)
    except Exception:
        logger.debug("PostHog shutdown error (swallowed)", exc_info=True)
