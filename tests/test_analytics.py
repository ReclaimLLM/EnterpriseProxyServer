from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import analytics
from app.config import settings


def test_track_litellm_error_captures_with_filterable_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str, dict]] = []

    def mock_capture(distinct_id: str, event: str, properties: dict) -> None:
        captured.append((distinct_id, event, properties))

    monkeypatch.setattr(analytics, "capture", mock_capture)

    exc = ValueError("Model not found")
    analytics.track_litellm_error(
        exc,
        operation="chat_completions",
        model="openai/gpt-4o",
        provider="openai",
        status_code=404,
        org_id="org-123",
        org_slug="acme",
        user_id="usr-456",
        team_id="team-789",
    )

    assert len(captured) == 1
    distinct_id, event, props = captured[0]
    assert distinct_id == "usr-456"
    assert event == "litellm_error"
    assert props["tag"] == "litellm_exception"
    assert props["error_tag"] == "litellm_exception"
    assert props["operation"] == "chat_completions"
    assert props["model"] == "openai/gpt-4o"
    assert props["provider"] == "openai"
    assert props["status_code"] == 404
    assert props["org_id"] == "org-123"
    assert props["org_slug"] == "acme"
    assert props["user_id"] == "usr-456"
    assert props["team_id"] == "team-789"
    assert props["error_type"] == "ValueError"
    assert props["error_message"] == "Model not found"
    assert props["component"] == "enterprise_proxy"


def test_track_litellm_error_custom_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str, dict]] = []

    def mock_capture(distinct_id: str, event: str, properties: dict) -> None:
        captured.append((distinct_id, event, properties))

    monkeypatch.setattr(analytics, "capture", mock_capture)

    exc = RuntimeError("Timeout")
    analytics.track_litellm_error(
        exc,
        operation="responses",
        model="azure/gpt-4o",
        provider="azure_openai",
        tag="custom_litellm_tag",
    )

    assert len(captured) == 1
    _, _, props = captured[0]
    assert props["tag"] == "custom_litellm_tag"
    assert props["error_tag"] == "custom_litellm_tag"


def test_capture_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "posthog_key", "phc_test_key")

    class FaultyPostHog:
        @staticmethod
        def capture(*args, **kwargs):
            raise ConnectionError("Network down")

    monkeypatch.setattr(analytics, "posthog", FaultyPostHog)

    # Should not raise
    analytics.capture("distinct-1", "test_event", {"key": "val"})


def test_capture_disabled_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "posthog_key", None)
    monkeypatch.delenv("POSTHOG_KEY", raising=False)
    called = False

    class MockPostHog:
        @staticmethod
        def capture(*args, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr(analytics, "posthog", MockPostHog)
    analytics.capture("distinct-1", "test_event", {"key": "val"})
    assert not called


def test_capture_uses_env_var_when_settings_key_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "posthog_key", None)
    monkeypatch.setenv("POSTHOG_KEY", "phc_env_var_key")
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.i.posthog.com")

    captured = []

    class MockPostHog:
        project_api_key = None
        host = None
        disabled = True

        @classmethod
        def capture(cls, distinct_id, event, properties):
            captured.append((distinct_id, event, properties, cls.project_api_key, cls.host))

    monkeypatch.setattr(analytics, "posthog", MockPostHog)
    analytics.capture("user-abc", "custom_event", {"foo": "bar"})

    assert len(captured) == 1
    assert captured[0][0] == "user-abc"
    assert captured[0][1] == "custom_event"
    assert captured[0][3] == "phc_env_var_key"
    assert captured[0][4] == "https://eu.i.posthog.com"
