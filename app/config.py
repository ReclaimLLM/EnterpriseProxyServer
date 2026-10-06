from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROXY_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_PROJECT_ROOT / "local.env", _PROXY_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend_base_url: str = Field(..., validation_alias="BACKEND_SERVER")
    proxy_shared_secret: str = Field(
        ...,
        validation_alias="ENTERPRISE_PROXY_SHARED_SECRET",
    )
    supabase_project_ref: str | None = Field(default=None, validation_alias="SUPABASE_PROJECT_REF")
    supabase_service_role_key: str | None = Field(
        default=None,
        validation_alias="SUPABASE_SERVICE_ROLE_KEY",
    )
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    aws_endpoint_url: str | None = Field(default=None, validation_alias="AWS_ENDPOINT_URL")
    s3_bucket_us: str | None = Field(default=None, validation_alias="S3_BUCKET_US")
    s3_bucket_eu: str | None = Field(default=None, validation_alias="S3_BUCKET_EU")
    log_queue_path: Path = Field(
        default=Path("/var/lib/reclaimllm-enterprise-proxy/failed_logs.jsonl"),
        validation_alias="GATEWAY_LOG_QUEUE_PATH",
    )
    backend_timeout_seconds: float = Field(
        default=15.0,
        validation_alias="BACKEND_TIMEOUT_SECONDS",
    )
    flush_interval_seconds: float = Field(
        default=15.0,
        validation_alias="LOG_FLUSH_INTERVAL_SECONDS",
    )
    posthog_key: str | None = Field(default=None, validation_alias="POSTHOG_KEY")
    posthog_host: str = Field(
        default="https://us.i.posthog.com",
        validation_alias="POSTHOG_HOST",
    )

    @property
    def supabase_rest_url(self) -> str:
        if not self.supabase_project_ref:
            raise RuntimeError("SUPABASE_PROJECT_REF is not configured")
        return f"https://{self.supabase_project_ref}.supabase.co/rest/v1"

    def bucket_for(self, region: str) -> str:
        if not self.s3_bucket_us or not self.s3_bucket_eu:
            raise RuntimeError("S3 buckets are not configured")
        return self.s3_bucket_eu if region == "eu" else self.s3_bucket_us


settings = Settings()
